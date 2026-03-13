import argparse
import collections
import glob
import logging
import os
import time
from typing import Optional, Tuple, List

import cv2
from decord import VideoReader, cpu
import torch
import torch.nn.functional as F

import src.models.vision_transformer as vit
from src.models.attentive_pooler import AttentiveClassifier

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def preprocess_frames(frames: torch.Tensor) -> torch.Tensor:
    """Preprocess [T, H, W, C] float tensor into [1, C, T, H, W].

    Follows the eval transform from utils.py:
      1. Resize shortest side to int(resolution * 256/224)
      2. Center crop to (resolution, resolution)
      3. ImageNet normalization
    """
    short_side_size = int(224 * 256 / 224)

    frames = frames.permute(0, 3, 1, 2)
    h, w = frames.shape[2], frames.shape[3]
    if h < w:
        new_h = short_side_size
        new_w = int(w * short_side_size / h)
    else:
        new_w = short_side_size
        new_h = int(h * short_side_size / w)
    frames = F.interpolate(
        frames,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )

    # Center crop
    top = (new_h - 224) // 2
    left = (new_w - 224) // 2
    frames = frames[:, :, top : top + 224, left : left + 224]

    # Normalize (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std

    # [T, C, H, W]  ->  [1, C, T, H, W]
    frames = frames.permute(1, 0, 2, 3).unsqueeze(0)
    return frames


def load_video_frames(video_path: str) -> torch.Tensor:

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    frames_np = vr.get_batch(list(range(total))).asnumpy()
    frames = torch.from_numpy(frames_np).float() / 255.0

    total_frames = frames.shape[0]
    if total_frames >= 16:
        indices = torch.linspace(0, total_frames - 1, 16).long()
    else:
        # Repeat last frame to pad
        indices = list(range(total_frames)) + [total_frames - 1] * (
            16 - total_frames
        )
    frames = frames[indices]
    return preprocess_frames(frames)


def load_encoder(
    checkpoint_path: Optional[str] = None,
    uniform_power: bool = True,
    use_sdpa: bool = True,
    checkpoint_key: str = "target_encoder",
    device: str = "cpu",
) -> vit.VisionTransformer:
    encoder = vit.__dict__["vit_large"](
        img_size=224,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
    )

    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        logger.info(f"Loading pretrained checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        try:
            state_dict = checkpoint[checkpoint_key]
        except KeyError:
            state_dict = checkpoint.get("encoder", checkpoint)

        # Remove common prefixes added by DDP / wrapper modules
        state_dict = {
            k.replace("module.", "").replace("backbone.", ""): v
            for k, v in state_dict.items()
        }

        # Interpolate pos_embed if checkpoint was trained with
        # different num_frames / resolution
        if "pos_embed" in state_dict and encoder.pos_embed is not None:
            ckpt_pos = state_dict["pos_embed"]
            model_pos = encoder.pos_embed
            if ckpt_pos.shape != model_pos.shape:
                logger.info(
                    f"Interpolating pos_embed: "
                    f"{ckpt_pos.shape} -> {model_pos.shape}"
                )
                dim = ckpt_pos.shape[-1]
                # Infer checkpoint grid from the total patch count
                ckpt_N = ckpt_pos.shape[1]
                # Model grid
                gs = 224 // 16
                gd = 16 // 2
                # Checkpoint grid: same spatial size, solve for depth
                ckpt_gd = ckpt_N // (gs * gs)
                ckpt_pos = ckpt_pos.reshape(1, ckpt_gd, gs, gs, dim).permute(
                    0, 4, 1, 2, 3
                )
                ckpt_pos = torch.nn.functional.interpolate(
                    ckpt_pos,
                    size=(gd, gs, gs),
                    mode="trilinear",
                    align_corners=False,
                )
                state_dict["pos_embed"] = ckpt_pos.permute(
                    0, 2, 3, 4, 1
                ).reshape(1, -1, dim)

        msg = encoder.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pretrained encoder  (msg: {msg})")
        if "epoch" in checkpoint:
            logger.info(f"  checkpoint epoch: {checkpoint['epoch']}")
        del checkpoint
    else:
        logger.warning("No checkpoint loaded - running with random weights!")

    encoder = encoder.to(device)
    encoder.eval()
    encoder.half()  # FP16 inference to reduce GPU memory
    return encoder


def load_classifier(
    checkpoint_path: str,
    embed_dim: int = 1024,
    num_heads: int = 16,
    depth: int = 1,
    device: str = "cpu",
) -> Tuple[AttentiveClassifier, int]:

    logger.info(f"Loading classifier probe from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("classifier", ckpt)

    # Strip DDP "module." prefix
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # Auto-detect num_classes from linear layer
    num_classes = state["linear.weight"].shape[0]
    logger.info(f"  detected num_classes = {num_classes}")

    classifier = AttentiveClassifier(
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        num_classes=num_classes,
    )

    msg = classifier.load_state_dict(state, strict=False)
    logger.info(f"Loaded classifier probe  (msg: {msg})")
    del ckpt

    classifier = classifier.to(device)
    classifier.eval()
    classifier.half()  # FP16 inference to reduce GPU memory
    return classifier, num_classes


def predict_action(
    features: torch.Tensor,
    classifier: AttentiveClassifier,
    labels: List[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    with torch.no_grad():
        logits = classifier(features)
        probs = torch.softmax(logits, dim=-1)

    # Take the first sample in the batch
    probs = probs[0]
    top_values, top_indices = probs.topk(top_k)

    predictions = []
    for score, idx in zip(top_values, top_indices):
        predictions.append((labels[int(idx.item())], float(score.item())))
    return predictions


def stream_predict(
    video_source: str,
    encoder: vit.VisionTransformer,
    classifier: AttentiveClassifier,
    labels: List[str],
    top_k: int = 5,
    device: str = "cpu",
    output_video: Optional[str] = None,
):
    src = 0 if video_source == "webcam" else video_source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_webcam = video_source == "webcam"
    source_label = "webcam" if is_webcam else video_source

    if not is_webcam and total > 0:
        logger.info(f"  total frames: {total}")

    buf = collections.deque(maxlen=16)
    frame_idx = 0
    window_count = 0
    current_label = ""
    current_conf = 0.0
    class_counts = collections.Counter()

    display_lines = top_k + 4

    writer = None
    if output_video is not None:
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_video,
            fourcc,
            fps,
            (frame_w, frame_h),
        )
        logger.info(f"Writing annotated video to: {output_video}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_webcam:
                    continue
                break

            # BGR -> RGB, uint8 -> float
            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )
            frame_t = torch.from_numpy(frame_rgb).float() / 255.0
            buf.append(frame_t)
            frame_idx += 1

            # Write annotated frame to output video
            if writer is not None and current_label:
                out_frame = frame.copy()
                h, w = out_frame.shape[:2]
                # Semi-transparent dark banner
                overlay = out_frame.copy()
                cv2.rectangle(
                    overlay,
                    (0, h - 70),
                    (w, h),
                    (0, 0, 0),
                    -1,
                )
                cv2.addWeighted(
                    overlay,
                    0.6,
                    out_frame,
                    0.4,
                    0,
                    out_frame,
                )
                # Action label text
                text = f"{current_label}  {current_conf * 100:.1f}%"
                cv2.putText(
                    out_frame,
                    text,
                    (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(out_frame)
            elif writer is not None:
                writer.write(frame)

            # Wait until buffer is full, then predict every 16 frames
            if len(buf) == 16 and (frame_idx - 16) % 16 == 0:
                clip = torch.stack(list(buf))  # [T,H,W,C]
                clip = preprocess_frames(clip).to(
                    device=device, dtype=torch.float16
                )

                t0 = time.time()
                with torch.no_grad():
                    feats = encoder(clip)
                preds = predict_action(
                    feats,
                    classifier,
                    labels,
                    top_k,
                )
                dt = time.time() - t0
                del clip, feats
                torch.cuda.empty_cache()

                window_count += 1
                current_label = preds[0][0]
                current_conf = preds[0][1]
                class_counts[current_label] += 1

                # Overwrite previous output
                if window_count > 1:
                    print(
                        f"\033[{display_lines}A",
                        end="",
                    )

                time_sec = frame_idx / fps
                print(f"\033[K  [{source_label}]  frame {frame_idx}")
                if not is_webcam:
                    print(f"\033[K  time {time_sec:.1f}s  ({dt:.2f}s/window)")
                else:
                    print(f"\033[K  ({dt:.2f}s/window)")
                print("\033[K  " + "-" * 46)
                for rank, (name, conf) in enumerate(
                    preds,
                    1,
                ):
                    bar = "\u2588" * int(conf * 30)
                    print(
                        f"\033[K  {rank}. {name:36s} {conf * 100:5.1f}% {bar}"
                    )
                print("\033[K  " + "-" * 46)

    except KeyboardInterrupt:
        print("\n  Stopped by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            logger.info(f"Output video saved to: {output_video}")

    logger.info(
        f"Streaming finished. "
        f"Processed {frame_idx} frames, "
        f"{window_count} windows."
    )

    # Summary: most common predicted class
    if class_counts:
        top_class, top_count = class_counts.most_common(1)[0]
        total_w = sum(class_counts.values())
        print("\n" + "=" * 50)
        print("  Overall Summary")
        print("=" * 50)
        print(
            f"  Most predicted action: {top_class}  "
            f"({top_count}/{total_w} windows, "
            f"{top_count / total_w * 100:.0f}%)"
        )
        print("  All classes seen:")
        for cls, cnt in class_counts.most_common():
            pct = cnt / total_w * 100
            bar = "\u2588" * int(pct / 3)
            print(f"    {cls:36s} {cnt:3d}x  {pct:4.0f}% {bar}")
        print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video file / video directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pretrained/vitl16.pth.tar",
        help="Path to pretrained V-JEPA checkpoint (.pth.tar)",
    )
    parser.add_argument(
        "--checkpoint-key",
        type=str,
        default="target_encoder",
        help="Key in checkpoint dict that holds the encoder weights",
    )
    parser.add_argument(
        "--probe-checkpoint",
        type=str,
        default="pretrained/custom-jepa.pth.tar",
        help="Path to a trained classification probe (.pth.tar) ",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of top action predictions to show",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming mode: slide a window "
        "through the video (or use 'webcam')",
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load model
    encoder = load_encoder(
        checkpoint_path=args.checkpoint,
        checkpoint_key=args.checkpoint_key,
        device=device,
    )

    # Load classifier (if probe provided)
    classifier = None
    labels = None
    if args.probe_checkpoint:
        classifier, num_classes = load_classifier(
            checkpoint_path=args.probe_checkpoint,
            embed_dim=encoder.embed_dim,
            num_heads=encoder.num_heads,
            device=device,
        )
        if num_classes == 2:
            labels = ["normal", "anomaly"]
        else:
            labels = [f"class_{i}" for i in range(num_classes)]

    if args.video is None:
        raise SystemExit("Error: --video is required")

    if os.path.isdir(args.video):
        video_paths = glob.glob(os.path.join(args.video, "*.mp4"))
    else:
        video_paths = [args.video]

    for video_path in video_paths:
        if args.stream and (classifier is not None) and (labels is not None):
            output_video = os.path.join(
                os.path.dirname(video_path),
                "output_" + os.path.basename(video_path),
            )
            stream_predict(
                video_source=video_path,
                encoder=encoder,
                classifier=classifier,
                labels=labels,
                top_k=args.top_k,
                device=device,
                output_video=output_video,
            )
            continue

        logger.info(f"Loading video: {args.video}")
        video = load_video_frames(args.video)
        video = video.to(device=device, dtype=torch.float16)
        logger.info(
            f"Video tensor shape: {video.shape}  "
            f"(batch, channels, frames, height, width)"
        )

        logger.info("Running inference...")
        with torch.no_grad():
            features = encoder(video)
        del video
        features = features.cpu()
        torch.cuda.empty_cache()

        if classifier is not None and labels is not None:
            predictions = predict_action(
                features,
                classifier,
                labels,
                top_k=args.top_k,
            )
            print("\n" + "=" * 50)
            print("  Action Predictions")
            print("=" * 50)
            for rank, (name, conf) in enumerate(predictions, 1):
                bar = "\u2588" * int(conf * 30)
                print(f"  {rank}. {name:40s} {conf * 100:5.1f}% {bar}")
            print("=" * 50 + "\n")
        else:
            logger.info("Tip: pass --probe-checkpoint to predict actions.")

        if args.output:
            torch.save(features.cpu(), args.output)
            logger.info(f"Features saved to: {args.output}")


if __name__ == "__main__":
    main()
