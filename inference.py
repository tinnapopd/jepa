# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Standalone V-JEPA inference script for a single video file.
# Usage:
#   python inference.py --video sample.mp4 --checkpoint vitl16.pth.tar
#
# This script loads the pretrained V-JEPA encoder and extracts
# feature representations from the input video.

import argparse
import collections
import json
import logging
import os
import time

import torch
import torch.nn.functional as F
import numpy as np

import src.models.vision_transformer as vit
from src.models.attentive_pooler import AttentiveClassifier

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ------------------------------------------------------------------ #
#  Video loading & preprocessing
# ------------------------------------------------------------------ #

def preprocess_frames(frames, resolution=224):
    """Preprocess a [T, H, W, C] float tensor into [1, C, T, H, W].

    Applies spatial resize (shortest side), center-crop, and
    ImageNet normalization.
    """
    # --- Spatial: resize shortest side, then center-crop ---
    frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
    h, w = frames.shape[2], frames.shape[3]
    if h < w:
        new_h = resolution
        new_w = int(w * resolution / h)
    else:
        new_w = resolution
        new_h = int(h * resolution / w)
    frames = F.interpolate(
        frames, size=(new_h, new_w),
        mode="bilinear", align_corners=False,
    )

    # Center crop
    top = (new_h - resolution) // 2
    left = (new_w - resolution) // 2
    frames = frames[
        :, :, top:top + resolution, left:left + resolution
    ]

    # --- Normalize (ImageNet stats) ---
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std

    # [T, C, H, W]  ->  [1, C, T, H, W]
    frames = frames.permute(1, 0, 2, 3).unsqueeze(0)
    return frames


def load_video_frames(video_path, num_frames=16, resolution=224):
    """
    Decode a video file and return a tensor of shape [1, 3, T, H, W].

    Tries torchvision.io first (requires torchvision with video backend),
    then falls back to decord, and finally to OpenCV.
    """
    frames = None

    # --- Attempt 1: torchvision ---
    try:
        import torchvision.io as tio
        video_tensor, _, meta = tio.read_video(
            video_path, pts_unit="sec",
        )
        # video_tensor: [T_total, H, W, C] uint8
        frames = video_tensor.float() / 255.0
        logger.info(
            f"Loaded video with torchvision  "
            f"({frames.shape[0]} frames)"
        )
    except Exception:
        pass

    # --- Attempt 2: decord ---
    if frames is None:
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total = len(vr)
            frames_np = vr.get_batch(
                list(range(total))
            ).asnumpy()  # [T, H, W, C]
            frames = (
                torch.from_numpy(frames_np).float() / 255.0
            )
            logger.info(
                f"Loaded video with decord  "
                f"({frames.shape[0]} frames)"
            )
        except Exception:
            pass

    # --- Attempt 3: OpenCV ---
    if frames is None:
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frame_list = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_list.append(frame)
            cap.release()
            if len(frame_list) == 0:
                raise RuntimeError(
                    f"Could not read any frames "
                    f"from {video_path}"
                )
            frames = (
                torch.from_numpy(
                    np.stack(frame_list)
                ).float() / 255.0
            )
            logger.info(
                f"Loaded video with OpenCV  "
                f"({frames.shape[0]} frames)"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load video '{video_path}'. "
                "Install one of: torchvision "
                "(with video backend), "
                "decord, or opencv-python.\n"
                f"  Error: {e}"
            )

    # --- Temporally sample `num_frames` frames uniformly ---
    total_frames = frames.shape[0]
    if total_frames >= num_frames:
        indices = np.linspace(
            0, total_frames - 1, num_frames, dtype=int,
        )
    else:
        # Repeat last frame to pad
        indices = (
            list(range(total_frames))
            + [total_frames - 1]
            * (num_frames - total_frames)
        )
    frames = frames[indices]  # [T, H, W, C]

    return preprocess_frames(frames, resolution)


# ------------------------------------------------------------------ #
#  Model utilities
# ------------------------------------------------------------------ #

def load_encoder(
    model_name="vit_large",
    checkpoint_path=None,
    patch_size=16,
    num_frames=16,
    tubelet_size=2,
    resolution=224,
    uniform_power=True,
    use_sdpa=True,
    checkpoint_key="target_encoder",
    device="cpu",
):
    """Build a ViT encoder and load pretrained V-JEPA weights."""

    encoder = vit.__dict__[model_name](
        img_size=resolution,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
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
                gs = resolution // patch_size
                gd = num_frames // tubelet_size
                # Checkpoint grid: same spatial size, solve for depth
                ckpt_gd = ckpt_N // (gs * gs)
                ckpt_pos = ckpt_pos.reshape(
                    1, ckpt_gd, gs, gs, dim
                ).permute(0, 4, 1, 2, 3)   # [1, D, T', H', W']
                ckpt_pos = torch.nn.functional.interpolate(
                    ckpt_pos,
                    size=(gd, gs, gs),
                    mode="trilinear",
                    align_corners=False,
                )
                state_dict["pos_embed"] = (
                    ckpt_pos.permute(0, 2, 3, 4, 1)
                    .reshape(1, -1, dim)
                )

        msg = encoder.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pretrained encoder  (msg: {msg})")
        if "epoch" in checkpoint:
            logger.info(f"  checkpoint epoch: {checkpoint['epoch']}")
        del checkpoint
    else:
        logger.warning("No checkpoint loaded – running with random weights!")

    encoder = encoder.to(device)
    encoder.eval()
    return encoder


def load_classifier(
    checkpoint_path,
    embed_dim=1024,
    num_heads=16,
    depth=1,
    device="cpu",
):
    """Build an AttentiveClassifier and load trained probe weights.

    The number of classes is auto-detected from the checkpoint.

    Returns:
        (classifier, num_classes) tuple.
    """
    logger.info(
        f"Loading classifier probe from: {checkpoint_path}"
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("classifier", ckpt)

    # Strip DDP "module." prefix
    state = {
        k.replace("module.", ""): v
        for k, v in state.items()
    }

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
    return classifier, num_classes


def predict_action(features, classifier, labels, top_k=5):
    """Run the classifier on encoder features and return top-K
    predicted actions with confidence scores.

    Args:
        features: Tensor of shape [B, N, D] from the encoder.
        classifier: An AttentiveClassifier instance.
        labels: List of class name strings.
        top_k: Number of top predictions to return.

    Returns:
        List of (class_name, confidence) tuples.
    """
    with torch.no_grad():
        logits = classifier(features)          # [B, num_classes]
        probs = torch.softmax(logits, dim=-1)  # [B, num_classes]

    # Take the first sample in the batch
    probs = probs[0]
    top_values, top_indices = probs.topk(top_k)

    predictions = []
    for score, idx in zip(top_values, top_indices):
        predictions.append((labels[idx.item()], score.item()))
    return predictions


# ------------------------------------------------------------------ #
#  Streaming prediction
# ------------------------------------------------------------------ #

def stream_predict(
    video_source,
    encoder,
    classifier,
    labels,
    num_frames=16,
    resolution=224,
    stride=8,
    top_k=5,
    device="cpu",
    output_video=None,
):
    """Stream through a video file (or webcam) and predict
    actions on a sliding window of frames.

    Args:
        video_source: path to video file, or 0 for webcam.
        encoder: pretrained V-JEPA encoder.
        classifier: trained AttentiveClassifier probe.
        labels: list of class name strings.
        num_frames: frames per window (must match encoder).
        resolution: spatial crop size.
        stride: predict every N new frames.
        top_k: number of top predictions to display.
        device: torch device.
        output_video: if set, path to save annotated output
            video with predictions overlaid.
    """
    import cv2

    src = 0 if video_source == "webcam" else video_source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video source: {video_source}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_webcam = (video_source == "webcam")
    source_label = "webcam" if is_webcam else video_source

    logger.info(
        f"Streaming from {source_label}  "
        f"(fps={fps:.1f}, stride={stride})"
    )
    if not is_webcam and total > 0:
        logger.info(f"  total frames: {total}")

    buf = collections.deque(maxlen=num_frames)
    frame_idx = 0
    window_count = 0
    current_label = ""
    current_conf = 0.0
    class_counts = collections.Counter()

    # Number of header + prediction lines to overwrite
    display_lines = top_k + 4

    # Video writer for annotated output
    writer = None
    if output_video is not None:
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_video, fourcc, fps, (frame_w, frame_h),
        )
        logger.info(
            f"Writing annotated video to: {output_video}"
        )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_webcam:
                    continue
                break

            # BGR -> RGB, uint8 -> float
            frame_rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB,
            )
            frame_t = (
                torch.from_numpy(frame_rgb).float() / 255.0
            )
            buf.append(frame_t)
            frame_idx += 1

            # Write annotated frame to output video
            if writer is not None and current_label:
                out_frame = frame.copy()
                h, w = out_frame.shape[:2]
                # Semi-transparent dark banner
                overlay = out_frame.copy()
                cv2.rectangle(
                    overlay, (0, h - 70), (w, h),
                    (0, 0, 0), -1,
                )
                cv2.addWeighted(
                    overlay, 0.6, out_frame, 0.4,
                    0, out_frame,
                )
                # Action label text
                text = (
                    f"{current_label}  "
                    f"{current_conf*100:.1f}%"
                )
                cv2.putText(
                    out_frame, text,
                    (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 2,
                    cv2.LINE_AA,
                )
                writer.write(out_frame)
            elif writer is not None:
                writer.write(frame)

            # Wait until buffer is full, then predict
            # every `stride` frames
            if (
                len(buf) == num_frames
                and (frame_idx - num_frames) % stride == 0
            ):
                clip = torch.stack(list(buf))  # [T,H,W,C]
                clip = preprocess_frames(
                    clip, resolution,
                ).to(device)

                t0 = time.time()
                with torch.no_grad():
                    feats = encoder(clip)
                preds = predict_action(
                    feats, classifier, labels, top_k,
                )
                dt = time.time() - t0

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
                print(
                    f"\033[K  [{source_label}]  "
                    f"frame {frame_idx}"
                )
                if not is_webcam:
                    print(
                        f"\033[K  time {time_sec:.1f}s  "
                        f"({dt:.2f}s/window)"
                    )
                else:
                    print(
                        f"\033[K  ({dt:.2f}s/window)"
                    )
                print("\033[K  " + "-" * 46)
                for rank, (name, conf) in enumerate(
                    preds, 1,
                ):
                    bar = "\u2588" * int(conf * 30)
                    print(
                        f"\033[K  {rank}. "
                        f"{name:36s} "
                        f"{conf*100:5.1f}% {bar}"
                    )
                print("\033[K  " + "-" * 46)

    except KeyboardInterrupt:
        print("\n  Stopped by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            logger.info(
                f"Output video saved to: {output_video}"
            )

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
            f"{top_count/total_w*100:.0f}%)"
        )
        print("  All classes seen:")
        for cls, cnt in class_counts.most_common():
            pct = cnt / total_w * 100
            bar = "\u2588" * int(pct / 3)
            print(f"    {cls:36s} {cnt:3d}x  {pct:4.0f}% {bar}")
        print("=" * 50 + "\n")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="V-JEPA single-video inference")
    parser.add_argument("--video", type=str, default="sample.mp4",
                        help="Path to input video file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained V-JEPA checkpoint (.pth.tar)")
    parser.add_argument("--model-name", type=str, default="vit_large",
                        choices=["vit_tiny", "vit_small", "vit_base",
                                 "vit_large", "vit_huge", "vit_giant"],
                        help="ViT model architecture")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=64,
                        help="Number of frames to sample from the video")
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=224,
                        help="Spatial crop resolution")
    parser.add_argument("--checkpoint-key", type=str, default="target_encoder",
                        help="Key in checkpoint dict that holds the encoder weights")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: cuda if available, else cpu)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save the feature tensor (.pt)")
    parser.add_argument(
        "--probe-checkpoint", type=str, default=None,
        help="Path to a trained classification probe (.pth.tar) "
             "for action prediction (e.g. k400-probe.pth.tar)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top action predictions to show",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Enable streaming mode: slide a window "
             "through the video (or use 'webcam')",
    )
    parser.add_argument(
        "--stride", type=int, default=8,
        help="Frames between predictions in stream mode",
    )
    parser.add_argument(
        "--output-video", type=str, default=None,
        help="Path to save annotated output video with "
             "predictions overlaid (.mp4)",
    )
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load model ----
    encoder = load_encoder(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        patch_size=args.patch_size,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        resolution=args.resolution,
        checkpoint_key=args.checkpoint_key,
        device=device,
    )

    # ---- Load classifier (if probe provided) ----
    classifier = None
    labels = None
    if args.probe_checkpoint:
        classifier, num_classes = load_classifier(
            checkpoint_path=args.probe_checkpoint,
            embed_dim=encoder.embed_dim,
            num_heads=encoder.num_heads,
            device=device,
        )

        # Auto-select label file based on num_classes
        labels_map = {
            174: "ssv2_labels.json",
            400: "kinetics400_labels.json",
            1000: "imagenet1k_labels.json",
        }
        script_dir = os.path.dirname(__file__)
        label_file = labels_map.get(num_classes)
        if label_file is not None:
            lpath = os.path.join(script_dir, label_file)
            with open(lpath) as f:
                labels = json.load(f)
            logger.info(
                f"Loaded {len(labels)} labels "
                f"from {label_file}"
            )
        else:
            labels = [
                f"class_{i}" for i in range(num_classes)
            ]
            logger.warning(
                f"No label file for {num_classes} "
                f"classes — using numeric labels"
            )

    # ================================================================
    #  Streaming mode
    # ================================================================
    if args.stream:
        if classifier is None:
            raise SystemExit(
                "Error: --stream requires "
                "--probe-checkpoint"
            )
        stream_predict(
            video_source=args.video,
            encoder=encoder,
            classifier=classifier,
            labels=labels,
            num_frames=args.num_frames,
            resolution=args.resolution,
            stride=args.stride,
            top_k=args.top_k,
            device=device,
            output_video=args.output_video,
        )
        return

    # ================================================================
    #  Batch mode (original behavior)
    # ================================================================
    logger.info(f"Loading video: {args.video}")
    video = load_video_frames(
        args.video,
        num_frames=args.num_frames,
        resolution=args.resolution,
    )
    video = video.to(device)
    logger.info(
        f"Video tensor shape: {video.shape}  "
        f"(batch, channels, frames, height, width)"
    )

    # ---- Run inference ----
    logger.info("Running inference...")
    with torch.no_grad():
        features = encoder(video)

    logger.info(f"Output features shape: {features.shape}")
    logger.info(
        f"  num_patches = {features.shape[1]}  "
        f"(= (T/tubelet) \u00d7 (H/patch) \u00d7 (W/patch) "
        f"= {args.num_frames // args.tubelet_size}"
        f" \u00d7 {args.resolution // args.patch_size}"
        f" \u00d7 {args.resolution // args.patch_size})"
    )
    logger.info(f"  embed_dim   = {features.shape[2]}")

    global_feat = features.mean(dim=1)
    logger.info(
        f"Global (mean-pooled) feature shape: "
        f"{global_feat.shape}"
    )

    # ---- Action prediction (batch) ----
    if classifier is not None:
        predictions = predict_action(
            features, classifier, labels,
            top_k=args.top_k,
        )
        print("\n" + "=" * 50)
        print("  Action Predictions (Kinetics-400)")
        print("=" * 50)
        for rank, (name, conf) in enumerate(
            predictions, 1,
        ):
            bar = "\u2588" * int(conf * 30)
            print(
                f"  {rank}. {name:40s} "
                f"{conf*100:5.1f}% {bar}"
            )
        print("=" * 50 + "\n")
    else:
        logger.info(
            "Tip: pass --probe-checkpoint "
            "to predict actions."
        )

    # ---- Optionally save ----
    if args.output:
        torch.save(features.cpu(), args.output)
        logger.info(f"Features saved to: {args.output}")

    return features


if __name__ == "__main__":
    main()
