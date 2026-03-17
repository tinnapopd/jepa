#!/usr/bin/env python3
"""
Anomaly Detection Model Evaluation Script
==========================================
Evaluates model anomaly scores against ground-truth labels to prove the model
is not producing random results.

Usage:
    python evaluate.py [--test-dataset-dir TEST_DATASET_DIR]
                       [--threshold THRESHOLD]
                       [--n-permutations N_PERMUTATIONS]
                       [--output-dir OUTPUT_DIR]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_labels(label_path: str) -> list[tuple[float, float]]:
    """Load ground-truth anomaly intervals from a CSV label file.

    Each non-empty line: start_sec, end_sec
    Returns a list of (start, end) tuples.  Empty file → empty list (normal).
    """
    intervals = []
    with open(label_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # skip empty rows / trailing newlines
            stripped = [c.strip() for c in row if c.strip()]
            if len(stripped) >= 2:
                intervals.append((float(stripped[0]), float(stripped[1])))
    return intervals


def load_scores(score_path: str) -> dict:
    """Load anomaly score JSON for a single video.

    Returns the parsed dict with keys: video_source, total_frames,
    total_windows, fps, scores[{frame, anomaly_score}].
    """
    with open(score_path, "r") as f:
        return json.load(f)


def is_in_anomaly(
    time_sec: float, intervals: list[tuple[float, float]]
) -> bool:
    """Return True if *time_sec* falls within any ground-truth interval."""
    for start, end in intervals:
        if start <= time_sec <= end:
            return True
    return False


# ---------------------------------------------------------------------------
# Dataset collection
# ---------------------------------------------------------------------------


def collect_dataset(
    test_dataset_dir: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Walk every subfolder and build parallel arrays of scores and labels.

    Returns:
        all_scores  – (N,) float array of anomaly scores
        all_labels  – (N,) int   array of binary ground-truth labels (0/1)
        per_video   – list of dicts with per-video metadata for reporting
    """
    all_scores: list[float] = []
    all_labels: list[int] = []
    per_video: list[dict] = []

    base = Path(test_dataset_dir)
    for subfolder in sorted(base.iterdir()):
        if not subfolder.is_dir() or subfolder.name.startswith("."):
            continue
        labels_dir = subfolder / "labels"
        videos_dir = subfolder / "videos"
        if not labels_dir.exists() or not videos_dir.exists():
            continue

        for label_file in sorted(labels_dir.glob("video_*.csv")):
            video_name = label_file.stem  # e.g. "video_1"
            score_file = videos_dir / f"anomaly_scores_{video_name}.json"
            if not score_file.exists():
                continue

            intervals = load_labels(str(label_file))
            score_data = load_scores(str(score_file))
            fps = score_data["fps"]

            v_scores = []
            v_labels = []
            for entry in score_data["scores"]:
                frame = entry["frame"]
                score = entry["anomaly_score"]
                time_sec = frame / fps
                label = 1 if is_in_anomaly(time_sec, intervals) else 0
                v_scores.append(score)
                v_labels.append(label)

            all_scores.extend(v_scores)
            all_labels.extend(v_labels)

            per_video.append(
                {
                    "subfolder": subfolder.name,
                    "video": video_name,
                    "fps": fps,
                    "total_frames": score_data["total_frames"],
                    "n_windows": len(v_scores),
                    "n_anomalous": sum(v_labels),
                    "intervals": intervals,
                    "scores": v_scores,
                    "labels": v_labels,
                    "frames": [e["frame"] for e in score_data["scores"]],
                }
            )

    return np.array(all_scores), np.array(all_labels), per_video


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict:
    """Compute all standard binary-classification metrics."""
    preds = (scores >= threshold).astype(int)

    metrics: dict = {}

    # AUC-ROC  (needs both classes present)
    if len(np.unique(labels)) == 2:
        metrics["auc_roc"] = roc_auc_score(labels, scores)
        metrics["avg_precision"] = average_precision_score(labels, scores)
    else:
        metrics["auc_roc"] = float("nan")
        metrics["avg_precision"] = float("nan")

    metrics["precision"] = precision_score(labels, preds, zero_division=0)
    metrics["recall"] = recall_score(labels, preds, zero_division=0)
    metrics["f1"] = f1_score(labels, preds, zero_division=0)
    metrics["confusion_matrix"] = confusion_matrix(
        labels,  # type: ignore
        preds,  # type: ignore
        labels=[0, 1],
    )
    return metrics


def permutation_test(
    scores: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 1000,
    rng_seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Compare real AUC to a null distribution of shuffled scores.

    Returns:
        real_auc          - AUC on actual data
        p_value           - fraction of permuted AUCs >= real AUC
        permuted_aucs     - array of AUC values from permutations
    """
    rng = np.random.default_rng(rng_seed)

    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan"), np.array([])

    real_auc = roc_auc_score(labels, scores)

    permuted_aucs = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(scores)
        permuted_aucs[i] = roc_auc_score(labels, shuffled)

    p_value = np.mean(permuted_aucs >= real_auc)
    return real_auc, p_value, permuted_aucs  # type: ignore


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_roc_curve(
    scores: np.ndarray, labels: np.ndarray, output_path: str
) -> None:
    """Save ROC-curve plot with AUC annotation and random-chance diagonal."""
    if len(np.unique(labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(labels, scores)  # type: ignore
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"Model (AUC = {roc_auc:.4f})")  # type: ignore
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Anomaly Detection")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(
    scores: np.ndarray, labels: np.ndarray, output_path: str
) -> None:
    """Save Precision-Recall curve plot."""
    if len(np.unique(labels)) < 2:
        return
    precision, recall, _ = precision_recall_curve(labels, scores)  # type: ignore
    ap = average_precision_score(labels, scores)
    baseline = labels.sum() / len(labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, lw=2, label=f"Model (AP = {ap:.4f})")  # type: ignore
    ax.axhline(
        baseline,
        color="k",
        ls="--",
        lw=1,
        label=f"Random baseline ({baseline:.4f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_score_distribution(
    scores: np.ndarray, labels: np.ndarray, output_path: str
) -> None:
    """Histogram comparing anomaly-score distributions of the two classes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        scores[labels == 0],
        bins=30,
        alpha=0.6,
        label="Normal windows",
        color="#2196F3",
        density=True,
    )
    ax.hist(
        scores[labels == 1],
        bins=30,
        alpha=0.6,
        label="Anomalous windows",
        color="#F44336",
        density=True,
    )
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution — Normal vs. Anomalous")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_permutation_test(
    real_auc: float,
    permuted_aucs: np.ndarray,
    p_value: float,
    output_path: str,
) -> None:
    """Histogram of null-distribution AUCs with the real AUC marked."""
    if len(permuted_aucs) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        permuted_aucs,
        bins=40,
        alpha=0.7,
        color="#9E9E9E",
        label="Permuted AUCs (null)",
    )
    ax.axvline(
        real_auc,
        color="#F44336",
        lw=2,
        ls="--",
        label=f"Real AUC = {real_auc:.4f}",
    )
    ax.set_xlabel("AUC-ROC")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation Test  (p = {p_value:.4f})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_timeline_batch(
    videos: list[dict], title: str, output_path: str
) -> None:
    """Plot a batch of video timelines into a single figure."""
    n = len(videos)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, v in zip(axes, videos):
        times = [f / v["fps"] for f in v["frames"]]
        ax.plot(
            times,
            v["scores"],
            "o-",
            markersize=4,
            color="#1976D2",
            label="Anomaly Score",
        )
        for start, end in v["intervals"]:
            ax.axvspan(
                start, end, alpha=0.25, color="#F44336", label="Ground Truth"
            )
        ax.axhline(0.5, ls=":", color="gray", lw=1)
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel("Score")
        ax.set_title(f"{v['subfolder']} / {v['video']}")
        # deduplicate legend entries
        handles, lbls = ax.get_legend_handles_labels()
        by_label = dict(zip(lbls, handles))
        ax.legend(
            by_label.values(), by_label.keys(), loc="upper right", fontsize=8
        )
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(title, y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sample_timelines(
    per_video: list[dict], output_path: str, max_videos: int = 6
) -> None:
    """Plot anomaly scores over time for a sample of videos."""
    anomalous = [v for v in per_video if v["n_anomalous"] > 0]
    sample = anomalous[:max_videos] if anomalous else per_video[:max_videos]
    _plot_timeline_batch(
        sample, "Sample Video Timelines — Score vs Ground Truth", output_path
    )


def plot_all_timelines(
    per_video: list[dict], output_dir: str, per_page: int = 5
) -> list[str]:
    """Save timeline plots for ALL videos, grouped by subfolder.

    Videos are batched *per_page* per image file.  Files are saved to
    ``output_dir/timelines/<subfolder>/page_N.png``.

    Returns a list of all saved file paths.
    """
    saved: list[str] = []
    subfolders = sorted(set(v["subfolder"] for v in per_video))

    for sf in subfolders:
        sf_videos = sorted(
            [v for v in per_video if v["subfolder"] == sf],
            key=lambda v: int(v["video"].split("_")[-1]),
        )
        sf_dir = os.path.join(output_dir, "timelines", sf)
        os.makedirs(sf_dir, exist_ok=True)

        # Batch into pages of `per_page` videos each
        for page_idx in range(0, len(sf_videos), per_page):
            batch = sf_videos[page_idx : page_idx + per_page]
            page_num = page_idx // per_page + 1
            first_vid = batch[0]["video"]
            last_vid = batch[-1]["video"]
            fname = f"page_{page_num}.png"
            fpath = os.path.join(sf_dir, fname)
            title = f"{sf}  ({first_vid} – {last_vid})"
            _plot_timeline_batch(batch, title, fpath)
            saved.append(fpath)

    return saved


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_section(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def report(
    all_scores: np.ndarray,
    all_labels: np.ndarray,
    per_video: list[dict],
    threshold: float,
    n_permutations: int,
    output_dir: str,
) -> None:
    """Print evaluation report and save all plots."""
    os.makedirs(output_dir, exist_ok=True)

    # ---- Data Summary ----
    print_section("DATA SUMMARY")
    n_total = len(all_scores)
    n_pos = int(all_labels.sum())
    n_neg = n_total - n_pos
    subfolders = sorted(set(v["subfolder"] for v in per_video))
    print(f"  Subfolders       : {subfolders}")
    print(f"  Total videos     : {len(per_video)}")
    print(f"  Total windows    : {n_total}")
    print(f"  Anomalous windows: {n_pos}  ({100 * n_pos / n_total:.1f}%)")
    print(f"  Normal windows   : {n_neg}  ({100 * n_neg / n_total:.1f}%)")

    # ---- Overall Metrics ----
    print_section("OVERALL METRICS")
    m = compute_metrics(all_scores, all_labels, threshold)
    print(f"  AUC-ROC          : {m['auc_roc']:.4f}")
    print(f"  Avg Precision (AP): {m['avg_precision']:.4f}")
    print(f"  Precision @{threshold}: {m['precision']:.4f}")
    print(f"  Recall    @{threshold}: {m['recall']:.4f}")
    print(f"  F1        @{threshold}: {m['f1']:.4f}")
    cm = m["confusion_matrix"]
    print(f"\n  Confusion Matrix (threshold={threshold}):")
    print("                  Pred Normal  Pred Anomaly")
    print(f"   Actual Normal      {cm[0][0]:>5d}         {cm[0][1]:>5d}")
    print(f"   Actual Anomaly     {cm[1][0]:>5d}         {cm[1][1]:>5d}")

    # ---- Per-Subfolder Metrics ----
    print_section("PER-SUBFOLDER METRICS")
    for sf in subfolders:
        vids = [v for v in per_video if v["subfolder"] == sf]
        sf_scores = np.concatenate([np.array(v["scores"]) for v in vids])
        sf_labels = np.concatenate([np.array(v["labels"]) for v in vids])
        sm = compute_metrics(sf_scores, sf_labels, threshold)
        n_vids = len(vids)
        n_w = len(sf_scores)
        n_p = int(sf_labels.sum())
        print(f"\n  [{sf}]  ({n_vids} videos, {n_w} windows, {n_p} anomalous)")
        print(
            f"    AUC-ROC     : {sm['auc_roc']:.4f}"
            if not np.isnan(sm["auc_roc"])
            else "    AUC-ROC     : N/A (single class)"
        )
        print(
            f"    Avg Precision: {sm['avg_precision']:.4f}"
            if not np.isnan(sm["avg_precision"])
            else "    Avg Precision: N/A (single class)"
        )
        print(f"    Precision   : {sm['precision']:.4f}")
        print(f"    Recall      : {sm['recall']:.4f}")
        print(f"    F1          : {sm['f1']:.4f}")
        mean_score = sf_scores.mean()
        print(f"    Mean score  : {mean_score:.4f}")

    # ---- Permutation Test ----
    print_section("PERMUTATION TEST  (is the model random?)")
    real_auc, p_value, permuted_aucs = permutation_test(
        all_scores, all_labels, n_permutations
    )
    if np.isnan(real_auc):
        print("  Cannot run (only one class present).")
    else:
        print(f"  Real AUC         : {real_auc:.4f}")
        print(
            f"  Permutation mean : {permuted_aucs.mean():.4f} "
            f"± {permuted_aucs.std():.4f}"
        )
        print(f"  p-value          : {p_value:.4f}")
        if p_value < 0.01:
            print("  ✅ HIGHLY SIGNIFICANT — model is NOT random (p < 0.01)")
        elif p_value < 0.05:
            print("  ✅ SIGNIFICANT — model is NOT random (p < 0.05)")
        else:
            print(
                "  ⚠️  NOT significant — cannot reject random baseline "
                f"(p = {p_value:.4f})"
            )

    # ---- Plots ----
    print_section("SAVING PLOTS")
    plots = {
        "roc_curve.png": lambda p: plot_roc_curve(all_scores, all_labels, p),
        "pr_curve.png": lambda p: plot_pr_curve(all_scores, all_labels, p),
        "score_distribution.png": lambda p: plot_score_distribution(
            all_scores, all_labels, p
        ),
        "permutation_test.png": lambda p: plot_permutation_test(
            real_auc, permuted_aucs, p_value, p
        ),
        "sample_timelines.png": lambda p: plot_sample_timelines(per_video, p),
    }
    for name, fn in plots.items():
        path = os.path.join(output_dir, name)
        fn(path)
        print(f"  → {path}")

    # Per-subfolder timeline plots (all videos)
    print("\n  Generating per-subfolder timeline plots (all videos)...")
    timeline_paths = plot_all_timelines(per_video, output_dir)
    for tp in timeline_paths:
        print(f"  → {tp}")

    print(f"\n{'=' * 60}")
    print("  Evaluation complete.")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate anomaly detection model against ground truth."
    )
    parser.add_argument(
        "--test-dataset-dir",
        default="test-dataset",
        help="Path to the test-dataset directory (default: test-dataset)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for binary classification (default: 0.5)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of permutations for the randomness test (default: 1000)",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Directory to save plots (default: evaluation_results)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.test_dataset_dir):
        print(f"Error: {args.test_dataset_dir} not found.", file=sys.stderr)
        sys.exit(1)

    all_scores, all_labels, per_video = collect_dataset(args.test_dataset_dir)

    if len(all_scores) == 0:
        print("Error: no score/label pairs found.", file=sys.stderr)
        sys.exit(1)

    report(
        all_scores,
        all_labels,
        per_video,
        args.threshold,
        args.n_permutations,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
