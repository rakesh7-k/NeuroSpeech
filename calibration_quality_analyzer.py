"""
NeuroSpeech Bridge - Calibration Quality Analyzer
=================================================
This script analyzes the quality of recorded calibration samples without
changing the DTW classifier or the phrase classification logic.

It focuses on:
1. Sequence length consistency
2. Feature variance across frames and samples
3. Cross-sample agreement within each phrase
4. Identification of noisy or inconsistent recordings

The output is intended to help decide which samples should be re-recorded or
removed before continuing with DTW-based recognition.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = SCRIPT_DIR / "calibration.json"
FEATURE_LENGTH = 69


def load_calibration_data(path: Path) -> Dict[str, List[List[List[float]]]]:
    """Load and validate calibration.json."""
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    if not isinstance(raw_data, dict):
        raise ValueError("Calibration file must contain a JSON object.")

    validated: Dict[str, List[List[List[float]]]] = {}
    for phrase, samples in raw_data.items():
        if not isinstance(phrase, str) or not phrase.strip():
            raise ValueError("Calibration contains an invalid phrase name.")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Phrase '{phrase}' must contain a non-empty list of samples.")

        validated_samples: List[List[List[float]]] = []
        for sample_idx, sample in enumerate(samples):
            if not isinstance(sample, list) or not sample:
                raise ValueError(f"Phrase '{phrase}' sample {sample_idx} is empty.")
            sequence: List[List[float]] = []
            for frame_idx, frame in enumerate(sample):
                if not isinstance(frame, list):
                    raise ValueError(f"Phrase '{phrase}' sample {sample_idx} frame {frame_idx} is invalid.")
                if len(frame) != FEATURE_LENGTH:
                    raise ValueError(
                        f"Phrase '{phrase}' sample {sample_idx} frame {frame_idx} has length {len(frame)}, expected {FEATURE_LENGTH}."
                    )
                sequence.append([float(value) for value in frame])
            validated_samples.append(sequence)

        validated[phrase] = validated_samples

    return validated


def compute_sequence_stats(sequence) -> Dict[str, float]:
    """Compute basic statistics for one feature sequence."""
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Each sequence must be a 2D array.")

    return {
        "length": float(arr.shape[0]),
        "variance": float(arr.var()),
        "mean_abs_change": float(np.mean(np.abs(np.diff(arr, axis=0)))) if arr.shape[0] > 1 else 0.0,
    }


def resample_to_common_length(sequence, target_length: int) -> np.ndarray:
    """Resample a sequence to a fixed number of frames."""
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Each sequence must be a 2D array.")
    if arr.shape[0] == target_length:
        return arr
    if arr.shape[0] == 1:
        return np.repeat(arr, target_length, axis=0)

    positions = np.linspace(0, arr.shape[0] - 1, target_length)
    resampled = np.empty((target_length, arr.shape[1]), dtype=np.float32)
    for dim in range(arr.shape[1]):
        resampled[:, dim] = np.interp(positions, np.arange(arr.shape[0]), arr[:, dim])
    return resampled


def compute_phrase_quality(phrase: str, samples: List[List[List[float]]]) -> Dict[str, object]:
    """Assess how consistent a phrase's samples are."""
    arrays = [np.asarray(sample, dtype=np.float32) for sample in samples]
    lengths = [arr.shape[0] for arr in arrays]
    variances = [float(arr.var()) for arr in arrays]

    target_length = max(lengths) if lengths else 24
    resampled_arrays = [resample_to_common_length(arr, target_length) for arr in arrays]

    # Compare each sample against the mean sample shape.
    stacked = np.stack(resampled_arrays, axis=0)
    mean_sample = np.mean(stacked, axis=0)
    sample_distances = []
    for arr in resampled_arrays:
        sample_distances.append(float(np.linalg.norm(arr - mean_sample) / max(1, arr.shape[0])))

    avg_length = mean(lengths)
    avg_variance = mean(variances)
    consistency_score = 1.0 / (1.0 + mean(sample_distances))

    return {
        "phrase": phrase,
        "average_length": avg_length,
        "variance": avg_variance,
        "consistency_score": consistency_score,
        "lengths": lengths,
        "sample_distances": sample_distances,
        "samples": samples,
    }


def identify_problematic_samples(phrase_stats: Dict[str, object]) -> List[Dict[str, object]]:
    """Flag samples that are unusually short, noisy, or inconsistent."""
    phrase_name = phrase_stats["phrase"]
    lengths = phrase_stats["lengths"]
    sample_distances = phrase_stats["sample_distances"]
    samples = phrase_stats["samples"]

    results: List[Dict[str, object]] = []
    avg_len = mean(lengths) if lengths else 0.0
    avg_dist = mean(sample_distances) if sample_distances else 0.0

    for idx, (sample, dist, length) in enumerate(zip(samples, sample_distances, lengths)):
        noisy = dist > avg_dist + max(0.5, 0.3 * avg_dist)
        short = length < avg_len * 0.7
        flagged = noisy or short
        results.append(
            {
                "sample_index": idx,
                "length": length,
                "distance_to_mean": dist,
                "noisy": noisy,
                "short": short,
                "flagged": flagged,
            }
        )

    return results


def save_plot(phrase: str, samples: List[List[List[float]]], output_path: Path) -> None:
    """Plot feature stability over time for all samples in a phrase."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Calibration stability for {phrase}")

    for idx, sample in enumerate(samples):
        arr = np.asarray(sample, dtype=np.float32)
        if arr.ndim != 2:
            continue
        # Use the first 10 feature dimensions for a compact summary.
        summary = arr[:, :10]
        x = np.arange(summary.shape[0])
        for dim in range(summary.shape[1]):
            axes[0].plot(x, summary[:, dim] + dim * 0.2, alpha=0.6, label=f"sample {idx + 1} dim {dim + 1}" if dim == 0 else None)

    # Visualize average frame energy and sample length.
    lengths = [np.asarray(sample, dtype=np.float32).shape[0] for sample in samples]
    axes[1].bar(range(1, len(lengths) + 1), lengths, color="steelblue")
    axes[1].set_ylabel("Frames")
    axes[1].set_xlabel("Sample")
    axes[1].set_title("Sample length across recordings")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        axes[0].legend(by_label.values(), by_label.keys(), loc="upper right", fontsize="small")
    axes[0].set_ylabel("Feature value (first 10 dims)")
    axes[0].set_title("Feature stability over time")

    plt.tight_layout(rect=[0, 0, 0.98, 0.96])
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_report(calibration_data: Dict[str, List[List[List[float]]]], output_dir: Path) -> None:
    """Generate a beginner-friendly report and plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("NeuroSpeech Bridge - Calibration Quality Analyzer")
    print("=" * 46)
    print("This report checks whether your calibration recordings are stable and consistent.\n")

    for phrase, samples in calibration_data.items():
        phrase_stats = compute_phrase_quality(phrase, samples)
        problem_samples = identify_problematic_samples(phrase_stats)

        print(f"Phrase: {phrase}")
        print(f"  Average length: {phrase_stats['average_length']:.1f} frames")
        print(f"  Average variance: {phrase_stats['variance']:.4f}")
        print(f"  Consistency score: {phrase_stats['consistency_score']:.4f}")

        if problem_samples:
            print("  Problem samples:")
            for item in problem_samples:
                reason_parts = []
                if item["short"]:
                    reason_parts.append("short")
                if item["noisy"]:
                    reason_parts.append("noisy")
                print(
                    f"    - Sample {item['sample_index'] + 1}: length={item['length']}, distance={item['distance_to_mean']:.4f}, reasons={', '.join(reason_parts)}"
                )
        else:
            print("  Problem samples: none detected")

        if phrase_stats["consistency_score"] < 0.45:
            recommendation = "Re-record this phrase"
        elif phrase_stats["consistency_score"] < 0.65:
            recommendation = "Review this phrase and consider re-recording one sample"
        else:
            recommendation = "This phrase looks reasonably stable"
        print(f"  Recommendation: {recommendation}\n")

        plot_path = output_dir / f"{phrase.lower().replace(' ', '_')}_stability.png"
        save_plot(phrase, samples, plot_path)
        print(f"  Plot saved: {plot_path.name}")

    print("\nBeginner-friendly summary:")
    print("- A higher consistency score means the samples for the same phrase look more alike.")
    print("- Short samples or samples that are far from the other recordings are likely weak calibration data.")
    print("- Phrases with low consistency should be re-recorded before using them for recognition.")


if __name__ == "__main__":
    output_dir = SCRIPT_DIR / "calibration_reports"
    calibration_data = load_calibration_data(CALIBRATION_PATH)
    generate_report(calibration_data, output_dir)
