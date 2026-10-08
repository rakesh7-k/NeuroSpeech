"""
NeuroSpeech Bridge - DTW Diagnostic and Evaluation Tool
======================================================
This script evaluates the current DTW-based phrase matching pipeline without
changing the classifier implementation.

What it does:
1. Loads calibration.json.
2. Compares every recorded sample against all other samples.
3. Prints average within-phrase and between-phrase DTW distances.
4. Builds a confusion-style report for leave-one-out evaluation.
5. Prints the top 3 closest matches for an offline test phrase.
6. Explains whether the current feature extraction, confidence scoring, or
   calibration quality is the likely bottleneck.

Important:
- This file is diagnostic-only.
- It does not modify the classifier.
- It does not add TTS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = SCRIPT_DIR / "calibration.json"
FEATURE_LENGTH = 69
MIN_SEQUENCE_LENGTH = 2
SAKOE_CHIBA_RATIO = 0.20


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


def validate_sequence(sequence) -> np.ndarray:
    """Convert input sequence to a validated numpy array."""
    if sequence is None:
        raise ValueError("Sequence is empty.")

    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Sequence must be a 2D array of shape (time_steps, 69).")
    if array.shape[1] != FEATURE_LENGTH:
        raise ValueError(f"Sequence must have {FEATURE_LENGTH} features per frame.")
    if array.shape[0] < MIN_SEQUENCE_LENGTH:
        print(f"⚠ Warning: sequence length {array.shape[0]} is below the recommended minimum; continuing.")

    return array


def normalize_sequence(sequence) -> np.ndarray:
    """Standardize each feature dimension."""
    seq = validate_sequence(sequence)
    mean_value = seq.mean(axis=0, keepdims=True)
    std_value = seq.std(axis=0, keepdims=True) + 1e-8
    return (seq - mean_value) / std_value


def resample_sequence(sequence, target_length: int = 24) -> np.ndarray:
    """Resample a sequence to a fixed number of frames for stable DTW comparison."""
    seq = normalize_sequence(sequence)
    if seq.shape[0] == target_length:
        return seq
    if seq.shape[0] == 1:
        return np.repeat(seq, target_length, axis=0)

    positions = np.linspace(0, seq.shape[0] - 1, target_length)
    resampled = np.empty((target_length, seq.shape[1]), dtype=np.float32)
    for dim in range(seq.shape[1]):
        resampled[:, dim] = np.interp(positions, np.arange(seq.shape[0]), seq[:, dim])
    return resampled


def dtw_distance(sequence1, sequence2) -> float:
    """Compute DTW distance using a Sakoe-Chiba band and normalized features."""
    seq1 = resample_sequence(sequence1)
    seq2 = resample_sequence(sequence2)

    n_frames = seq1.shape[0]
    m_frames = seq2.shape[0]
    band = max(2, int(max(n_frames, m_frames) * SAKOE_CHIBA_RATIO))

    dtw_table = np.full((n_frames + 1, m_frames + 1), np.inf, dtype=np.float32)
    dtw_table[0, 0] = 0.0

    for i in range(1, n_frames + 1):
        start_j = max(1, i - band)
        end_j = min(m_frames, i + band)
        for j in range(start_j, end_j + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            candidates = [
                dtw_table[i - 1, j],
                dtw_table[i, j - 1],
                dtw_table[i - 1, j - 1],
            ]
            dtw_table[i, j] = cost + min(candidates)

    total_cost = dtw_table[n_frames, m_frames]
    return float(total_cost / max(n_frames, m_frames))


def print_pairwise_distance_report(calibration_data: Dict[str, List[List[List[float]]]]) -> Dict[str, Dict[str, float]]:
    """Print average distances between phrase pairs and return the summary."""
    print("\n=== Pairwise DTW distance report ===")
    print("Distances are averaged across all sample pairs for each phrase combination.\n")

    summary: Dict[str, Dict[str, float]] = {}
    phrases = list(calibration_data.keys())

    for target_phrase in phrases:
        target_samples = calibration_data[target_phrase]
        summary[target_phrase] = {}
        print(f"Phrase: {target_phrase}")

        for comparison_phrase in phrases:
            comparison_samples = calibration_data[comparison_phrase]
            distances: List[float] = []

            for sample_a in target_samples:
                for sample_b in comparison_samples:
                    if target_phrase == comparison_phrase and sample_a is sample_b:
                        continue
                    if target_phrase == comparison_phrase:
                        # Avoid duplicate pairs for same-phrase comparisons.
                        if target_samples.index(sample_a) >= comparison_samples.index(sample_b):
                            continue
                    distances.append(dtw_distance(sample_a, sample_b))

            average_distance = mean(distances) if distances else float("nan")
            summary[target_phrase][comparison_phrase] = average_distance
            print(f"  Average {target_phrase} vs {comparison_phrase}: {average_distance:.4f}")

        print()

    return summary


def build_confusion_report(calibration_data: Dict[str, List[List[List[float]]]]) -> List[Dict[str, object]]:
    """Build a confusion-style report using leave-one-out evaluation."""
    print("=== Confusion-style evaluation ===")
    print("Each sample is tested against the full calibration set while excluding its own sample from the same phrase.\n")

    results: List[Dict[str, object]] = []
    for phrase, samples in calibration_data.items():
        for sample_idx, sample in enumerate(samples):
            reference_samples = [s for idx, s in enumerate(samples) if idx != sample_idx]
            predicted_phrase = None
            best_distance = float("inf")

            for other_phrase, other_samples in calibration_data.items():
                candidate_samples = reference_samples if other_phrase == phrase else other_samples
                if not candidate_samples:
                    continue
                distances = [dtw_distance(sample, candidate) for candidate in candidate_samples]
                candidate_distance = min(distances)
                if candidate_distance < best_distance:
                    best_distance = candidate_distance
                    predicted_phrase = other_phrase

            is_correct = predicted_phrase == phrase
            results.append(
                {
                    "phrase": phrase,
                    "sample_index": sample_idx,
                    "predicted_phrase": predicted_phrase,
                    "distance": best_distance,
                    "correct": is_correct,
                }
            )
            print(
                f"{phrase} sample {sample_idx + 1}: predicted {predicted_phrase} | distance {best_distance:.4f} | {'correct' if is_correct else 'incorrect'}"
            )

    print()
    return results


def print_top_matches(query_sequence, calibration_data: Dict[str, List[List[List[float]]]]) -> None:
    """Print the top 3 phrase matches for a query sequence."""
    phrase_distances: List[Tuple[str, float]] = []
    for phrase, samples in calibration_data.items():
        sample_distances = [dtw_distance(query_sequence, sample) for sample in samples]
        best_sample_distance = float(min(sample_distances))
        phrase_distances.append((phrase, best_sample_distance))

    phrase_distances.sort(key=lambda item: item[1])
    print("Top 3 closest phrase matches:")
    for rank, (phrase, distance) in enumerate(phrase_distances[:3], start=1):
        print(f"{rank}. {phrase} -> distance: {distance:.4f}")
    print()


def analyze_results(summary: Dict[str, Dict[str, float]], confusion_results: List[Dict[str, object]]) -> None:
    """Explain the likely source of the current DTW behavior."""
    accuracy = sum(1 for item in confusion_results if item["correct"]) / len(confusion_results) if confusion_results else float("nan")

    within_phrase_means: List[float] = []
    between_phrase_means: List[float] = []
    for target_phrase, comparisons in summary.items():
        for comparison_phrase, distance in comparisons.items():
            if target_phrase == comparison_phrase:
                within_phrase_means.append(distance)
            else:
                between_phrase_means.append(distance)

    average_within = mean(within_phrase_means) if within_phrase_means else float("nan")
    average_between = mean(between_phrase_means) if between_phrase_means else float("nan")
    margin = average_between - average_within if not np.isnan(average_within) and not np.isnan(average_between) else float("nan")

    print("=== Analysis ===")
    print(f"Overall leave-one-out accuracy: {accuracy * 100:.2f}%")
    print(f"Average within-phrase distance: {average_within:.4f}")
    print(f"Average between-phrase distance: {average_between:.4f}")
    print(f"Separation gap: {margin:.4f}")

    if np.isnan(accuracy):
        print("No evaluation results were produced.")
        return

    if accuracy >= 0.8 and margin > 0.1:
        print("Conclusion: the feature extraction is separating phrases reasonably well.")
        print("The main issue is likely the confidence calculation, because the classifier is selecting the right label but the margin between the best and second-best match may be too small.")
    elif accuracy >= 0.8 and margin <= 0.1:
        print("Conclusion: the feature extraction is producing some separation, but the margin between matches is narrow.")
        print("This suggests the confidence function is overly pessimistic or the DTW distances are compressed into a small numeric range.")
    elif accuracy < 0.5:
        print("Conclusion: the current features do not separate phrases well enough for reliable classification.")
        print("This points to either poor feature extraction, inconsistent calibration samples, or overly noisy landmarks.")
    else:
        print("Conclusion: the system is partially separating classes, but there is still noticeable overlap between phrases.")
        print("The next likely bottlenecks are calibration quality and the stability of the extracted features.")

    print("\nSuggested improvements (analysis only):")
    if accuracy >= 0.8:
        print("- Review the confidence formula and consider a more informative margin-based or probability-like score.")
    else:
        print("- Re-record calibration samples with more consistent timing and posture.")
        print("- Inspect whether the selected lip/jaw landmarks are sufficiently discriminative for each phrase.")
        print("- Consider normalizing or smoothing the feature sequence more aggressively before DTW.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the current DTW classifier behavior")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH, help="Path to calibration.json")
    parser.add_argument("--phrase", type=str, default="YES", help="Phrase to evaluate for top-3 offline matches")
    args = parser.parse_args()

    calibration_data = load_calibration_data(args.calibration)
    print("NeuroSpeech Bridge - DTW Diagnostic Tool")
    print("=" * 40)
    print(f"Loaded calibration data from: {args.calibration}")
    print(f"Phrases detected: {', '.join(calibration_data.keys())}")

    summary = print_pairwise_distance_report(calibration_data)
    confusion_results = build_confusion_report(calibration_data)

    phrase_to_test = args.phrase
    if phrase_to_test in calibration_data and calibration_data[phrase_to_test]:
        print(f"Offline test phrase: {phrase_to_test}")
        print_top_matches(calibration_data[phrase_to_test][0], calibration_data)
    else:
        print(f"Phrase '{phrase_to_test}' was not found in the calibration data.")

    analyze_results(summary, confusion_results)


if __name__ == "__main__":
    main()
