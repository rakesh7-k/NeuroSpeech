"""
NeuroSpeech Bridge - Stage 6 DTW Recognition Engine
==================================================
This script loads the recorded calibration data and uses Dynamic Time Warping
(DTW) to compare a live or offline sequence of lip/jaw feature vectors against
stored phrase examples.

What DTW is:
- DTW is an algorithm for comparing two temporal sequences that may have
  different lengths or different speeds.
- It finds the best alignment between the sequences so that similar motion
  patterns can be compared even when the timing is not exactly the same.

Why DTW works for silent speech:
- Silent speech and mouth motion are not perfectly timed or perfectly uniform.
- The same phrase can be said more slowly or faster, and the mouth may open and
  close at slightly different times.
- DTW handles this by warping the time axis and matching the most similar
  points between two sequences.

Why sequence length can differ:
- A recorded phrase may contain different numbers of frames depending on how
  quickly the user moves their mouth or how stable the tracker is.
- DTW is designed to compare sequences even when one is longer than the other.

How confidence is computed:
- Confidence is based on the margin between the best and second-best phrase
  matches.
- A clear winner produces a higher confidence score, while a close tie produces
  a lower confidence score.

Important note:
- This script does not perform TTS, classification training, or prediction
  beyond phrase matching using DTW.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from typing import Optional

from feature_extractor import (
    check_model_file,
    initialize_landmarker,
    initialize_camera,
    extract_features,
    draw_face_mesh,
    draw_lip_landmarks,
)
from tts_engine import speak, toggle_mute


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
CALIBRATION_PATH = SCRIPT_DIR / "calibration.json"
WINDOW_NAME = "NeuroSpeech Bridge - DTW Classifier"
FEATURE_LENGTH = 69
MIN_SEQUENCE_LENGTH = 2
SAKOE_CHIBA_RATIO = 0.20
ACTIVITY_WINDOW_SIZE = 30
MIN_ACTIVITY_FRAMES = 8
MOUTH_OPENING_THRESHOLD = 0.030
LIP_MOVEMENT_THRESHOLD = 0.020
JAW_MOVEMENT_THRESHOLD = 0.020
MOVEMENT_SCORE_THRESHOLD = 0.028

# ---------------------------------------------------------------------------
# Utility and validation helpers
# ---------------------------------------------------------------------------
def load_calibration_data(path: Path) -> Dict[str, List[List[List[float]]]]:
    """Load calibration.json and validate that it has the expected structure."""
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw_data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Calibration file is corrupted: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ValueError("Calibration file must contain a JSON object.")

    validated: Dict[str, List[List[List[float]]]] = {}
    for phrase, samples in raw_data.items():
        if not isinstance(phrase, str) or not phrase.strip():
            raise ValueError("Calibration file contains an invalid phrase name.")
        if not isinstance(samples, list):
            raise ValueError(f"Phrase '{phrase}' must contain a list of samples.")

        validated_samples: List[List[List[float]]] = []
        for sample_idx, sample in enumerate(samples):
            if not isinstance(sample, list) or not sample:
                raise ValueError(f"Phrase '{phrase}' sample {sample_idx} is empty.")

            validated_sequence: List[List[float]] = []
            for frame_idx, frame in enumerate(sample):
                if not isinstance(frame, list):
                    raise ValueError(f"Phrase '{phrase}' sample {sample_idx} frame {frame_idx} is invalid.")
                if len(frame) != FEATURE_LENGTH:
                    raise ValueError(
                        f"Phrase '{phrase}' sample {sample_idx} frame {frame_idx} has length {len(frame)}, expected {FEATURE_LENGTH}."
                    )
                validated_sequence.append([float(value) for value in frame])

            validated_samples.append(validated_sequence)

        validated[phrase] = validated_samples

    if not validated:
        raise ValueError("Calibration data is empty.")

    return validated


def validate_sequence(sequence) -> np.ndarray:
    """Convert sequence input into a validated NumPy array."""
    if sequence is None:
        raise ValueError("Sequence is empty.")

    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Sequence must be a 2D array of shape (time_steps, 69).")
    if array.shape[1] != FEATURE_LENGTH:
        raise ValueError(f"Sequence must have {FEATURE_LENGTH} features per frame.")
    if array.shape[0] < MIN_SEQUENCE_LENGTH:
        print(f"⚠ Warning: sequence length {array.shape[0]} is below the recommended minimum; continuing with available frames.")

    return array


def normalize_sequence(sequence) -> np.ndarray:
    """Standardize each feature dimension to improve DTW robustness."""
    seq = validate_sequence(sequence)
    mean = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0, keepdims=True) + 1e-8
    return (seq - mean) / std


def resample_sequence(sequence, target_length: int = 24) -> np.ndarray:
    """Resample a sequence to a fixed number of frames for more stable comparison."""
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


def build_phrase_prototypes(calibration_data: Dict[str, List[List[List[float]]]], target_length: int = 24) -> Dict[str, np.ndarray]:
    """Create a representative prototype sequence for each phrase."""
    prototypes: Dict[str, np.ndarray] = {}
    for phrase, samples in calibration_data.items():
        if not samples:
            continue

        resampled_samples = [resample_sequence(sample, target_length) for sample in samples]
        prototype = np.mean(np.stack(resampled_samples, axis=0), axis=0).astype(np.float32)
        prototypes[phrase] = prototype

    return prototypes


def dtw_distance(sequence1, sequence2) -> float:
    """
    Compute DTW distance between two feature sequences.

    This implementation uses:
    - Euclidean distance between feature vectors at each step
    - A Sakoe-Chiba band to limit the search window
    - A dynamic programming table for efficient comparison
    - Sequence normalization so shape-based differences matter more than raw scale
    """
    seq1 = normalize_sequence(sequence1)
    seq2 = normalize_sequence(sequence2)

    n_frames = seq1.shape[0]
    m_frames = seq2.shape[0]

    # The Sakoe-Chiba band restricts the allowable warping path.
    # A larger band allows more flexibility, while a smaller band is faster.
    band = max(2, int(max(n_frames, m_frames) * SAKOE_CHIBA_RATIO))

    # Infinite values are used so that invalid cells are skipped.
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


def analyze_speech_activity(sequence_buffer, window_size: int = ACTIVITY_WINDOW_SIZE) -> Dict[str, float]:
    """
    Detect whether the user is actively moving the mouth.

    This gate is intentionally simple and uses the existing 69D feature vectors
    as a proxy for mouth motion. It avoids running DTW during silent periods.
    """
    if not sequence_buffer or len(sequence_buffer) < MIN_ACTIVITY_FRAMES:
        return {
            "mouth_opening": 0.0,
            "lip_movement": 0.0,
            "jaw_movement": 0.0,
            "movement_score": 0.0,
            "is_active": False,
            "status": "IDLE",
        }

    recent_frames = np.asarray(sequence_buffer[-window_size:], dtype=np.float32)
    if recent_frames.ndim != 2:
        recent_frames = recent_frames.reshape(-1, FEATURE_LENGTH)

    reshaped = recent_frames.reshape(-1, 23, 3)
    lip_points = reshaped[:, :12, 1]
    jaw_points = reshaped[:, 12:, 1]

    mouth_opening = float(np.mean(np.abs(lip_points[-1])))
    lip_movement = float(np.mean(np.linalg.norm(np.diff(lip_points, axis=0), axis=1))) if lip_points.shape[0] > 1 else 0.0
    jaw_movement = float(np.mean(np.linalg.norm(np.diff(jaw_points, axis=0), axis=1))) if jaw_points.shape[0] > 1 else 0.0
    movement_score = float(
        0.45 * mouth_opening + 0.35 * lip_movement + 0.20 * jaw_movement
    )

    is_active = (
        movement_score > MOVEMENT_SCORE_THRESHOLD
        or mouth_opening > MOUTH_OPENING_THRESHOLD
        or lip_movement > LIP_MOVEMENT_THRESHOLD
        or jaw_movement > JAW_MOVEMENT_THRESHOLD
    )

    return {
        "mouth_opening": mouth_opening,
        "lip_movement": lip_movement,
        "jaw_movement": jaw_movement,
        "movement_score": movement_score,
        "is_active": bool(is_active),
        "status": "ACTIVE" if is_active else "IDLE",
    }


def classify(
    sequence,
    calibration_data: Dict[str, List[List[List[float]]]],
    return_details: bool = False,
) -> Tuple[object, ...]:
    """
    Compare a live or offline sequence against all calibration samples.

    Returns:
        If return_details is False: predicted_phrase, confidence_score, top_matches
        If return_details is True: predicted_phrase, confidence_score, top_matches, details
    """
    if not calibration_data:
        raise ValueError("Calibration data is empty")

    validated_sequence = validate_sequence(sequence)
    prototypes = build_phrase_prototypes(calibration_data)

    phrase_distances: List[Tuple[str, float]] = []

    # Each phrase is represented by several recorded samples. We compare the
    # incoming sequence against every sample and use the best sample match for
    # the phrase-level result. We also include the phrase prototype to stabilize
    # the decision when the recorded samples vary slightly.
    for phrase, samples in calibration_data.items():
        if not samples:
            continue

        sample_distances = [dtw_distance(validated_sequence, sample) for sample in samples]
        best_sample_distance = float(np.min(sample_distances))
        prototype_distance = float(dtw_distance(validated_sequence, prototypes[phrase]))
        combined_distance = 0.7 * best_sample_distance + 0.3 * prototype_distance
        phrase_distances.append((phrase, combined_distance))

    phrase_distances.sort(key=lambda item: item[1])

    best_phrase, best_distance = phrase_distances[0]
    second_phrase, second_distance = phrase_distances[1] if len(phrase_distances) > 1 else (best_phrase, best_distance)

    # Confidence is derived from the margin between the best and second-best
    # phrase matches. A large margin means the result is more reliable.
    if second_distance <= 1e-8:
        confidence_percent = 100.0
    else:
        gap = max(0.0, second_distance - best_distance)
        confidence_percent = float(min(100.0, max(0.0, 100.0 / (1.0 + gap))))

    top_matches = phrase_distances[:3]
    print("\nTop 3 closest phrase matches:")
    for rank, (phrase, distance) in enumerate(top_matches, start=1):
        print(f"{rank}. {phrase} -> DTW distance: {distance:.4f}")

    print(f"\nBest match: {best_phrase}")
    print(f"DTW distance: {best_distance:.4f}")
    print(f"Second-best distance: {second_distance:.4f}")
    print(f"Confidence: {confidence_percent:.2f}%")

    if return_details:
        return best_phrase, confidence_percent, top_matches, {
            "best_phrase": best_phrase,
            "best_distance": best_distance,
            "second_distance": second_distance,
        }
    return best_phrase, confidence_percent, top_matches


def run_offline_test(calibration_data: Dict[str, List[List[List[float]]]]) -> bool:
    """Load one recorded sample and verify that DTW identifies the same phrase."""
    if not calibration_data:
        raise ValueError("Calibration data is empty")

    first_phrase = next(iter(calibration_data.keys()))
    test_sample = calibration_data[first_phrase][0]

    print(f"\nOffline test using phrase '{first_phrase}'")
    predicted_phrase, confidence_score, _ = classify(test_sample, calibration_data)

    if predicted_phrase == first_phrase:
        print("Offline test passed.")
        print(f"Predicted phrase: {predicted_phrase}")
        print(f"Confidence: {confidence_score:.2f}%")
        return True

    print("Offline test failed: predicted phrase did not match the known phrase.")
    return False


def run_live_classification(landmarker) -> None:
    """Run a live webcam session and classify each captured sequence."""
    print("\nLive classification mode started. Hold the phrase briefly in front of the camera.")
    print("Press 'Q' to quit.\n")

    cap, (width, height) = initialize_camera()
    if cap is None:
        sys.exit(1)

    calibration_data = load_calibration_data(CALIBRATION_PATH)

    sequence_buffer: List[np.ndarray] = []
    prev_time = time.time()
    fps = 0.0
    last_predicted_phrase: Optional[str] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✗ Could not read frame from camera")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)
        face_detected = bool(detection_result.face_landmarks)

        if face_detected:
            face_landmarks = detection_result.face_landmarks[0]
            draw_face_mesh(frame, face_landmarks, width, height)
            draw_lip_landmarks(frame, face_landmarks, width, height)

            feature_vector = extract_features(face_landmarks)
            if feature_vector.shape != (FEATURE_LENGTH,):
                feature_vector = np.resize(feature_vector, FEATURE_LENGTH).astype(np.float32)

            sequence_buffer.append(feature_vector)

            activity = analyze_speech_activity(sequence_buffer)
            cv2.putText(frame, f"Activity: {activity['status']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Movement: {activity['movement_score']:.3f}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if activity["is_active"] and len(sequence_buffer) >= MIN_SEQUENCE_LENGTH:
                predicted_phrase, confidence_score, _ = classify(sequence_buffer, calibration_data)
                cv2.putText(frame, f"Prediction: {predicted_phrase}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Confidence: {confidence_score:.1f}%", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if predicted_phrase != last_predicted_phrase:
                    speak(predicted_phrase)
                    last_predicted_phrase = predicted_phrase
            else:
                cv2.putText(frame, "Prediction: IDLE", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, "Confidence: 0.0%", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        current_time = time.time()
        if current_time - prev_time > 0:
            fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'Q' to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        if key == ord('m') or key == ord('M'):
            toggle_mute()

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    """Entry point for the DTW classifier."""
    print("\n" + "=" * 72)
    print("NeuroSpeech Bridge - DTW Recognition Engine")
    print("=" * 72)

    if "--offline-test" in sys.argv:
        calibration_data = load_calibration_data(CALIBRATION_PATH)
        run_offline_test(calibration_data)
        return

    if "--live" in sys.argv:
        if not check_model_file():
            sys.exit(1)

        landmarker = initialize_landmarker()
        if landmarker is None:
            sys.exit(1)

        run_live_classification(landmarker)
        return

    # Default behavior is to run the offline verification test.
    calibration_data = load_calibration_data(CALIBRATION_PATH)
    run_offline_test(calibration_data)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
    except Exception as exc:
        print(f"\n✗ Unexpected error: {exc}")
        import traceback
        traceback.print_exc()
