"""
NeuroSpeech Bridge - Stage 5 Calibration Recorder
===============================================
This script records short phrase-based calibration sequences for later use in
silence-speech or lip-reading analysis.

What calibration means:
- Calibration is the process of collecting repeated examples of known phrases.
- These examples act as reference data for later comparison methods such as DTW.
- The goal is to build a small library of phrase sequences that capture the
  typical mouth motion patterns for each phrase.

Why DTW requires calibration data:
- Dynamic Time Warping compares sequences of features over time.
- It needs multiple examples of the same phrase so it can learn the typical
  timing and shape variations.
- Without calibration data, there is nothing meaningful to compare against.

How calibration.json will be used later:
- The saved JSON file will serve as the dataset for future analysis stages.
- Each phrase will contain multiple recorded sequences.
- Later stages can load these sequences to compare incoming speech-like motion
  against the stored reference patterns.

Important note:
- This script does not implement DTW, classification, prediction, or TTS.
- It only records phrase sequences and saves them in calibration.json.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import mediapipe as mp

from feature_extractor import (
    check_model_file,
    initialize_landmarker,
    initialize_camera,
    extract_features,
    draw_face_mesh,
    draw_lip_landmarks,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_PATH = SCRIPT_DIR / "calibration.json"
WINDOW_NAME = "NeuroSpeech Bridge - Calibration Recorder"

PHRASES = ["YES", "NO", "WATER", "PAIN", "HELP", "CALL NURSE", "STOP"]
SAMPLES_PER_PHRASE = 3
TARGET_FRAMES_PER_SAMPLE = 30
MIN_SEQUENCE_LENGTH = 15
QUALITY_WARNING_THRESHOLD = 0.6  # warn if we capture fewer than 60% of target frames


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def show_overlay(frame, phrase: str, sample_num: int, frame_count: int, face_detected: bool, status_text: str):
    """Draw a simple status overlay in the video window."""
    cv2.putText(frame, f"Phrase: {phrase}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Sample: {sample_num}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Frames: {frame_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if face_detected:
        color = (0, 255, 0)
        face_text = "Face detected"
    else:
        color = (0, 0, 255)
        face_text = "No face detected"

    cv2.putText(frame, face_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, status_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def show_countdown(frame, countdown_value: int):
    """Display countdown numbers before each recording sample."""
    if countdown_value > 0:
        text = str(countdown_value)
        color = (0, 255, 255)
    else:
        text = "RECORDING"
        color = (0, 255, 0)

    cv2.putText(frame, text, (frame.shape[1] // 2 - 80, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)


def countdown_window(cap, landmarker, width: int, height: int, phrase: str, sample_num: int):
    """Show a countdown before recording begins."""
    print(f"\nPreparing to record: {phrase} | Sample {sample_num}")
    print("3...")
    for value in [3, 2, 1]:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        show_countdown(frame, value)
        show_overlay(frame, phrase, sample_num, 0, True, "Get ready")
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(300)

    # Final recording banner
    for _ in range(3):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        show_countdown(frame, 0)
        show_overlay(frame, phrase, sample_num, 0, True, "Recording")
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(150)

    return True


def record_sample(landmarker, cap, width: int, height: int, phrase: str, sample_num: int) -> List[np.ndarray]:
    """
    Record one sample of feature vectors for a phrase.

    The recording loop collects one feature vector per frame when a face is
    detected. Frames without a face are skipped to keep the sequence clean.
    """
    sequence: List[np.ndarray] = []
    frame_count = 0
    valid_frame_count = 0

    print(f"\nRecording {phrase} - Sample {sample_num}")
    print(f"Target frames: {TARGET_FRAMES_PER_SAMPLE}")

    countdown_window(cap, landmarker, width, height, phrase, sample_num)

    while len(sequence) < TARGET_FRAMES_PER_SAMPLE:
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
            if feature_vector.shape != (69,):
                feature_vector = np.resize(feature_vector, 69).astype(np.float32)

            sequence.append(feature_vector)
            valid_frame_count += 1

        frame_count += 1

        show_overlay(frame, phrase, sample_num, valid_frame_count, face_detected, "Recording")
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            print("\n✓ Recording interrupted by user")
            break

    # Validate recording quality.
    if len(sequence) < MIN_SEQUENCE_LENGTH:
        print("⚠ Warning: recording quality is poor. Sequence length is below the minimum threshold.")
    elif len(sequence) < int(TARGET_FRAMES_PER_SAMPLE * QUALITY_WARNING_THRESHOLD):
        print("⚠ Warning: recording quality is below the recommended level.")
    else:
        print(f"✓ Sample recorded successfully with {len(sequence)} valid frames")

    return sequence


def save_calibration_data(calibration_data: Dict[str, List[List[List[float]]]], output_path: Path):
    """Save the recorded phrase sequences to calibration.json."""
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(calibration_data, handle, indent=2)
    print(f"✓ Saved calibration data to {output_path}")


def main():
    """Run the full calibration recording workflow."""
    print("\n" + "=" * 72)
    print("NeuroSpeech Bridge - Calibration Recorder")
    print("=" * 72)
    print("\nCalibration collects repeated examples of known phrases so later")
    print("stages can compare new mouth motion sequences against these references.")

    if not check_model_file():
        sys.exit(1)

    landmarker = initialize_landmarker()
    if landmarker is None:
        sys.exit(1)

    cap, (width, height) = initialize_camera()
    if cap is None:
        sys.exit(1)

    calibration_data: Dict[str, List[List[List[float]]]] = {phrase: [] for phrase in PHRASES}

    try:
        for phrase in PHRASES:
            for sample_num in range(1, SAMPLES_PER_PHRASE + 1):
                sequence = record_sample(landmarker, cap, width, height, phrase, sample_num)
                calibration_data[phrase].append([feature_vector.tolist() for feature_vector in sequence])

        save_calibration_data(calibration_data, OUTPUT_PATH)

        total_phrases = len(PHRASES)
        total_samples = sum(len(calibration_data[phrase]) for phrase in PHRASES)
        print("\nCalibration completed successfully.")
        print(f"Total phrases recorded: {total_phrases}")
        print(f"Total samples recorded: {total_samples}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
    except Exception as exc:
        print(f"\n✗ Unexpected error: {exc}")
        import traceback
        traceback.print_exc()
