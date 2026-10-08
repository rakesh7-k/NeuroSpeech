"""
NeuroSpeech Bridge - Final Integrated Demo App
=============================================
This application combines:
- webcam capture
- MediaPipe face landmark detection
- lip/jaw feature extraction
- DTW-based phrase classification
- optional speech feedback using edge-tts

It is designed as a polished hackathon demo with a clean UI and keyboard controls.
"""

from __future__ import annotations

import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import List

import cv2
import mediapipe as mp
import numpy as np

from dtw_classifier import analyze_speech_activity, classify, load_calibration_data
from feature_extractor import (
    check_model_file,
    draw_face_mesh,
    draw_lip_landmarks,
    extract_features,
    initialize_camera,
    initialize_landmarker,
)
from tts_engine import speak, toggle_mute


SCRIPT_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = SCRIPT_DIR / "calibration.json"
WINDOW_NAME = "NeuroSpeech Bridge"
FEATURE_LENGTH = 69
ROLLING_BUFFER_SIZE = 30
CLASSIFY_INTERVAL_SECONDS = 1.0
MIN_FRAMES_FOR_CLASSIFY = 8
SPEECH_COOLDOWN_SECONDS = 3.0


def draw_panel(frame, prediction: str, confidence: float, fps: float, face_detected: bool, muted: bool, status: str) -> None:
    """Draw a clean information panel over the webcam feed."""
    h, w = frame.shape[:2]
    panel_h = 180
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, panel_h), (12, 18, 24), -1)
    cv2.rectangle(overlay, (10, 10), (w - 10, panel_h), (80, 200, 255), 2)

    alpha = 0.85
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.putText(frame, "NeuroSpeech Bridge", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, "Silent speech demo", (28, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1)

    cv2.putText(frame, f"Prediction: {prediction}", (28, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(frame, f"Confidence: {confidence:.1f}%", (28, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (28, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    face_color = (0, 255, 0) if face_detected else (0, 0, 255)
    mute_color = (0, 255, 0) if not muted else (255, 255, 0)
    cv2.putText(frame, f"Face: {'Detected' if face_detected else 'Not detected'}", (260, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, face_color, 2)
    cv2.putText(frame, f"Mute: {'ON' if muted else 'OFF'}", (260, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, mute_color, 2)
    cv2.putText(frame, f"Status: {status}", (260, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    cv2.putText(frame, "M = mute/unmute   Q = quit", (28, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def run_demo() -> None:
    """Run the full live demo pipeline."""
    print("=" * 72)
    print("NeuroSpeech Bridge - Integrated Demo")
    print("=" * 72)

    if not check_model_file():
        raise RuntimeError("Face landmarker model file is missing.")

    landmarker = initialize_landmarker()
    if landmarker is None:
        raise RuntimeError("Failed to initialize MediaPipe Face Landmarker.")

    cap, _ = initialize_camera(width=640, height=480, fps=30)
    if cap is None:
        raise RuntimeError("Unable to open webcam.")

    calibration_data = load_calibration_data(CALIBRATION_PATH)
    print(f"Loaded {len(calibration_data)} phrases from {CALIBRATION_PATH.name}")

    sequence_buffer: List[np.ndarray] = []
    last_classify_time = 0.0
    last_speech_time = 0.0
    current_prediction = "Waiting"
    confidence = 0.0
    face_detected = False
    muted = False
    status = "Initializing"
    activity_status = "IDLE"
    movement_score = 0.0
    final_decision = "IDLE"
    prev_time = time.time()
    fps = 0.0
    last_face_seen = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                status = "Camera disconnected"
                print("Camera feed lost.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect(mp_image)
            face_detected = bool(detection_result.face_landmarks)

            if face_detected:
                face_landmarks = detection_result.face_landmarks[0]
                draw_face_mesh(frame, face_landmarks, frame.shape[1], frame.shape[0])
                lip_points = draw_lip_landmarks(frame, face_landmarks, frame.shape[1], frame.shape[0])

                feature_vector = extract_features(face_landmarks)
                if feature_vector.shape != (FEATURE_LENGTH,):
                    feature_vector = np.resize(feature_vector, FEATURE_LENGTH).astype(np.float32)

                sequence_buffer.append(feature_vector)
                if len(sequence_buffer) > ROLLING_BUFFER_SIZE:
                    sequence_buffer.pop(0)

                activity = analyze_speech_activity(sequence_buffer)
                activity_status = activity["status"]
                movement_score = activity["movement_score"]

                now_monotonic = time.monotonic()
                if activity["is_active"] and len(sequence_buffer) >= MIN_FRAMES_FOR_CLASSIFY and (now_monotonic - last_classify_time) >= CLASSIFY_INTERVAL_SECONDS:
                    last_classify_time = now_monotonic
                    try:
                        with redirect_stdout(io.StringIO()):
                            predicted_phrase, confidence_score, _, details = classify(
                                sequence_buffer,
                                calibration_data,
                                return_details=True,
                            )
                        current_prediction = predicted_phrase
                        confidence = confidence_score
                        final_decision = details["best_phrase"]
                        status = "Classifying"
                    except Exception as exc:
                        status = f"Classify error: {exc}"
                        print(f"Classification warning: {exc}")
                else:
                    if not activity["is_active"]:
                        current_prediction = "IDLE"
                        confidence = 0.0
                        final_decision = "IDLE"
                    status = "Waiting for activity"

                if lip_points:
                    status = "Tracking lips"
                last_face_seen = True
            else:
                status = "Waiting for face" if not last_face_seen else "Face lost"
                last_face_seen = False

            current_time = time.time()
            if current_time - prev_time > 0:
                fps = 1.0 / (current_time - prev_time)
            prev_time = current_time

            display_prediction = current_prediction if current_prediction in {"YES", "NO", "WATER", "PAIN", "HELP", "CALL NURSE", "STOP"} else ""
            draw_panel(frame, display_prediction or "IDLE", confidence, fps, face_detected, muted, status)
            cv2.putText(frame, f"Activity: {activity_status}", (28, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            cv2.putText(frame, f"Movement: {movement_score:.3f}", (28, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            cv2.putText(frame, f"Decision: {final_decision}", (28, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break
            if key == ord('m') or key == ord('M'):
                muted = not muted
                toggle_mute()
                status = "Muted" if muted else "Unmuted"

            now_monotonic = time.monotonic()
            should_speak = (
                face_detected
                and current_prediction in {"YES", "NO", "WATER", "PAIN", "HELP", "CALL NURSE", "STOP"}
                and (now_monotonic - last_speech_time) >= SPEECH_COOLDOWN_SECONDS
                and not muted
            )
            if should_speak:
                try:
                    speak(current_prediction, force=False)
                    last_speech_time = now_monotonic
                except Exception as exc:
                    print(f"TTS warning: {exc}")

        except KeyboardInterrupt:
            break
        except Exception as exc:
            status = f"Runtime error: {exc}"
            print(f"Runtime error: {exc}")
            time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()
    print("Demo stopped.")


def main() -> None:
    try:
        run_demo()
    except Exception as exc:
        print(f"Failed to start NeuroSpeech Bridge: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
