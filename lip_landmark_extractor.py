"""
Lip Landmark Extractor for NeuroSpeech Bridge
============================================
This script uses MediaPipe Face Landmarker to detect one face from the webcam
and extract the lip and jaw landmarks that will later be used for silent
speech recognition and lip-reading related analysis.

What this script does:
1. Opens the webcam
2. Detects one face in real time
3. Extracts a focused set of lip and jaw landmarks
4. Draws the full face mesh and highlights the lip landmarks
5. Prints the coordinates of each lip landmark to the terminal
6. Shows FPS, lip count, and face status on the screen

Important note:
- This script does not perform speech output, classification, DTW, or
  calibration.
- Its purpose is only to extract and visualize lip landmarks.

Requirements:
- Python 3.10+
- OpenCV
- MediaPipe
- NumPy
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_PATH = SCRIPT_DIR / "face_landmarker.task"
WINDOW_NAME = "NeuroSpeech Bridge - Lip Landmark Extractor"
TOTAL_LANDMARKS = 468

# ---------------------------------------------------------------------------
# Lip and jaw landmark indices
# ---------------------------------------------------------------------------
# These indices are a focused subset of the face mesh around the mouth area.
# They include landmarks for the upper lip, lower lip, mouth corners,
# and the jaw opening area. These points are important because they describe
# how the lips move during speech, even when no sound is produced.
#
# Why are these important?
# - Silent speech recognition uses mouth shape and movement patterns.
# - The lips and jaw help describe opening, closing, tension, and shape.
# - These landmarks can later be used for feature extraction, sequence
#   comparison, or machine learning.
LIP_JAW_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 191, 80, 81,
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def check_model_file():
    """
    Check whether the MediaPipe face model file exists.

    A model file is the trained neural network weights that MediaPipe uses to
    detect faces and landmarks. Without it, the face landmarker cannot run.
    """
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"✓ Model file found: {MODEL_PATH}")
        print(f"  File size: {size_mb:.1f} MB")
        return True

    print(f"✗ Model file not found: {MODEL_PATH}")
    print("Please download face_landmarker.task and place it in this folder.")
    return False


def initialize_landmarker():
    """
    Create the MediaPipe Face Landmarker object.

    This is the core AI model that finds landmarks on the face.
    It uses the downloaded .task model file.
    """
    try:
        base_options = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.2,
            min_face_presence_confidence=0.2,
            min_tracking_confidence=0.2,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        print("✓ Face Landmarker initialized successfully")
        return landmarker
    except Exception as exc:
        print(f"✗ Failed to initialize Face Landmarker: {exc}")
        return None


def initialize_camera(width=640, height=480, fps=30):
    """
    Open the webcam using several Windows-friendly camera backends.

    Some systems behave differently with OpenCV camera access, so we try a few
    common backends and camera indices to improve reliability.
    """
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    camera_indices = [0, 1, 2]

    for backend in backends:
        for index in camera_indices:
            try:
                cap = cv2.VideoCapture(index, backend)
                if not cap.isOpened():
                    cap.release()
                    continue

                time.sleep(0.3)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)

                ret, _ = cap.read()
                if not ret:
                    cap.release()
                    continue

                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"✓ Camera initialized successfully with backend={backend} index={index}")
                print(f"  Resolution: {actual_width}x{actual_height}")
                print(f"  FPS: {actual_fps}")
                return cap, (actual_width, actual_height)
            except Exception as exc:
                print(f"  • Backend {backend} index {index} failed: {exc}")

    print("✗ Camera failed to open with any tested backend/index")
    print("Check camera connection, permissions, or close other apps using the camera.")
    return None, None


def extract_lip_landmarks(face_landmarks):
    """
    Extract important lip and jaw landmarks from the face mesh.

    Parameters:
        face_landmarks: A list of landmarks returned by MediaPipe Face Landmarker.

    Returns:
        A list of tuples in the form (x, y, z) for the selected landmarks.

    Why this function matters:
    - It isolates only the mouth-related points that matter for silent speech
      recognition.
    - It reduces the amount of data while keeping the important movement cues.
    """
    if not face_landmarks:
        return []

    extracted = []
    for landmark_id in LIP_JAW_INDICES:
        if landmark_id < len(face_landmarks):
            landmark = face_landmarks[landmark_id]
            extracted.append((landmark.x, landmark.y, landmark.z))
    return extracted


def draw_face_mesh(frame, face_landmarks, frame_width, frame_height):
    """
    Draw the full face mesh in one color.

    Each landmark is drawn as a small point. This gives a visual overview of the
    whole face structure while keeping the output simple for beginners.
    """
    if not face_landmarks:
        return

    for landmark in face_landmarks:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        if 0 <= x < frame_width and 0 <= y < frame_height:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def draw_lip_landmarks(frame, face_landmarks, frame_width, frame_height):
    """
    Draw the selected lip and jaw landmarks in a different color.

    These are the landmarks we care about for silent speech analysis.
    We also draw the landmark number next to each point so it is easier to
    understand which index is being tracked.
    """
    if not face_landmarks:
        return []

    lip_points = []
    for landmark_id in LIP_JAW_INDICES:
        if landmark_id >= len(face_landmarks):
            continue

        landmark = face_landmarks[landmark_id]
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        if 0 <= x < frame_width and 0 <= y < frame_height:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(frame, str(landmark_id), (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            lip_points.append((landmark_id, x, y))

    return lip_points


def draw_info(frame, fps, lip_count, face_detected):
    """
    Draw information on the screen so the user can see the status in real time.
    """
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    if face_detected:
        status_text = f"Face detected | Lip landmarks: {lip_count}"
        status_color = (0, 255, 0)
    else:
        status_text = "No face detected"
        status_color = (0, 0, 255)

    cv2.putText(frame, status_text, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    cv2.putText(frame, "Press 'Q' to quit", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def print_lip_landmarks(lip_landmarks):
    """
    Print the selected lip points as coordinates to the terminal.

    This helps you inspect the raw values and later use them for feature
    extraction in Stage 4.
    """
    if not lip_landmarks:
        return

    print("\nLip landmarks:")
    for landmark_id, x, y in lip_landmarks:
        print(f"{landmark_id} -> x={x:.4f}, y={y:.4f}")


def main():
    """
    Main loop for webcam capture and lip landmark extraction.
    """
    print("\n" + "=" * 70)
    print("NeuroSpeech Bridge - Lip Landmark Extractor")
    print("=" * 70)

    if not check_model_file():
        sys.exit(1)

    landmarker = initialize_landmarker()
    if landmarker is None:
        sys.exit(1)

    cap, (width, height) = initialize_camera()
    if cap is None:
        sys.exit(1)

    prev_time = time.time()
    fps = 0.0
    frame_count = 0

    print("\nDetection started. Move your face into the camera view.")
    print("Press 'Q' to quit.\n")

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
        lip_landmarks = []

        if face_detected:
            face_landmarks = detection_result.face_landmarks[0]

            # Draw the full mesh in green
            draw_face_mesh(frame, face_landmarks, width, height)

            # Draw the selected lip/jaw landmarks in red and show their IDs
            lip_positions = draw_lip_landmarks(frame, face_landmarks, width, height)
            lip_landmarks = lip_positions

            # Extract the numerical landmark coordinates for later analysis
            extracted = extract_lip_landmarks(face_landmarks)
            if extracted:
                # Print the coordinates to the terminal for inspection.
                # We print every 15 frames to keep the output readable.
                if frame_count % 15 == 0:
                    print(f"\n[Frame {frame_count}] Face detected")
                    for idx, landmark_id in enumerate(LIP_JAW_INDICES):
                        if idx < len(extracted):
                            x, y, z = extracted[idx]
                            print(f"{landmark_id} -> x={x:.4f}, y={y:.4f}, z={z:.4f}")

        # Calculate FPS
        current_time = time.time()
        if current_time - prev_time > 0:
            fps = 1 / (current_time - prev_time)
        prev_time = current_time
        frame_count += 1

        draw_info(frame, fps, len(lip_landmarks), face_detected)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            print("\n✓ Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✓ Program terminated successfully")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
    except Exception as exc:
        print(f"\n✗ Unexpected error: {exc}")
        import traceback
        traceback.print_exc()
