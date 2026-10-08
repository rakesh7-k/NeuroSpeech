"""
NeuroSpeech Bridge - Stage 4 Feature Extraction
=============================================
This script turns the lip/jaw landmarks from MediaPipe Face Landmarker into a
stable 69-dimensional feature vector for each video frame.

What feature extraction means:
- Raw landmark coordinates are useful, but they are hard to compare directly
  across different faces, camera distances, and head poses.
- Feature extraction converts those landmarks into a compact numeric vector that
  summarizes the mouth region in a consistent way.
- This vector can later be used for sequence analysis, machine learning, or
  comparison between frames.

Why normalization is required:
- The webcam may be closer or farther from the face.
- Different people have different head sizes and face proportions.
- Without normalization, the same mouth movement could produce very different
  coordinate values just because the face is bigger or smaller in the image.

Why nose-relative coordinates are used:
- The nose is a stable and central reference point on the face.
- Measuring each lip landmark relative to the nose makes the features less
  sensitive to global head movement and camera position.
- It helps the system focus on mouth motion rather than the face's absolute
  placement in the frame.

Why IOD normalization improves robustness:
- IOD means Inter-Ocular Distance, which is the distance between the two eyes.
- It acts like a built-in scale measurement for the face.
- Dividing coordinates by IOD makes the feature values more consistent across
  people and across small changes in camera distance.

Important note:
- This script does not perform calibration, DTW, classification, or TTS.
- Its job is only to generate a stable 69-dimensional feature vector.
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
WINDOW_NAME = "NeuroSpeech Bridge - Feature Extractor"
TOTAL_LANDMARKS = 468

# ---------------------------------------------------------------------------
# Lip and jaw landmark indices
# ---------------------------------------------------------------------------
# These are the same mouth-related landmarks selected in the earlier Stage 3
# script. Each landmark contributes 3 values: x, y, and z.
# 23 landmarks * 3 values = 69 features total.
LIP_JAW_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 191, 80, 81,
]

# ---------------------------------------------------------------------------
# Landmark indices used for normalization
# ---------------------------------------------------------------------------
# Nose reference point: a stable central point near the middle of the face.
NOSE_INDEX = 1

# Inter-Ocular Distance (IOD) uses the left and right eye landmarks.
# These are common face mesh landmarks used to estimate face scale.
LEFT_EYE_INDEX = 133
RIGHT_EYE_INDEX = 362

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def check_model_file():
    """Verify that the MediaPipe face model file exists."""
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"✓ Model file found: {MODEL_PATH}")
        print(f"  File size: {size_mb:.1f} MB")
        return True

    print(f"✗ Model file not found: {MODEL_PATH}")
    print("Please download face_landmarker.task and place it in this folder.")
    return False


def initialize_landmarker():
    """Create the MediaPipe Face Landmarker object."""
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
    """Open the webcam using a few Windows-friendly camera backends."""
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
    return None, None


def extract_features(face_landmarks):
    """
    Convert lip/jaw landmarks into a normalized 69-dimensional feature vector.

    Each selected landmark contributes three values: normalized x, normalized y,
    and normalized z. The values are measured relative to the nose and scaled
    by the Inter-Ocular Distance (IOD) so that the vector is more stable.

    Returns:
        numpy.ndarray with shape (69,)
    """
    # If there are no landmarks, return a zero vector with the correct shape.
    if not face_landmarks:
        return np.zeros(69, dtype=np.float32)

    # We use the nose landmark as a stable center point.
    nose = face_landmarks[NOSE_INDEX] if NOSE_INDEX < len(face_landmarks) else None

    # IOD gives a face-size scale that helps normalize the coordinates.
    left_eye = face_landmarks[LEFT_EYE_INDEX] if LEFT_EYE_INDEX < len(face_landmarks) else None
    right_eye = face_landmarks[RIGHT_EYE_INDEX] if RIGHT_EYE_INDEX < len(face_landmarks) else None

    if nose is None or left_eye is None or right_eye is None:
        return np.zeros(69, dtype=np.float32)

    dx_iod = right_eye.x - left_eye.x
    dy_iod = right_eye.y - left_eye.y
    dz_iod = right_eye.z - left_eye.z
    iod = np.sqrt(dx_iod * dx_iod + dy_iod * dy_iod + dz_iod * dz_iod)

    # If the IOD is too small, we cannot normalize safely.
    if iod < 1e-6:
        return np.zeros(69, dtype=np.float32)

    features = []

    # For each selected lip/jaw landmark, compute nose-relative and IOD-scaled
    # coordinates. This produces one feature for x, one for y, and one for z.
    for landmark_id in LIP_JAW_INDICES:
        if landmark_id < len(face_landmarks):
            landmark = face_landmarks[landmark_id]
            dx = landmark.x - nose.x
            dy = landmark.y - nose.y
            dz = landmark.z - nose.z
            features.extend([
                dx / iod,
                dy / iod,
                dz / iod,
            ])
        else:
            # In case a landmark index is missing, fill with zeros.
            features.extend([0.0, 0.0, 0.0])

    feature_vector = np.asarray(features, dtype=np.float32).reshape(-1)

    # We must ensure the output shape is exactly (69,).
    if feature_vector.shape != (69,):
        feature_vector = np.resize(feature_vector, 69).astype(np.float32)

    return feature_vector


def draw_face_mesh(frame, face_landmarks, frame_width, frame_height):
    """Draw the full face mesh as a light green set of points."""
    if not face_landmarks:
        return

    for landmark in face_landmarks:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        if 0 <= x < frame_width and 0 <= y < frame_height:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def draw_lip_landmarks(frame, face_landmarks, frame_width, frame_height):
    """Draw only the selected lip/jaw landmarks in red."""
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


def draw_info(frame, fps, lip_count, feature_count, face_detected):
    """Draw status text onto the live camera window."""
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

    cv2.putText(frame, f"Feature count: {feature_count}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, "Press 'Q' to quit", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def print_feature_summary(feature_vector, frame_count):
    """Print a compact summary of the extracted feature vector."""
    print(f"\n[Frame {frame_count}] Face detected")
    print(f"Feature vector length: {feature_vector.shape[0]}")
    print(f"First 10 values: {feature_vector[:10]}")
    print(
        "Feature statistics: "
        f"mean={feature_vector.mean():.6f}, "
        f"std={feature_vector.std():.6f}, "
        f"min={feature_vector.min():.6f}, "
        f"max={feature_vector.max():.6f}"
    )


def main():
    """Run the webcam loop and extract features from each frame."""
    print("\n" + "=" * 72)
    print("NeuroSpeech Bridge - Feature Extractor")
    print("=" * 72)
    print("\nFeature extraction turns mouth landmarks into a stable numeric vector.")
    print("The vector is normalized relative to the nose and scaled by the face's IOD.")

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
        feature_vector = np.zeros(69, dtype=np.float32)

        if face_detected:
            face_landmarks = detection_result.face_landmarks[0]

            # Draw the full face mesh in green.
            draw_face_mesh(frame, face_landmarks, width, height)

            # Draw the selected lip/jaw landmarks in red.
            lip_landmarks = draw_lip_landmarks(frame, face_landmarks, width, height)

            # Convert the lip/jaw landmarks into a normalized numerical vector.
            feature_vector = extract_features(face_landmarks)

            # The feature vector should always be exactly 69 numbers long.
            assert feature_vector.shape == (69,), f"Expected shape (69,), got {feature_vector.shape}"

            # Print a readable summary every 15 frames.
            if frame_count % 15 == 0:
                print_feature_summary(feature_vector, frame_count)

        # Calculate FPS for the live display.
        current_time = time.time()
        if current_time - prev_time > 0:
            fps = 1 / (current_time - prev_time)
        prev_time = current_time
        frame_count += 1

        draw_info(frame, fps, len(lip_landmarks), feature_vector.shape[0], face_detected)
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
