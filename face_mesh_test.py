"""
MediaPipe Face Mesh Real-time Detection - Enhanced Version
==========================================================
This script detects and visualizes facial landmarks from webcam in real-time.
It includes comprehensive troubleshooting and detailed explanations for beginners.

Author: NeuroSpeech Bridge Project
Version: 2.0 (Enhanced with Setup Guidance)
Requirements: opencv-python, mediapipe>=0.10.0, numpy
Python: 3.10+
"""

# ============================================================================
# STEP 1: IMPORTS
# ============================================================================
import cv2              # Computer vision library
import time             # For timing and FPS calculation
import sys              # System operations
import numpy as np      # Numerical arrays (for image data)
from pathlib import Path  # File path handling

# MediaPipe imports
try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import RunningMode
    import mediapipe as mp
    print("✓ MediaPipe imported successfully")
    print(f"  MediaPipe Version: {mp.__version__}")
except ImportError as e:
    print(f"✗ Error importing MediaPipe: {e}")
    print("Install with: pip install mediapipe")
    sys.exit(1)


# ============================================================================
# STEP 2: CONFIGURATION AND CONSTANTS
# ============================================================================

# Get script directory for model file
SCRIPT_DIR = Path(__file__).parent
MODEL_FILE = SCRIPT_DIR / "face_landmarker.task"

# Lip landmarks indices (out of 468 total)
# These are the indices of landmarks on the lips
LIP_LANDMARKS = set([
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95
])
TOTAL_LANDMARKS = 468  # Total face landmarks detected


# ============================================================================
# STEP 3: MODEL FILE VERIFICATION AND SETUP
# ============================================================================

def check_and_setup_model():
    """
    Checks if the model file exists and provides setup instructions if needed.
    
    HOW IT WORKS:
    1. Looks for 'face_landmarker.task' in the script directory
    2. If found, verifies file size
    3. If not found, displays detailed download instructions
    4. Returns the model path if valid, exits if not available
    
    WHAT IS A MODEL FILE?
    - A .task file contains pre-trained neural network weights
    - MediaPipe uses this to detect face landmarks
    - It's ~90MB in size and contains thousands of parameters
    - Downloaded once, then cached locally
    
    Returns:
        str: Path to model file if valid
    """
    print("\n" + "="*70)
    print("STEP 1: MODEL FILE VERIFICATION")
    print("="*70)
    
    # Check if model exists
    if MODEL_FILE.exists():
        file_size = MODEL_FILE.stat().st_size / (1024*1024)
        print(f"✓ Model file found: {MODEL_FILE}")
        print(f"  File size: {file_size:.1f} MB")
        return str(MODEL_FILE)
    
    # Model not found - display setup instructions
    print(f"✗ Model file NOT found: {MODEL_FILE}")
    print("\n" + "-"*70)
    print("HOW TO FIX THIS - CHOOSE ONE METHOD")
    print("-"*70)
    
    print("\n[METHOD 1] DOWNLOAD USING PYTHON")
    print("-"*70)
    print("Open a Python terminal in this folder and run:")
    print("https://www.kaggle.com/models/search?q=face_landmarker")
    print(f"Then place downloaded file in: {SCRIPT_DIR}")
    
    print("\n[METHOD 2] ALTERNATIVE DOWNLOAD SOURCES")
    print("-"*70)
    print("Option A: GitHub Releases")
    print("  https://github.com/google/mediapipe/releases")
    print("  Look for: face_landmarker.task")
    print("")
    print("Option B: MediaPipe Official")
    print("  https://developers.google.com/mediapipe/solutions/vision/face_landmarker")
    print("  Download from: 'Model downloads' section")
    print("")
    print("Option C: Kaggle Datasets")
    print("  https://www.kaggle.com/datasets/")
    print("  Search for: 'face_landmarker' or 'mediapipe'")
    
    print("\n[METHOD 3] MANUAL SETUP")
    print("-"*70)
    print("1. Download face_landmarker.task from any source above")
    print("2. Copy to this directory:")
    print(f"   {SCRIPT_DIR}")
    print("3. Verify it's named exactly: face_landmarker.task")
    print("4. Run this script again")
    
    print("\n[IMPORTANT] FILE VERIFICATION")
    print("-"*70)
    print("After downloading, verify:")
    print("  • File name: face_landmarker.task (exact spelling)")
    print("  • File size: ~87-92 MB (should be large)")
    print("  • Location: " + str(SCRIPT_DIR))
    
    print("\n" + "="*70)
    print("AFTER SETTING UP MODEL:")
    print("="*70)
    print("Once the model file is in place, run this script again.")
    print("="*70 + "\n")
    
    return None


# ============================================================================
# STEP 4: INITIALIZE FACE LANDMARKER
# ============================================================================

def initialize_landmarker(model_path):
    """
    Creates and configures the Face Landmarker object.
    
    HOW IT WORKS:
    - BaseOptions: Contains the path to the pre-trained model
    - FaceLandmarkerOptions: Settings for detection behavior
    - Returns a configured Face Landmarker ready to detect faces
    
    PARAMETERS EXPLAINED:
    - model_asset_path: Where the .task file is located
    - running_mode: IMAGE = process single frames (best for webcam)
    - num_faces: How many faces to detect (1=faster)
    - min_face_detection_confidence: 0.0-1.0, higher = stricter
      • 0.5 = moderate (default)
      • 0.3 = easier to detect (more false positives)
      • 0.7 = harder to detect (fewer detections)
    """
    print("\nSTEP 2: INITIALIZING FACE LANDMARKER")
    print("-"*70)
    
    try:
        # Create base configuration with model path
        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path
        )
        
        # Create landmarker options (detection settings)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,  # Process frame-by-frame
            num_faces=1,  # Detect 1 face (faster than multiple)
            min_face_detection_confidence=0.2,  # Lower threshold helps on small or dim faces
            min_face_presence_confidence=0.2,   # More forgiving for quick detection
            min_tracking_confidence=0.2,        # Easier tracking when the face moves
            output_face_blendshapes=False,      # Skip blend shapes (not needed)
            output_facial_transformation_matrixes=False  # Skip transforms
        )
        
        # Create the actual Face Landmarker object
        landmarker = vision.FaceLandmarker.create_from_options(options)
        print("✓ Face Landmarker initialized successfully")
        return landmarker
        
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        print("\nCommon causes:")
        print("  1. Model file path is incorrect")
        print("  2. Model file is corrupted (download fresh copy)")
        print("  3. Insufficient permissions (try running as admin)")
        return None


# ============================================================================
# STEP 5: INITIALIZE CAMERA
# ============================================================================

def initialize_camera(width=640, height=480, fps=30):
    """
    Opens and configures the webcam.
    
    This version tries several Windows-friendly camera backends and indices.
    On some systems, cv2.VideoCapture(0) opens the object but does not provide
    frames until the camera backend is changed or a different index is used.
    """
    print("\nSTEP 3: INITIALIZING CAMERA")
    print("-"*70)
    
    backends = [
        cv2.CAP_DSHOW,   # DirectShow (often best on Windows)
        cv2.CAP_MSMF,    # Microsoft Media Foundation
        cv2.CAP_ANY      # generic fallback
    ]
    camera_indices = [0, 1, 2]

    for backend in backends:
        for index in camera_indices:
            try:
                cap = cv2.VideoCapture(index, backend)
                if not cap.isOpened():
                    cap.release()
                    continue

                # Give the camera a moment to initialize
                time.sleep(0.3)

                # Request camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)

                # Try to read one test frame
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

            except Exception as e:
                print(f"  • Backend {backend} index {index} failed: {e}")

    print("✗ Camera failed to open with any tested backend/index")
    print("\nTroubleshooting:")
    print("  1. Check if camera is connected")
    print("  2. Close other apps using camera (Zoom, Teams, etc.)")
    print("  3. Check camera permissions in system settings")
    print("  4. Try a different camera index manually: 0, 1, or 2")
    print("  5. Restart the computer if the camera is still unavailable")
    return None, None


# ============================================================================
# STEP 6: DRAWING FUNCTIONS
# ============================================================================

def draw_landmarks(frame, landmarks, frame_width, frame_height):
    """
    Draws facial landmarks on the video frame.
    
    HOW IT WORKS:
    - Iterate through each detected landmark
    - Convert normalized coordinates (0-1) to pixel coordinates
    - Draw circles at each point
    - Use RED for lips, GREEN for other features
    
    WHAT ARE NORMALIZED COORDINATES?
    - Range: 0.0 to 1.0 (instead of 0 to pixel_count)
    - 0.0 = left/top edge
    - 0.5 = middle
    - 1.0 = right/bottom edge
    - Formula: pixel_x = normalized_x * frame_width
    
    DRAWING PARAMETERS:
    - cv2.circle(frame, (x,y), radius, color, thickness)
    - radius: size of circle in pixels
    - color: (Blue, Green, Red) in BGR format
    - thickness: -1 = filled, 1-10 = outline thickness
    """
    if not landmarks:
        return 0
    
    landmark_count = 0
    
    # Draw each landmark
    for idx, landmark in enumerate(landmarks):
        # Convert normalized (0-1) to pixel coordinates
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        
        # Skip if point is outside frame
        if x < 0 or x >= frame_width or y < 0 or y >= frame_height:
            continue
        
        # Choose color and size
        if idx in LIP_LANDMARKS:
            color = (0, 0, 255)  # RED for lips
            radius = 3           # Slightly larger
        else:
            color = (0, 255, 0)  # GREEN for others
            radius = 2           # Regular size
        
        # Draw the landmark point
        cv2.circle(frame, (x, y), radius, color, -1)  # -1 = filled
        landmark_count += 1
    
    return landmark_count


def draw_info(frame, fps, landmark_count, face_detected):
    """
    Draws FPS, status, and instructions on the frame.
    
    TEXT POSITIONING:
    - (10, 30): Top-left for FPS
    - (10, 70): Below FPS for status
    - (10, height-20): Bottom-left for instructions
    - Coordinates are in pixels
    
    FONT PARAMETERS:
    - cv2.FONT_HERSHEY_SIMPLEX: Basic sans-serif font
    - scale: 1.0 = normal size, 0.5 = half size
    - thickness: 1-5 pixels
    """
    # FPS display (top-left)
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Face detection status
    if face_detected:
        status_text = f"✓ FACE DETECTED | Landmarks: {landmark_count}/{TOTAL_LANDMARKS}"
        status_color = (0, 255, 0)  # GREEN
    else:
        status_text = "✗ NO FACE DETECTED"
        status_color = (0, 0, 255)  # RED
    
    cv2.putText(frame, status_text, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Instructions (bottom-left)
    instructions = "Press 'Q' to quit"
    cv2.putText(frame, instructions, (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def print_console_info(detection_result, frame_num):
    """
    Prints detailed detection info to console.
    
    WHY PRINT EVERY N FRAMES?
    - Prevents console spam
    - Still provides regular feedback
    - Easier to read output
    - Reduces performance impact
    """
    if frame_num % 30 != 0:  # Print every 30 frames
        return
    
    print(f"\n[Frame {frame_num}]", end=" ")
    
    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        total = len(landmarks)
        
        # Count lip landmarks
        lip_count = sum(1 for i in range(len(landmarks)) if i in LIP_LANDMARKS)
        
        print(f"✓ FACE DETECTED")
        print(f"  Total landmarks: {total}/{TOTAL_LANDMARKS}")
        print(f"  Lip landmarks: {lip_count}")
        print(f"  Other landmarks: {total - lip_count}")
    else:
        print("✗ NO FACE")


# ============================================================================
# STEP 7: MAIN DETECTION LOOP
# ============================================================================

def main():
    """
    Main function: Runs the complete face detection pipeline.
    
    PIPELINE STEPS:
    1. Check model file
    2. Initialize Face Landmarker
    3. Initialize camera
    4. Main loop:
       a. Read frame from camera
       b. Convert to RGB format
       c. Create MediaPipe Image
       d. Detect landmarks
       e. Draw landmarks on frame
       f. Calculate and display FPS
       g. Show frame in window
       h. Check for quit key
    5. Cleanup resources
    """
    print("\n" + "="*70)
    print(" MediaPipe Face Mesh - Real-Time Face Landmark Detection ")
    print("="*70)
    
    # Check model file
    model_path = check_and_setup_model()
    if not model_path:
        sys.exit(1)
    
    # Initialize landmarker
    print("\nInitializing Face Landmarker...")
    landmarker = initialize_landmarker(model_path)
    if not landmarker:
        sys.exit(1)
    
    # Initialize camera
    cap, (frame_width, frame_height) = initialize_camera()
    if not cap:
        sys.exit(1)
    
    print("\n" + "="*70)
    print(" DETECTION STARTED - Press 'Q' to quit ")
    print("="*70 + "\n")
    
    # Initialize timing variables
    prev_time = time.time()
    current_fps = 0.0
    frame_count = 0
    
    # ====================================================================
    # MAIN LOOP - RUNS UNTIL USER QUITS
    # ====================================================================
    while True:
        # READ FRAME
        # ret = success flag, frame = image data
        ret, frame = cap.read()
        
        if not ret:
            print("\n✗ Error: Could not read frame")
            break
        
        # MIRROR EFFECT
        # Flip horizontally so it looks like a mirror (selfie camera)
        frame = cv2.flip(frame, 1)
        
        # FORMAT CONVERSION
        # OpenCV uses BGR, MediaPipe needs RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # CREATE MEDIAPIPE IMAGE
        # Wraps the numpy array in MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # DETECT LANDMARKS
        # This is where the ML model does its work
        detection_result = landmarker.detect(mp_image)
        
        # CALCULATE FPS
        # FPS = Frames Per Second = 1 / seconds_per_frame
        current_time = time.time()
        if (current_time - prev_time) > 0:
            current_fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # PROCESS DETECTION RESULTS
        landmark_count = 0
        face_detected = False
        
        if detection_result.face_landmarks:
            face_detected = True
            landmark_count = draw_landmarks(
                frame,
                detection_result.face_landmarks[0],
                frame_width,
                frame_height
            )
        
        # DRAW INFO ON FRAME
        draw_info(frame, current_fps, landmark_count, face_detected)
        
        # PRINT CONSOLE INFO (every 30 frames)
        print_console_info(detection_result, frame_count)
        frame_count += 1
        
        # SHOW FRAME IN WINDOW
        cv2.imshow('MediaPipe Face Mesh - NeuroSpeech Bridge', frame)
        
        # HANDLE KEYBOARD INPUT
        # waitKey(1) = wait 1ms for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
            print("\n✓ User quit")
            break
    
    # ====================================================================
    # CLEANUP - RELEASE RESOURCES
    # ====================================================================
    print("✓ Cleaning up...")
    cap.release()  # Close camera
    cv2.destroyAllWindows()  # Close all windows
    print("✓ Program terminated successfully\n")


# ============================================================================
# ENTRY POINT - RUN WHEN SCRIPT IS EXECUTED
# ============================================================================

if __name__ == "__main__":
    """
    This block only runs if script is executed directly (not imported).
    It wraps main() with error handling.
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n✗ Program interrupted (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# ADDITIONAL EXPLANATIONS
# ============================================================================

"""
COMPLETE BEGINNER'S GUIDE TO THIS CODE
========================================

1. WHAT IS MEDIAPIPE FACE MESH?
   • Framework by Google for detecting faces in real-time
   • Detects 468 points (landmarks) on a face
   • Runs on CPU (no GPU needed)
   • Very fast: ~100+ FPS on modern computers

2. WHAT IS A LANDMARK?
   • A single point on a face (corner of eye, tip of nose, etc.)
   • Represented as (x, y) coordinates
   • 468 total landmarks form a complete 3D face mesh
   • Useful for face recognition, emotion detection, etc.

3. WHAT DO THE NUMBERS MEAN?
   • 0-1 range: Normalized coordinates (relative to frame size)
   • 0 = left/top edge
   • 0.5 = middle
   • 1.0 = right/bottom edge
   • To convert: pixel_x = normalized_x * frame_width

4. COLOR SYSTEM (BGR)
   • OpenCV uses BGR (Blue, Green, Red) NOT RGB
   • (255, 0, 0) = BLUE
   • (0, 255, 0) = GREEN  ← We use this for faces
   • (0, 0, 255) = RED    ← We use this for lips
   • (255, 255, 255) = WHITE
   • (0, 0, 0) = BLACK

5. CONFIDENCE THRESHOLDS (0.0 - 1.0)
   • 0.0 = accept anything
   • 0.5 = moderate (default, good balance)
   • 1.0 = perfect detection only
   • Higher = fewer false detections but might miss faces
   • Lower = more detections but more errors

6. FPS (FRAMES PER SECOND)
   • How many frames processed per second
   • Higher = smoother video
   • Formula: FPS = 1 / (time_for_one_frame)
   • Display every frame but calculate every N frames for stability

7. RUNNING MODES
   • IMAGE: Process single frames (best for webcam) ← We use this
   • VIDEO: Process frame sequences with tracking
   • LIVE_STREAM: Async processing (complex, for real-time apps)

8. HOW TO MODIFY FOR YOUR NEEDS

   a) DETECT MULTIPLE FACES:
      Change: num_faces=1
      To: num_faces=5
      (Will be slower)

   b) STRICTER DETECTION (fewer false positives):
      Change: min_face_detection_confidence=0.5
      To: min_face_detection_confidence=0.8
      (Won't detect faces as easily)

   c) EASIER DETECTION (more detections):
      Change: min_face_detection_confidence=0.5
      To: min_face_detection_confidence=0.3
      (More false positives)

   d) HIGHER RESOLUTION:
      Change: width=640, height=480
      To: width=1280, height=720
      (More detail, slower processing)

   e) SAVE VIDEO:
      After camera init:
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      out = cv2.VideoWriter('output.mp4', fourcc, 30, (width, height))
      
      In main loop (after draw_info):
      out.write(frame)
      
      After loop (in cleanup):
      out.release()

   f) EXTRACT LANDMARK DATA:
      landmarks = detection_result.face_landmarks[0]
      for idx, landmark in enumerate(landmarks):
          x, y = landmark.x, landmark.y
          z = landmark.z  # Depth coordinate
          print(f"Landmark {idx}: ({x}, {y}, {z})")

9. COMMON ERRORS

   Error: "Model file not found"
   Fix: Download face_landmarker.task and place in script folder
   
   Error: "Cannot open camera"
   Fix: Check camera connection, close other apps using camera
   
   Error: "No face detected"
   Fix: Lower confidence threshold, improve lighting, move closer
   
   Error: "Slow FPS"
   Fix: Reduce resolution, close other apps, use smaller num_faces
   
   Error: "cv2.error: (-215:Assertion failed)"
   Fix: Check frame dimensions are correct

10. PERFORMANCE OPTIMIZATION TIPS
    • Use IMAGE mode (faster than VIDEO)
    • Detect 1 face (num_faces=1)
    • Lower resolution (640x480 instead of 1280x720)
    • Disable unnecessary outputs
    • Close background applications
    • Use GPU if available (not supported in basic MediaPipe)

11. LANDMARK INDICES (for reference)
    • 0: Mouth center
    • 1-2: Top/bottom of head
    • 33, 263: Eye centers
    • 61-80: Outer lips
    • 78-95: Inner lips
    • See LIP_LANDMARKS set for exact indices

12. NEXT STEPS AFTER THIS
    • Save landmark data to CSV file
    • Use landmarks for face recognition
    • Detect emotions from landmarks
    • Track facial movement over time
    • Implement 3D face reconstruction
"""


