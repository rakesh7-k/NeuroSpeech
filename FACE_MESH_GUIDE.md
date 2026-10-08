# MediaPipe Face Mesh - NeuroSpeech Bridge Setup Guide

## Overview
`face_mesh_test.py` is a complete real-time face landmark detection system using MediaPipe. It detects 468 facial landmarks in real-time and displays them on your webcam feed.

---

## What You Need to Do

### Step 1: Download the Model File (REQUIRED)

The script needs a pre-trained model file to work. This is a **one-time download**. Choose ONE method below:

#### **METHOD 1: Download from Kaggle (Recommended)**
1. Go to: https://www.kaggle.com/
2. Search for: **"face_landmarker"** or **"mediapipe face landmarker"**
3. Download the `face_landmarker.task` file (~90 MB)
4. Place it in: `D:\NEURO\neurospeech\` folder
5. **Important:** File must be named exactly `face_landmarker.task`

#### **METHOD 2: Download from GitHub**
1. Go to: https://github.com/google/mediapipe/releases
2. Look for releases with `face_landmarker` files
3. Download `face_landmarker.task`
4. Place in: `D:\NEURO\neurospeech\` folder

#### **METHOD 3: Download from MediaPipe Official**
1. Go to: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
2. Scroll to "Model downloads" section
3. Download the `.task` model file
4. Place in: `D:\NEURO\neurospeech\` folder

**Verification Checklist:**
- [ ] File is named: `face_landmarker.task` (exact spelling)
- [ ] File size is approximately 87-92 MB
- [ ] File is in: `D:\NEURO\neurospeech\`

### Step 2: Run the Script

After placing the model file, run:

```powershell
cd D:\NEURO\neurospeech
.\venv\Scripts\Activate.ps1
python face_mesh_test.py
```

Or directly with the full path:
```powershell
& "D:\NEURO\neurospeech\venv\Scripts\python.exe" "D:\NEURO\neurospeech\face_mesh_test.py"
```

### Step 3: What to Expect

When running successfully, you should see:
1. A window titled "MediaPipe Face Mesh - NeuroSpeech Bridge"
2. Your webcam feed with:
   - **GREEN dots** on all facial features
   - **RED dots** specifically on lips
   - **FPS counter** in top-left
   - **Face detection status** showing landmark count

Example output in console:
```
[Frame 30] ✓ FACE DETECTED
  Total landmarks: 468/468
  Lip landmarks: 40
  Other landmarks: 428
```

---

## Code Structure Explained

### Section 1: Imports (Lines 1-29)
```python
import cv2              # Computer vision library
import time             # For FPS calculation
import sys              # System operations
import numpy as np      # Numerical arrays
from pathlib import Path  # File path handling
```

**What this means for beginners:**
- These are "libraries" or "modules" - pre-written code we reuse
- Each import brings in specific functionality
- Like importing tools from a toolbox

### Section 2: Configuration (Lines 34-47)
```python
SCRIPT_DIR = Path(__file__).parent  # Current folder
MODEL_FILE = SCRIPT_DIR / "face_landmarker.task"  # Where model goes
LIP_LANDMARKS = set([...])  # Indices of lip points (out of 468 total)
TOTAL_LANDMARKS = 468  # Total number of face points detected
```

**What this means for beginners:**
- These are "constants" - values that don't change
- We define them once at the top
- Makes the code easier to modify later

### Section 3: Model Setup (Lines 52-120)
```python
def check_and_setup_model():
    """Function that checks if model file exists"""
    if MODEL_FILE.exists():
        return str(MODEL_FILE)
    # If not found, display instructions
```

**What this means for beginners:**
- A "function" is reusable code block
- `def` means "define this function"
- `"""..."""` is documentation (explains what the function does)
- Functions can return values

### Section 4: Initialize Landmarker (Lines 125-167)
```python
def initialize_landmarker(model_path):
    """Create the Face Landmarker object"""
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(...)
    landmarker = vision.FaceLandmarker.create_from_options(options)
    return landmarker
```

**What this means for beginners:**
- This sets up the AI model for detecting faces
- `BaseOptions`: Where to find the model file
- `FaceLandmarkerOptions`: Settings for how to detect
- Returns the configured landmarker object

### Section 5: Camera Setup (Lines 172-217)
```python
def initialize_camera(width=640, height=480, fps=30):
    """Open and configure webcam"""
    cap = cv2.VideoCapture(0)  # Open camera 0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # Set width
    # ... more configuration ...
    return cap, (actual_width, actual_height)
```

**What this means for beginners:**
- Opens your webcam
- Sets resolution and FPS
- `0` = first camera (usually your main webcam)
- Returns camera object and dimensions

### Section 6: Drawing Functions (Lines 222-307)
```python
def draw_landmarks(frame, landmarks, frame_width, frame_height):
    """Draw dots on each landmark"""
    for idx, landmark in enumerate(landmarks):
        x = int(landmark.x * frame_width)  # Convert to pixels
        y = int(landmark.y * frame_height)
        cv2.circle(frame, (x, y), radius, color, -1)  # Draw circle
```

**What this means for beginners:**
- `for idx, landmark in enumerate(...)` = loop through each landmark
- Convert from 0-1 range to pixel coordinates
- Draw colored circles (red for lips, green for others)

### Section 7: Main Loop (Lines 377-445)
```python
while True:  # Run forever until quit
    ret, frame = cap.read()  # Get frame from camera
    # ... process frame ...
    cv2.imshow('...', frame)  # Show in window
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # If user pressed Q
        break  # Exit loop
```

**What this means for beginners:**
- `while True:` = infinite loop
- Each iteration processes one frame
- `cv2.waitKey(1)` = wait 1ms for keyboard
- When user presses 'Q', break out of loop

---

## Code Walkthrough: Frame Processing

Here's what happens **every frame** (30 times per second):

1. **Read Frame**
   ```python
   ret, frame = cap.read()
   ```
   - Gets one image from camera
   - `ret` = success flag (True/False)
   - `frame` = image data (as numpy array)

2. **Mirror Effect**
   ```python
   frame = cv2.flip(frame, 1)  # 1 = flip horizontally
   ```
   - Like looking in a mirror (left/right reversed)
   - Better for user experience

3. **Format Conversion**
   ```python
   rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   ```
   - OpenCV uses BGR (Blue-Green-Red)
   - MediaPipe needs RGB (Red-Green-Blue)
   - `cvtColor` = "convert color"

4. **Create MediaPipe Image**
   ```python
   mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
   ```
   - Wraps numpy array in MediaPipe format
   - Prepares it for the AI model

5. **Detect Landmarks**
   ```python
   detection_result = landmarker.detect(mp_image)
   ```
   - **This is where the AI magic happens**
   - Model analyzes image and finds face landmarks
   - Returns 468 points on the face

6. **Draw Landmarks**
   ```python
   if detection_result.face_landmarks:
       draw_landmarks(frame, detection_result.face_landmarks[0], ...)
   ```
   - If a face was detected
   - Draw green dots for landmarks
   - Draw red dots for lips

7. **Display Information**
   ```python
   draw_info(frame, current_fps, landmark_count, face_detected)
   ```
   - Add FPS counter
   - Add face detection status
   - Add instructions

8. **Show Frame**
   ```python
   cv2.imshow('MediaPipe Face Mesh - NeuroSpeech Bridge', frame)
   ```
   - Display the frame in a window
   - User sees live video with landmarks

9. **Handle Quit**
   ```python
   key = cv2.waitKey(1) & 0xFF
   if key == ord('q'):  # User pressed Q
       break
   ```
   - Wait 1ms for key press
   - If 'Q' pressed, exit loop

---

## Understanding the 468 Landmarks

MediaPipe Face Mesh provides 468 facial landmarks grouped by region:

| Region | Landmarks | Count |
|--------|-----------|-------|
| Lips (outer) | 61-80 | ~20 |
| Lips (inner) | 78-95 | ~20 |
| Left Eye | 33, 246, 161, 160, 159, 158, 157, 173 | ~8 |
| Right Eye | 263, 466, 388, 387, 386, 385, 384, 398 | ~8 |
| Eyebrows | Multiple indices | ~20 |
| Nose | 1, 2, 98, 326, 327, 94, etc. | ~10 |
| Face Contour | 10, 338, 297, 332, 284, etc. | ~33 |
| Other | Remaining indices | ~360 |

The script highlights **lip landmarks in RED** to show you how to work with specific regions.

---

## Coordinate System Explained

### Normalized Coordinates (What MediaPipe Returns)
- Range: 0.0 to 1.0
- 0.0 = left/top edge
- 0.5 = middle
- 1.0 = right/bottom edge

Example:
```
normalized_x = 0.3  (30% from left)
normalized_y = 0.7  (70% from top)
frame_width = 640
frame_height = 480

pixel_x = 0.3 * 640 = 192
pixel_y = 0.7 * 480 = 336
```

### Converting to Pixels
```python
x = int(landmark.x * frame_width)
y = int(landmark.y * frame_height)
```

Why normalized? Because it's **scale-independent** - works with any camera resolution!

---

## Performance Tips

### For Better FPS:
1. **Lower resolution**
   ```python
   initialize_camera(width=320, height=240)  # Instead of 640x480
   ```
2. **Detect fewer faces**
   ```python
   num_faces=1  # Already optimized
   ```
3. **Close background apps** (YouTube, Discord, etc.)
4. **Use GPU** (if available - requires CUDA setup)

### For Better Detection:
1. **Lower confidence threshold**
   ```python
   min_face_detection_confidence=0.3  # Instead of 0.5
   ```
2. **Improve lighting**
3. **Move closer to camera**
4. **Ensure full face is visible**

### For Better Accuracy:
1. **Higher confidence threshold**
   ```python
   min_face_detection_confidence=0.8  # Instead of 0.5
   ```
2. **Higher resolution**
   ```python
   initialize_camera(width=1280, height=720)
   ```

---

## Troubleshooting

### Problem: Model File Not Found
**Solution:** Download `face_landmarker.task` and place it in `D:\NEURO\neurospeech\`

### Problem: No Face Detected
**Solutions:**
1. Improve lighting (move to well-lit area)
2. Move closer to camera
3. Lower confidence threshold (change 0.5 to 0.3)
4. Ensure entire face is visible

### Problem: Slow Performance (Low FPS)
**Solutions:**
1. Close other applications
2. Reduce resolution (width=320, height=240)
3. Disable background processing
4. Try different USB port for camera

### Problem: Camera Not Opening
**Solutions:**
1. Check if camera is connected
2. Close other apps using camera (Zoom, Teams, etc.)
3. Try different camera: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`
4. Restart computer

### Problem: Landmarks Not Drawing Correctly
**Solutions:**
1. Check if face is detected (status should say "✓ FACE DETECTED")
2. Ensure frame dimensions are correct
3. Check color format (BGR vs RGB)

---

## Modifying the Code

### Save Video Output
Add after `initialize_camera()`:
```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30, (frame_width, frame_height))
```

Add in main loop after `draw_info()`:
```python
out.write(frame)
```

Add in cleanup section:
```python
out.release()
```

### Save Landmark Data to CSV
Add at top:
```python
import csv
csv_file = open('landmarks.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
```

Add in main loop after detection:
```python
if detection_result.face_landmarks:
    landmarks = detection_result.face_landmarks[0]
    row = [f"{lm.x:.4f},{lm.y:.4f},{lm.z:.4f}" for lm in landmarks]
    csv_writer.writerow(row)
```

Add in cleanup:
```python
csv_file.close()
```

### Detect Multiple Faces
Change this line:
```python
num_faces=1
```
To:
```python
num_faces=5
```

Then modify the drawing loop:
```python
if detection_result.face_landmarks:
    for face_landmarks in detection_result.face_landmarks:  # Loop through all faces
        draw_landmarks(frame, face_landmarks, frame_width, frame_height)
```

---

## Next Steps for NeuroSpeech Bridge

Once you have face landmarks working:

1. **Feature Extraction**
   - Calculate distances between landmarks
   - Detect mouth opening
   - Detect eye blinking
   - Recognize facial expressions

2. **Audio Synthesis**
   - Use `edge-tts` to generate speech
   - Sync speech to mouth movement

3. **Lip Sync**
   - Use lip landmarks to detect talking
   - Generate appropriate audio

4. **Real-time Processing**
   - Save landmark data to file
   - Send landmarks over network
   - Integrate with speech recognition

---

## Resources

- **MediaPipe Official:** https://mediapipe.dev/
- **Face Mesh Documentation:** https://developers.google.com/mediapipe/solutions/vision/face_landmarker
- **OpenCV Documentation:** https://docs.opencv.org/
- **Python Documentation:** https://docs.python.org/3/
- **GitHub Repository:** https://github.com/google/mediapipe

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python face_mesh_test.py` | Run the script |
| `Q` key | Quit the application |
| `ESC` key | Also quits (alternative) |
| `Ctrl+C` | Force quit |

---

Good luck with your NeuroSpeech Bridge project! 🚀
