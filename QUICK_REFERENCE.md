# 🎯 Quick Reference - Face Mesh Detection

## Files Created

```
D:\NEURO\neurospeech\
├── face_mesh_test.py           ← Main detection script
├── FACE_MESH_GUIDE.md          ← Complete documentation
├── QUICK_REFERENCE.md          ← This file
├── neurospeech_bridge.py       ← Original OpenCV version
├── venv\                        ← Virtual environment
└── face_landmarker.task        ← MODEL FILE (you need to download this)
```

---

## ⚡ Quick Start

### 1. Download Model (Required - One Time)
- File: `face_landmarker.task` (~90 MB)
- Download from: Kaggle, GitHub, or MediaPipe official
- Place in: `D:\NEURO\neurospeech\`

### 2. Run Script
```powershell
cd D:\NEURO\neurospeech
.\venv\Scripts\Activate.ps1
python face_mesh_test.py
```

### 3. What You'll See
- Webcam window with face detection
- GREEN dots on face features
- RED dots on lips
- FPS counter in top-left
- Press 'Q' to exit

---

## 📊 The 468 Landmarks

| Feature | Indices | Count |
|---------|---------|-------|
| **Face Outline** | 10, 338, 297... | 33 |
| **Left Eye** | 33, 246, 161... | 8 |
| **Right Eye** | 263, 466, 388... | 8 |
| **Eyebrows** | 70-105, 334-363 | ~20 |
| **Nose** | 1-30 | ~30 |
| **Mouth** | 61-80 (outer), 78-95 (inner) | ~40 |
| **Other** | Remaining | ~329 |
| **TOTAL** | 0-467 | **468** |

---

## 🎨 Color Coding

```
GREEN: All other facial features
RED:   Lips (special highlighting)
```

---

## 📈 Understanding Output

### Console Output (Every 30 Frames)
```
[Frame 30] ✓ FACE DETECTED
  Total landmarks: 468/468
  Lip landmarks: 40
  Other landmarks: 428
```

### Screen Display
- **Top-left:** FPS counter
- **Below FPS:** Face detection status
- **Bottom-left:** Instructions
- **Over face:** Colored landmark dots

---

## 🔧 Common Modifications

### Detect Multiple Faces
```python
num_faces=5  # In initialize_landmarker()
```

### Lower Detection Threshold (Easier Detection)
```python
min_face_detection_confidence=0.3  # Default is 0.5
```

### Higher Resolution
```python
initialize_camera(width=1280, height=720)  # Default 640x480
```

### Save Video
```python
# After initialize_camera():
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30, (width, height))

# In loop after draw_info():
out.write(frame)

# In cleanup:
out.release()
```

### Extract Landmark Data
```python
if detection_result.face_landmarks:
    landmarks = detection_result.face_landmarks[0]
    for idx, landmark in enumerate(landmarks):
        x, y, z = landmark.x, landmark.y, landmark.z
        print(f"Landmark {idx}: ({x:.3f}, {y:.3f}, {z:.3f})")
```

---

## 📐 Coordinate System

### Normalized Coordinates (0.0 - 1.0)
```
(0, 0) ─── Top-left
 │ y
 │
 └─── x ──→ (1, 1) ─ Bottom-right

0.5, 0.5 = Center of frame
```

### Convert to Pixel Coordinates
```python
pixel_x = int(landmark.x * frame_width)
pixel_y = int(landmark.y * frame_height)
```

---

## 🎯 Landmark Indices Reference

```
# Face Outline
10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109

# Left Eye
33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173

# Right Eye  
263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398

# Nose
1, 2, 98, 326, 327, 94, 141, 329, 437, 438, 439, 440

# Lips (Outer)
61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146

# Lips (Inner)
78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95
```

---

## 🚨 Troubleshooting Checklist

| Issue | Cause | Fix |
|-------|-------|-----|
| Model not found | Missing download | Download face_landmarker.task |
| No face detected | Low lighting | Move to bright area |
| Camera won't open | In use by another app | Close other apps |
| Low FPS | High resolution | Use 640x480 or lower |
| Script crashes | Wrong Python path | Use venv: `.\venv\Scripts\Activate.ps1` |
| Landmarks misaligned | Wrong resolution settings | Check frame dimensions |

---

## 🔄 Performance Settings

### Fast Mode (Low Quality)
```python
initialize_camera(width=320, height=240)
min_face_detection_confidence=0.3
num_faces=1
```

### Balanced Mode (Recommended)
```python
initialize_camera(width=640, height=480)  # Default
min_face_detection_confidence=0.5         # Default
num_faces=1                                # Default
```

### High Quality Mode (Slow)
```python
initialize_camera(width=1280, height=720)
min_face_detection_confidence=0.8
num_faces=1
```

---

## 🎓 Code Structure Overview

```
face_mesh_test.py
├── Imports (line 1-29)
├── Configuration (line 34-47)
├── check_and_setup_model() (line 52-120)
├── initialize_landmarker() (line 125-167)
├── initialize_camera() (line 172-217)
├── draw_landmarks() (line 222-266)
├── draw_info() (line 271-305)
├── print_console_info() (line 310-330)
├── main() (line 335-447)
└── Entry point: if __name__ == "__main__" (line 452-465)
```

---

## 📝 Key Concepts

### What is MediaPipe?
- Framework by Google for ML perception tasks
- Pre-trained models for faces, hands, poses, etc.
- Runs on CPU (no GPU needed)
- Fast: 100+ FPS on modern hardware

### What is Face Mesh?
- MediaPipe solution for facial landmark detection
- Detects 468 points on a human face
- Returns normalized coordinates (0-1 range)
- Includes 3D information (x, y, z)

### What are Landmarks?
- Individual points on the face (eyes, nose, mouth, etc.)
- Each landmark has (x, y, z) coordinates
- x, y = 2D screen position
- z = depth (3D information)

### What is the Model File?
- Pre-trained neural network weights
- `face_landmarker.task` contains the AI model
- ~90 MB binary file
- Downloaded once, reused forever

---

## 🚀 Next Steps

### After Verification
1. ✅ Run face_mesh_test.py successfully
2. ✅ See face landmarks on screen
3. ✅ Verify RED lips and GREEN other features
4. ✅ Check FPS display

### Then Try
1. Modify detection confidence
2. Experiment with different resolutions
3. Extract landmark data
4. Save video output
5. Detect multiple faces

### For NeuroSpeech Integration
1. Extract lip landmarks for mouth shape
2. Detect mouth opening for speech sync
3. Extract other features for expression
4. Combine with edge-tts for audio
5. Sync speech to facial movement

---

## 📞 Quick Commands

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run main script
python face_mesh_test.py

# Check version
python -c "import mediapipe; print(mediapipe.__version__)"

# Verify imports
python -c "import cv2, mediapipe; print('✓ All imports OK')"

# List cameras (try different indices)
python -c "import cv2; cv2.VideoCapture(0); cv2.VideoCapture(1)"

# Deactivate virtual environment
deactivate
```

---

## 📚 Resources

| Resource | Link |
|----------|------|
| MediaPipe | https://mediapipe.dev/ |
| Face Mesh Docs | https://developers.google.com/mediapipe/solutions/vision/face_landmarker |
| OpenCV | https://docs.opencv.org/ |
| Python | https://docs.python.org/3/ |
| GitHub | https://github.com/google/mediapipe |

---

## ✅ Verification Checklist

- [ ] Model file `face_landmarker.task` downloaded (~90 MB)
- [ ] Model file placed in `D:\NEURO\neurospeech\`
- [ ] Virtual environment activated
- [ ] Script runs without errors
- [ ] Webcam window appears
- [ ] Face landmarks visible (GREEN dots)
- [ ] Lips highlighted (RED dots)
- [ ] FPS displayed in top-left
- [ ] Can quit with 'Q' key
- [ ] Console shows detection info

---

Good luck! 🎉 You now have a complete face landmark detection system ready for NeuroSpeech Bridge!
