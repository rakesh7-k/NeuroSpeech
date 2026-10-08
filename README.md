# NeuroSpeech

NeuroSpeech is an experimental, webcam-based communication prototype. It uses
facial and lip movement features to recognize a small set of calibrated phrases
and can speak recognized phrases aloud using text-to-speech.

> **Important:** This is a research/demo project, not a medical device or a
> replacement for reliable assistive communication. Recognition depends on
> webcam quality, lighting, calibration, and the user. Do not rely on it for
> emergency communication.

## How it works

1. MediaPipe Face Landmarker detects facial landmarks from webcam frames.
2. `feature_extractor.py` converts lip and jaw movement into feature sequences.
3. `calibration_recorder.py` records labeled examples for supported phrases.
4. `dtw_classifier.py` compares incoming movement with calibration examples
   using Dynamic Time Warping (DTW).
5. `neurospeech_bridge.py` connects live recognition to speech output through
   `tts_engine.py` and Edge TTS.

The project also includes a real-time landmark visualizer, diagnostic tools,
and calibration quality analysis.

## Requirements

- Python 3.10 or newer
- A webcam
- Windows, macOS, or Linux with a graphical desktop for webcam windows
- Internet access for Edge TTS speech synthesis

Install dependencies in a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```sh
source .venv/bin/activate
```

## Get the MediaPipe model

Download the Face Landmarker task model from the
[official MediaPipe guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker),
then save it in this project directory as `face_landmarker.task`. The model is
not included in this repository.

## Run

Start with the landmark visualizer to check the camera and model:

```sh
python face_mesh_test.py
```

Record your own phrase examples, then start the communication bridge:

```sh
python calibration_recorder.py
python neurospeech_bridge.py
```

The recorder writes `calibration.json` in the project directory. The bridge
requires that calibration file. Recognition quality is user- and setup-specific;
record fresh examples if phrases are not recognized reliably.

Useful offline diagnostics:

```sh
python evaluate_dtw.py
python calibration_quality_analyzer.py
```

These diagnostics use the local calibration data. `evaluate_dtw.py` reports an
error if no valid calibration file is available.

## Project files

| File | Purpose |
| --- | --- |
| `face_mesh_test.py` | Visualize real-time face landmarks and lip points |
| `feature_extractor.py` | Extract normalized face/lip movement features |
| `calibration_recorder.py` | Record labeled phrase examples |
| `dtw_classifier.py` | Compare movement sequences and classify phrases |
| `neurospeech_bridge.py` | Run live recognition and speech output |
| `tts_engine.py` | Text-to-speech integration |
| `lip_landmark_extractor.py` | Standalone lip landmark utility |
| `evaluate_dtw.py` | Evaluate local calibration examples |
| `calibration_quality_analyzer.py` | Analyze calibration quality |
| `FACE_MESH_GUIDE.md` | Face landmark setup and implementation guide |
| `QUICK_REFERENCE.md` | Landmark quick reference |
| `SUMMARY.md` | Project notes and setup summary |

## Privacy

Calibration recordings describe facial movement and may be sensitive biometric
data. They are created locally and are intentionally excluded from Git. Review
and protect your local `calibration.json` and generated reports; only share them
if you have appropriate consent.

## License

No license is currently specified. Contact the repository owner before
redistributing or reusing this project.
