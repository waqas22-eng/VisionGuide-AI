AI Visual Assistant for Visually Impaired People
VisionGuide AI is an experimental computer-vision prototype intended to provide audio descriptions of selected objects in a camera view. It combines YOLO object detection, OpenCV, spoken announcements, approximate left/center/right image-position labels, heuristic object-priority scores, and optional OCR text reading.
Safety notice: This is a student/research prototype, not a certified mobility or safety device. It can miss objects, misclassify scenes, and produce incorrect position or priority labels. Bounding-box size is not a reliable measurement of real-world distance or collision risk. Do not use this prototype as the sole navigation aid or replace a cane, guide dog, or established mobility training.

Current Features
- Camera-based object detection using Ultralytics YOLO.
- Voice announcements for configured object classes.
- Approximate left, center, and right position descriptions in the camera image.
- Experimental, rule-based object-priority scoring and visual labels.
- Voice cooldowns intended to reduce repeated announcements.
- Optional OCR text reading through pytesseract and the separate Tesseract OCR application.
- Optional experimental depth-model hook, disabled by default.
Only claim features such as tracking or movement analysis as implemented after they have been added to the code and tested successfully.
Technology Stack
- Python 3.10 (the current development environment)
- OpenCV
- Ultralytics YOLO
- Windows PowerShell System.Speech for text-to-speech in the current version
- pytesseract plus the separate Tesseract OCR executable for optional text reading
Run on Windows (Current Prototype)
1. Install Python 3.10 and Git.
2. Open this repository in VS Code.
3. Create/activate a virtual environment if you do not already have one.
4. Install the Python dependencies:
   python -m pip install -r requirements.txt
5. Make sure the configured YOLO model file (yolo26n.pt) is available in the project folder or change YOLO_MODEL_PATH in visionguide_complete.py to the path of a supported model. Do not commit model weights unless you have checked the file size, licence, and redistribution terms.
6. For OCR, install Tesseract OCR separately and update TESSERACT_EXE in visionguide_complete.py to match its actual installation path. If OCR is not installed, object detection can still be tested, but text reading will be unavailable.
7. Run:
   python visionguide_complete.py
Controls
- Q: Quit the application.
- S: Read text from the current camera frame, when OCR is configured.
Hardware Implementation Roadmap (Future Work)
The current version is designed and tested on a Windows laptop. A wearable, Raspberry Pi-based version is a future development goal, not a feature that should be described as already implemented. The hardware and software will need to be adapted and tested before the laptop code can run reliably on a small board.
Possible Hardware
Initial edge-computing prototype:
- Raspberry Pi 5 (an 8 GB version is a reasonable starting point for development, subject to budget and measured performance).
- A compatible camera, such as Raspberry Pi Camera Module 3 Wide for a broader view, or a supported USB webcam for a simpler first test.
- A compatible camera cable for the exact Pi and camera connectors.
- Small speaker, earphones, or an open-ear audio device so that the user can still hear surrounding traffic and people.
- Suitable USB-C power supply for bench testing; for a portable prototype, a properly rated battery/power solution with safe charging and sufficient current capacity.
- Ventilated enclosure and cooling appropriate for sustained processing.
- A secure mount or small belt/chest pack for the computer and battery. Avoid attaching a heavy board or battery directly to ordinary glasses.
- Optional physical button for reading text or enabling/disabling announcements.
Possible Edge-AI Acceleration
Real-time YOLO inference on a small computer may be slower than on a laptop. First benchmark the smaller model and input resolution on the Raspberry Pi CPU. If performance is insufficient, investigate an accelerator supported by the chosen Raspberry Pi model, such as the Raspberry Pi AI HAT+ ecosystem, or a compatible AI camera.
Important compatibility note: An accelerator does not automatically run every .pt model. For an AI HAT+/Hailo-based setup, the model and inference pipeline may need conversion to a supported format and use of the appropriate runtime. The Raspberry Pi AI Camera uses its own supported inference pipeline, so it also requires adaptation rather than simply changing the camera index. Confirm current compatibility and installation instructions before buying hardware.
Official documentation:
- Raspberry Pi 5
- Raspberry Pi camera documentation
- Raspberry Pi camera software
- Raspberry Pi AI Camera
- Raspberry Pi AI software and accelerators
Software Changes Required for Raspberry Pi
The current program uses Windows PowerShell's System.Speech for voice output. That speech implementation will not work unchanged on Raspberry Pi OS. A Linux-compatible speech engine will be needed, for example espeak-ng or another supported text-to-speech service.
Likely adaptation tasks:
1. Install a 64-bit Raspberry Pi OS version supported by the selected board and AI hardware.
2. Verify the camera independently using Raspberry Pi's current camera tools. For a CSI camera, follow the official camera and Picamera2 documentation; for a USB webcam, test the device with OpenCV.
3. Replace the Windows-only PowerShell speak() function with a Linux-compatible speech implementation.
4. Install a compatible Python, PyTorch/Ultralytics runtime and model for the selected architecture; confirm package support for the chosen OS before installation.
5. Install Tesseract OCR and pytesseract on the Pi if OCR is required.
6. Lower camera resolution and inference frequency as needed to maintain usable latency and temperature.
7. Benchmark end-to-end latency, inference rate, battery life, heat, false alarms, and missed detections before attempting any wearable trial.
8. Keep processing local where practical to reduce dependence on internet connectivity.
Suggested Development Stages
1. Stage 1 — Laptop demo: Stabilize detection, voice announcements, OCR, and documented test cases.
2. Stage 2 — Bench prototype: Run the camera, inference, and audio pipeline on Raspberry Pi on a desk, powered from a suitable supply.
3. Stage 3 — Portable prototype: Add a secure enclosure, battery, accessible controls, and open-ear audio; measure runtime and heat.
4. Stage 4 — Controlled testing: Test with stationary objects and recorded video in a safe, controlled environment. Use a sighted supervisor during trials. Do not test by walking near roads or stairs.
5. Stage 5 — Evaluation: Record latency, detection performance, false positives, missed objects, audio clarity, and battery performance. Any claim of benefit or safety requires appropriate validation.
Evaluation Plan
Before an exhibition, run repeatable tests and record actual results. For example, create a small test set of videos or controlled scenes and report how many trials correctly detected each selected class under defined lighting, background, and distance conditions. Do not invent accuracy, FPS, latency, or battery-life results. Clearly distinguish measured results from intended future capabilities.
Limitations
- Detection depends on visible objects, lighting, camera placement, and the model's supported classes.
- Left/center/right describes image position, not necessarily the user's physical left/right if the camera is rotated or mirrored.
- Priority levels are hand-designed demonstration scores, not a trained or validated hazard model.
- A bounding box does not provide dependable metric distance; actual collision risk requires substantially more sensing and validation.
- Object tracking and motion labels, if added, can fail when an object is occluded, the camera moves, or tracker identities change.
- Tesseract OCR may fail on blurry, angled, small, or poorly lit text.
- Raspberry Pi deployment, wearable power design, and any AI accelerator integration require additional engineering and testing.
Future Improvements
- Track object identities across frames and evaluate image motion.
- Add a properly evaluated depth or distance-sensing component.
- Improve alert scheduling and user controls to reduce speech overload.
- Explore voice-based scene questions after the core pipeline is stable.
- Move the pipeline to Raspberry Pi and validate camera, inference, speech, thermal behaviour, and power use.
Author
Waqas Ali Khan
Project Status
Educational prototype in development. Hardware deployment and independent navigation capability are not claimed