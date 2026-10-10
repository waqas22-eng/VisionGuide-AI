 VisionGuide AI 

AI Visual Assistant for Visually Impaired People

VisionGuide AI is a computer vision project designed to help visually impaired people understand their surroundings through object detection and voice announcements. It uses a camera, YOLO, OpenCV, and text-to-speech to describe supported objects and their positions in the camera view.

This is an experimental software prototype developed for learning, research, and university project exhibitions.

 Features

- Real-Time Object Detection: Detects supported objects using a pretrained YOLO model.
- Voice Alerts: Announces detected objects using Windows built-in speech.
- Directional Descriptions: Describes objects as being on the left, right, or straight ahead in the camera image.
- Smart Hazard Priority: Assigns experimental priority scores based on object category, image position, apparent size, and confidence.
- Visual Priority Labels: Displays object labels, priority levels, and scores on the camera view.
- Text Reading (OCR): Reads visible text aloud using Tesseract OCR when configured.
- Optional Experimental Depth View: Provides an optional model-based view when a compatible model is available.

 Technologies Used

- Python
- OpenCV
- Ultralytics YOLO
- PyTorch (used by the YOLO framework)
- pytesseract
- Tesseract OCR
- Windows PowerShell Speech Synthesis

 Installation and Setup



 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Configure the YOLO model

Make sure the YOLO model file configured in `visionguide_complete.py` is available. If it is not present locally, follow the model's official download or installation instructions.

 5. Configure OCR (optional)

Install Tesseract OCR separately on Windows and ensure its executable path matches the path configured in the Python code.

OCR will be unavailable if its required software or Python package is missing.

 6. Run the application

```bash


Allow camera access if Windows requests permission.

 Controls

| Key | Action |
|---|---|
| `Q` | Quit the application |
| `S` | Read visible text aloud |

 Project Structure


VisionGuide-AI/
├── visionguide_complete.py
├── requirements.txt
├── README.md
├── .gitignore


 How It Works

1. The camera captures the surroundings.
2. YOLO detects supported objects and their bounding boxes.
3. OpenCV displays the detections and position descriptions.
4. The priority engine calculates experimental scores for eligible objects.
5. The voice system announces selected objects.
6. When OCR is configured, the user can press `S` to read visible text.

Future Improvements

- Integrate Raspberry Pi hardware for a portable prototype.
- Connect a compatible camera module or USB camera.
- Replace Windows speech synthesis with a Linux-compatible voice engine.
- Optimize the detection model for Raspberry Pi performance.
- Investigate a compatible AI accelerator for faster inference.
- Improve object tracking and movement analysis.
- Explore more reliable depth sensing.
- Add voice-based questions about the surrounding scene.
- Develop a wearable enclosure, audio output, and suitable battery system.

Future Hardware Plan

The long-term goal is to investigate a wearable assistive device using a Raspberry Pi, compatible camera, audio output, power supply, and optionally an AI accelerator. Hardware integration and performance testing will be required before this can become a practical wearable prototype.

Limitations and Safety

- Object detection may miss objects or produce incorrect predictions.
- Directional descriptions refer to positions within the camera image, not guaranteed real-world directions.
- Priority scores are experimental rules, not a validated hazard prediction model.
- Bounding-box size does not reliably measure physical distance.
- Performance depends on lighting, camera quality, hardware, and model configuration.
- This prototype is not a certified mobility or safety device.

VisionGuide AI must not be used as the sole navigation aid or replace a white cane, guide dog, or established mobility training.

 Author

Waqas Ali Khan

AI Student | Aspiring AI/ML Engineer

 Project Status

Educational and research prototype. Features should be described as operational only after testing them on the target system.


