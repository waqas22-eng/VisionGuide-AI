
import cv2
import numpy as np
import pyttsx3
import threading
import time
from ultralytics import YOLO

CONFIDENCE = 0.40
VOICE_COOLDOWN = 4.0

# Higher scores mean greater priority for voice warnings.
PRIORITY = {
    "truck": 50,
    "bus": 48,
    "train": 45,
    "car": 44,
    "motorcycle": 42,
    "bicycle": 40,
    "dog": 30,
    "person": 25,
    "horse": 25,
    "traffic light": 20,
    "stop sign": 20,
    "chair": 12,
    "bench": 12,
    "suitcase": 12,
    "backpack": 8,
    "bottle": 5,
    "umbrella": 8,
    "fire hydrant": 12,
    "potted plant": 10
}

print("Loading object detection model...")
detector = YOLO("yolo26n.pt")

print("Loading depth estimation model...")
depth_model = YOLO("yolo26n-depth.pt")

print("Both models loaded.")

# Keep all speech on a single worker thread.
import queue

speech_queue = queue.Queue(maxsize=2)


def speech_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.setProperty("volume", 1.0)

    while True:
        message = speech_queue.get()

        if message is None:
            break

        try:
            engine.say(message)
            engine.runAndWait()
        except Exception as error:
            print("Speech error:", error)

        speech_queue.task_done()


threading.Thread(
    target=speech_worker,
    daemon=True
).start()


def speak(message):
    # Avoid building up a long queue of outdated warnings.
    if speech_queue.full():
        return

    speech_queue.put(message)


def get_position(center_x, frame_width):
    if center_x < frame_width / 3:
        return "on your left"
    elif center_x > frame_width * 2 / 3:
        return "on your right"
    return "ahead"


def estimate_object_depth(depth_map, x1, y1, x2, y2):
    """Estimate depth from the inner part of a detected box."""

    height, width = depth_map.shape

    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    # Avoid box edges, which may include the background.
    margin_x = int((x2 - x1) * 0.20)
    margin_y = int((y2 - y1) * 0.20)

    left = min(x2 - 1, x1 + margin_x)
    right = max(left + 1, x2 - margin_x)
    top = min(y2 - 1, y1 + margin_y)
    bottom = max(top + 1, y2 - margin_y)

    region = depth_map[top:bottom, left:right]

    valid = region[
        np.isfinite(region) &
        (region > 0) &
        (region < 100)
    ]

    if valid.size == 0:
        return None

    # Median reduces the influence of unusual pixels.
    return float(np.median(valid))


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit

last_spoken = {}

print("VisionGuide depth + voice is running.")
print("Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break

        frame_height, frame_width = frame.shape[:2]

        # Detect objects.
        detection = detector.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=416,
            device="cpu",
            verbose=False
        )[0]

        # Estimate depth for the same camera frame.
        depth_result = depth_model.predict(
            source=frame,
            imgsz=384,
            device="cpu",
            verbose=False
        )[0]

        if depth_result.depth is None:
            print("ERROR: Depth model returned no depth map.")
            break

        depth_map = depth_result.depth.data.cpu().numpy()
        depth_map = np.squeeze(depth_map)

        if depth_map.ndim != 2:
            print("ERROR: Unexpected depth map shape:", depth_map.shape)
            break

        # Align depth pixels with the original camera frame.
        depth_map = cv2.resize(
            depth_map,
            (frame_width, frame_height),
            interpolation=cv2.INTER_LINEAR
        )

        display = detection.plot()
        candidates = []

        if detection.boxes is not None:
            for box in detection.boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                name = detector.names[class_id]

                # Only speak selected useful object classes.
                if name not in PRIORITY:
                    continue

                x1, y1, x2, y2 = [
                    int(value) for value in box.xyxy[0].tolist()
                ]

                center_x = (x1 + x2) // 2

                position = get_position(
                    center_x,
                    frame_width
                )

                distance = estimate_object_depth(
                    depth_map,
                    x1, y1, x2, y2
                )

                risk = PRIORITY[name]

                if position == "ahead":
                    risk += 20

                if distance is not None:
                    if distance < 1.2:
                        risk += 50
                    elif distance < 2.5:
                        risk += 30
                    elif distance < 4.0:
                        risk += 15

                candidates.append({
                    "name": name,
                    "position": position,
                    "distance": distance,
                    "risk": risk,
                    "confidence": confidence
                })

                if distance is not None:
                    label = f"{name}: ~{distance:.1f} m"
                    cv2.putText(
                        display,
                        label,
                        (max(5, x1), max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2
                    )

        # Evaluate the highest-priority objects first.
        candidates.sort(
            key=lambda item: item["risk"],
            reverse=True
        )

        now = time.time()

        for item in candidates:
            key = (item["name"], item["position"])

            if now - last_spoken.get(key, 0) < VOICE_COOLDOWN:
                continue

            name = item["name"]
            position = item["position"]
            distance = item["distance"]

            if distance is None:
                message = f"{name} {position}"

            elif distance < 1.2:
                message = f"Warning. {name} may be very close {position}"

            elif distance < 2.5:
                message = (
                    f"Caution. {name}, estimated "
                    f"{distance:.1f} meters {position}"
                )

            else:
                message = (
                    f"{name}, estimated "
                    f"{distance:.1f} meters {position}"
                )

            print("VOICE:", message)
            speak(message)
            last_spoken[key] = now
            break

        cv2.imshow(
            "VisionGuide - Object Detection and Depth",
            display
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("VisionGuide stopped.")