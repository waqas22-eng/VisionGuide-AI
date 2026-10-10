import cv2
import pyttsx3
import threading
import time
from ultralytics import YOLO


# =========================================================
# SETTINGS
# =========================================================

MODEL_NAME = "yolo26n.pt"

CONFIDENCE = 0.40
VOICE_COOLDOWN = 3.0


# =========================================================
# DANGER PRIORITY
# Lower number = higher priority
# =========================================================

PRIORITY = {
    "truck": 1,
    "bus": 2,
    "car": 3,
    "motorcycle": 4,
    "bicycle": 5,
    "train": 6,
    "person": 7,
    "dog": 8,
    "traffic light": 9,
    "stop sign": 10,
    "bench": 11,
    "chair": 12,
    "backpack": 13,
    "suitcase": 14,
    "cat": 15,
    "bottle": 16
}


# =========================================================
# TEXT TO SPEECH
# =========================================================

engine = pyttsx3.init()

engine.setProperty("rate", 165)
engine.setProperty("volume", 1.0)

voice_lock = threading.Lock()


def speak(message):

    def run_voice():

        with voice_lock:
            engine.say(message)
            engine.runAndWait()

    thread = threading.Thread(
        target=run_voice,
        daemon=True
    )

    thread.start()


# =========================================================
# FIND OBJECT POSITION
# =========================================================

def get_position(center_x, frame_width):

    left_area = frame_width * 0.33
    right_area = frame_width * 0.66

    if center_x < left_area:
        return "on your left"

    elif center_x > right_area:
        return "on your right"

    else:
        return "directly ahead"


# =========================================================
# ESTIMATE CLOSENESS
# =========================================================

def get_closeness(box_width, box_height, frame_width, frame_height):

    object_area = box_width * box_height
    frame_area = frame_width * frame_height

    percentage = object_area / frame_area

    if percentage >= 0.30:
        return "very close"

    elif percentage >= 0.15:
        return "close"

    elif percentage >= 0.07:
        return "nearby"

    else:
        return "far"


# =========================================================
# CREATE DANGER LEVEL
# =========================================================

def calculate_danger(
    object_name,
    position,
    closeness,
    confidence
):

    base_priority = PRIORITY.get(
        object_name,
        50
    )

    score = 0

    # Dangerous vehicle classes
    if object_name in [
        "truck",
        "bus",
        "car",
        "motorcycle",
        "bicycle",
        "train"
    ]:
        score += 50

    # Person
    elif object_name == "person":
        score += 30

    # Animals
    elif object_name in ["dog", "cat"]:
        score += 20

    else:
        score += 10

    # Directly ahead is more important
    if position == "directly ahead":
        score += 30

    elif position in [
        "on your left",
        "on your right"
    ]:
        score += 10

    # Closeness
    if closeness == "very close":
        score += 40

    elif closeness == "close":
        score += 25

    elif closeness == "nearby":
        score += 10

    # Confidence
    if confidence >= 0.75:
        score += 10

    # Priority adjustment
    score += max(0, 20 - base_priority)

    return score


# =========================================================
# CREATE VOICE MESSAGE
# =========================================================

def create_message(object_name, position, closeness, danger_score):

    if danger_score >= 100:

        return (
            f"Warning! "
            f"{object_name} "
            f"{position}, "
            f"{closeness}"
        )

    elif danger_score >= 70:

        return (
            f"Careful. "
            f"{object_name} "
            f"{position}, "
            f"{closeness}"
        )

    else:

        return (
            f"{object_name} "
            f"{position}"
        )


# =========================================================
# LOAD YOLO
# =========================================================

print("====================================")
print("VisionGuide AI")
print("Smart Blind Assistance System")
print("====================================")

print("Loading YOLO model...")

model = YOLO(MODEL_NAME)

print("YOLO model loaded successfully.")

# =========================================================
# OPEN CAMERA
# =========================================================

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    raise SystemExit


print("Camera started successfully.")
print("VisionGuide is running.")
print("Press Q to quit.")


# =========================================================
# SPEECH MEMORY
# =========================================================

last_spoken = {}


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    success, frame = camera.read()

    if not success:

        print("ERROR: Could not read frame.")

        break


    frame_height, frame_width = frame.shape[:2]


    # -----------------------------------------------------
    # YOLO DETECTION
    # -----------------------------------------------------

    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        imgsz=640,
        verbose=False
    )

    result = results[0]


    # Draw detections on screen
    annotated_frame = result.plot()


    detected_objects = []


    # -----------------------------------------------------
    # PROCESS DETECTED OBJECTS
    # -----------------------------------------------------

    if result.boxes is not None:

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            object_name = model.names[class_id]


            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            box_width = x2 - x1
            box_height = y2 - y1


            center_x = (
                x1 + x2
            ) // 2


            # Position
            position = get_position(
                center_x,
                frame_width
            )


            # Approximate closeness
            closeness = get_closeness(
                box_width,
                box_height,
                frame_width,
                frame_height
            )


            # Danger score
            danger_score = calculate_danger(
                object_name,
                position,
                closeness,
                confidence
            )


            detected_objects.append({

                "name": object_name,

                "confidence": confidence,

                "position": position,

                "closeness": closeness,

                "danger": danger_score
            })


    # -----------------------------------------------------
    # SORT BY DANGER
    # -----------------------------------------------------

    detected_objects.sort(
        key=lambda x: x["danger"],
        reverse=True
    )


    # -----------------------------------------------------
    # SPEAK THE MOST IMPORTANT OBJECT
    # -----------------------------------------------------

    current_time = time.time()


    for obj in detected_objects:

        object_name = obj["name"]


        # Skip objects that are not part
        # of our useful vocabulary
        if object_name not in PRIORITY:
            continue


        previous_time = last_spoken.get(
            object_name,
            0
        )


        # Voice cooldown
        if (
            current_time - previous_time
            < VOICE_COOLDOWN
        ):
            continue


        message = create_message(
            object_name,
            obj["position"],
            obj["closeness"],
            obj["danger"]
        )


        print(
            f"[VOICE] {message}"
        )


        speak(message)


        last_spoken[object_name] = current_time


        # Only one message at a time
        break


    # -----------------------------------------------------
    # SHOW VISUAL RESULT
    # -----------------------------------------------------

    cv2.imshow(
        "VisionGuide AI - Smart Assistance",
        annotated_frame
    )


    # -----------------------------------------------------
    # QUIT
    # -----------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# =========================================================
# CLEANUP
# =========================================================

camera.release()

cv2.destroyAllWindows()

print("VisionGuide AI stopped.")