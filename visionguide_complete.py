import os
import time
import threading
import subprocess
from collections import deque

import cv2
from ultralytics import YOLO

try:
    import winsound
except ImportError:  # not Windows
    winsound = None



YOLO_MODEL_PATH = "yolo26n.pt"
DEPTH_MODEL_PATH = "yolo26n-depth.pt"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.35
ALERT_COOLDOWN = 6.0            
MIN_CONSECUTIVE_FRAMES = 2
MAX_ALERTS_PER_BATCH = 3
ENABLE_DEPTH_VIEW = False


CORRIDOR_LEFT = 0.30
CORRIDOR_RIGHT = 0.70


APPROACH_GROWTH = 1.35          
APPROACH_WINDOW = 1.0           
APPROACH_MIN_RATIO = 0.02       


PATH_CLEAR_DELAY = 2.0
PATH_CLEAR_COOLDOWN = 15.0
BEEP_COOLDOWN = 1.0
DARK_THRESHOLD = 40             
DARK_COOLDOWN = 30.0

LOG_FILE = "visionguide_log.csv"

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract = None
ocr_available = False

HAZARDS = 
    "car": ("Car", 0.02, 10),
    "truck": ("Truck", 0.02, 10),
    "bus": ("Bus", 0.02, 10),
    "motorcycle": ("Motorcycle", 0.02, 10),
    "train": ("Train", 0.02, 10),
    "bicycle": ("Bicycle", 0.02, 9),
    "person": ("Person", 0.03, 8),
    "dog": ("Dog", 0.03, 7),
    "cat": ("Cat", 0.03, 5),
    "traffic light": ("Traffic light", 0.01, 6),
    "stop sign": ("Stop sign", 0.01, 6),
    "fire hydrant": ("Fire hydrant", 0.02, 4),
    "chair": ("Chair", 0.04, 5),
    "bench": ("Bench", 0.04, 5),
    "couch": ("Couch", 0.05, 5),
    "dining table": ("Table", 0.05, 5),
    "suitcase": ("Suitcase", 0.03, 4),
    "backpack": ("Backpack", 0.03, 3),
    "umbrella": ("Umbrella", 0.03, 3),
    "potted plant": ("Plant", 0.03, 3),
    "skateboard": ("Skateboard", 0.02, 4),
}



_log_lock = threading.Lock()


def log_event(kind, details=""):
    """Append an event to a CSV file to measure false positives later."""
    try:
        with _log_lock:
            new_file = not os.path.isfile(LOG_FILE)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                if new_file:
                    f.write("time,event,details\n")
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                clean = str(details).replace('"', "'").replace("\n", " ")
                f.write(f'{stamp},{kind},"{clean}"\n')
    except Exception:
        pass




_speech_process = None
_speech_lock = threading.Lock()
muted = False
volume = 100
last_message = ""

PS_SCRIPT = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$s.Volume = [int]$env:VG_VOLUME; $s.Rate = 1; "
    "$s.Speak($env:VG_TEXT)"
)


def is_speaking():
    return (
        _speech_process is not None
        and _speech_process.poll() is None
    )


def speak(message, wait=False, urgent=False, force=False):
    """
    Speak through Windows speech.
    - urgent: interrupts whatever is currently being said.
    - force: speak even if muted (system / OCR messages).
    - Never blocks the camera loop unless wait=True.
    """
    global _speech_process, last_message

    if muted and not force:
        return False

    with _speech_lock:
        if is_speaking():
            if urgent or force:
                try:
                    _speech_process.kill()
                except Exception:
                    pass
            else:
                return False

        print("VOICE:", message, flush=True)
        last_message = str(message)

        try:
            env = os.environ.copy()
            env["VG_TEXT"] = str(message)
            env["VG_VOLUME"] = str(volume)
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

            _speech_process = subprocess.Popen(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    PS_SCRIPT,
                ],
                env=env,
                creationflags=flags,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            if wait:
                try:
                    _speech_process.wait(timeout=30)
                except Exception:
                    pass

            return True

        except Exception as error:
            print("VOICE ERROR:", repr(error), flush=True)
            return False




_beeping = False
_last_beep = 0.0


def beep(level):
    """HIGH = two high beeps, MEDIUM = one lower beep. Windows only."""
    global _beeping, _last_beep

    if winsound is None or muted or _beeping:
        return

    now = time.monotonic()
    if now - _last_beep < BEEP_COOLDOWN:
        return
    _last_beep = now

    if level == "HIGH":
        pattern = [(1400, 120), (1400, 120)]
    elif level == "MEDIUM":
        pattern = [(900, 150)]
    else:
        return

    def run():
        global _beeping
        _beeping = True
        try:
            for freq, duration in pattern:
                winsound.Beep(freq, duration)
                time.sleep(0.05)
        except Exception:
            pass
        finally:
            _beeping = False

    threading.Thread(target=run, daemon=True).start()




_ocr_running = False


def setup_ocr():
    global pytesseract, ocr_available

    try:
        import pytesseract as tess
        pytesseract = tess

        if os.path.isfile(TESSERACT_EXE):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

        pytesseract.get_tesseract_version()
        ocr_available = True
        print("OCR: Available.", flush=True)

    except Exception as error:
        ocr_available = False
        print("OCR: Unavailable:", repr(error), flush=True)
        print(
            "Install pytesseract and the Tesseract OCR application "
            "to enable text reading."
        )


def _ocr_worker(frame):
    global _ocr_running

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(
            gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC
        )

        text = pytesseract.image_to_string(gray)
        text = " ".join(text.split())

        if text:
            print("OCR TEXT:", text, flush=True)
            log_event("ocr", text[:220])
            speak("The text says: " + text[:220], force=True)
        else:
            speak("No readable text found.", force=True)

    except Exception as error:
        print("OCR ERROR:", repr(error), flush=True)
        speak("Sorry, I could not read the text.", force=True)

    finally:
        _ocr_running = False


def read_text_from_frame(frame):
    global _ocr_running

    if not ocr_available:
        speak(
            "Text reading is unavailable. Tesseract OCR is not configured.",
            force=True,
        )
        return

    if _ocr_running:
        return

    _ocr_running = True
    threading.Thread(
        target=_ocr_worker, args=(frame,), daemon=True
    ).start()


# ============================================================
# 4. OBJECT DETECTION, TRACKING AND POSITION
# ============================================================

# track_id -> deque of (time, area_ratio)
track_history = {}


def get_class_name(names, class_id):
    try:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id)).strip().lower()

        return str(names[class_id]).strip().lower()

    except (IndexError, KeyError, TypeError):
        return str(class_id).strip().lower()


def get_position(center_x, frame_width):
    if center_x < frame_width * CORRIDOR_LEFT:
        return "on your left"

    if center_x > frame_width * CORRIDOR_RIGHT:
        return "on your right"

    return "straight ahead"


def check_approaching(track_id, ratio, now):
    """True if this tracked object's box has grown quickly recently."""
    if track_id is None:
        return False

    history = track_history.setdefault(track_id, deque(maxlen=40))
    history.append((now, ratio))

    old = [
        (t, r) for t, r in history
        if now - t <= APPROACH_WINDOW
    ]

    if len(old) < 3:
        return False

    oldest_time, oldest_ratio = old[0]

    if now - oldest_time < 0.4 or oldest_ratio <= 0:
        return False

    return (
        ratio >= APPROACH_MIN_RATIO
        and ratio / oldest_ratio >= APPROACH_GROWTH
    )


def cleanup_history(active_ids, now):
    for tid in list(track_history):
        if tid in active_ids:
            continue

        history = track_history[tid]

        if not history or now - history[-1][0] > 3.0:
            del track_history[tid]


def extract_detections(result, frame_width, frame_height):
    """Extract classes, confidence, positions, areas, boxes, tracking."""
    detections = []

    if result.boxes is None:
        return detections

    frame_area = float(max(1, frame_width * frame_height))
    now = time.monotonic()

    try:
        ids = (
            result.boxes.id.tolist()
            if result.boxes.id is not None
            else [None] * len(result.boxes)
        )
    except Exception:
        ids = [None] * len(result.boxes)

    active_ids = set()

    for index, box in enumerate(result.boxes):
        try:
            confidence = float(box.conf[0].item())

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            class_id = int(box.cls[0].item())
            name = get_class_name(result.names, class_id)

            x1, y1, x2, y2 = [
                float(value) for value in box.xyxy[0].tolist()
            ]

            box_width = max(0.0, x2 - x1)
            box_height = max(0.0, y2 - y1)
            area_ratio = (box_width * box_height) / frame_area
            center_x = (x1 + x2) / 2.0

            track_id = ids[index]
            track_id = int(track_id) if track_id is not None else None

            if track_id is not None:
                active_ids.add(track_id)

            position = get_position(center_x, frame_width)

            detections.append(
                {
                    "name": name,
                    "confidence": confidence,
                    "position": position,
                    "in_path": position == "straight ahead",
                    "ratio": area_ratio,
                    "track_id": track_id,
                    "approaching": check_approaching(
                        track_id, area_ratio, now
                    ),
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                }
            )

        except Exception as error:
            print("Detection extraction error:", repr(error), flush=True)

    cleanup_history(active_ids, now)
    return detections




def assess_priority(item):
    """
    Experimental score from object category, position (path corridor),
    apparent size, confidence and whether it is approaching.
    This score does not establish actual danger or distance.
    """
    name = item["name"]
    hazard = HAZARDS.get(name)

    if hazard:
        label, _minimum_ratio, base_priority = hazard
    else:
        label = name.replace("_", " ").title()
        base_priority = 1

    position_bonus = 1.5 if item["in_path"] else 0.3
    size_bonus = min(item["ratio"] * 8.0, 2.0)
    confidence_bonus = max(0.0, min(item["confidence"], 1.0)) * 0.5
    approach_bonus = 2.5 if item.get("approaching") else 0.0

    score = (
        base_priority
        + position_bonus
        + size_bonus
        + confidence_bonus
        + approach_bonus
    )

    if score >= 10.0:
        level = "HIGH"
        color = (0, 0, 255)
    elif score >= 7.0:
        level = "MEDIUM"
        color = (0, 165, 255)
    else:
        level = "LOW"
        color = (0, 200, 0)

    return {
        "label": label,
        "score": score,
        "level": level,
        "color": color,
    }


def draw_detections(frame, detections):
    """Draw objects, priorities and the walking corridor."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    # Walking corridor lines
    for fraction in (CORRIDOR_LEFT, CORRIDOR_RIGHT):
        x = int(width * fraction)
        cv2.line(annotated, (x, 0), (x, height), (255, 255, 0), 1)

    for item in detections:
        x1, y1, x2, y2 = item["box"]
        details = assess_priority(item)
        color = details["color"]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        approach_text = " APPROACHING" if item["approaching"] else ""
        label = (
            f"{details['label']} | {details['level']} "
            f"({details['score']:.1f}) | {item['position']}"
            f"{approach_text}"
        )

        cv2.putText(
            annotated,
            label,
            (max(3, x1), max(40, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )

    status = "MUTED" if muted else f"VOL {volume}"
    cv2.putText(
        annotated,
        f"Q Quit | S Read | M Mute | R Repeat | +/- Vol | {status}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated




def update_stability(detections, seen_counts):
    """Require a hazard class to persist across frames before alerting."""
    present = {
        item["name"] for item in detections if item["name"] in HAZARDS
    }

    for name in list(seen_counts):
        if name not in present:
            del seen_counts[name]

    for name in present:
        seen_counts[name] = seen_counts.get(name, 0) + 1


def choose_warnings(detections, last_spoken, seen_counts):
    """
    Select eligible hazards in descending score order.
    Cooldown key is (name, position), so a person moving from left to
    center is announced again.
    Returns (list of (key, message), urgent_flag, strongest_level).
    """
    best_by_key = {}

    for item in detections:
        name = item["name"]

        if name not in HAZARDS:
            continue

        details = assess_priority(item)
        candidate = dict(item)
        candidate["priority_score"] = details["score"]
        candidate["priority_level"] = details["level"]

        key = (name, item["position"])
        current = best_by_key.get(key)

        if (
            current is None
            or candidate["priority_score"] > current["priority_score"]
        ):
            best_by_key[key] = candidate

    now = time.monotonic()
    candidates = []

    for key, item in best_by_key.items():
        name, position = key
        label, minimum_ratio, _base = HAZARDS[name]

        if item["ratio"] < minimum_ratio:
            continue

        if seen_counts.get(name, 0) < MIN_CONSECUTIVE_FRAMES:
            continue

        
        if (
            not item["in_path"]
            and not item["approaching"]
            and item["priority_level"] == "LOW"
        ):
            continue

        
        cooldown = ALERT_COOLDOWN / 2 if item["approaching"] else ALERT_COOLDOWN

        if now - last_spoken.get(key, 0.0) < cooldown:
            continue

        if item["approaching"]:
            message = f"Warning. {label} approaching {position}."
        elif item["priority_level"] == "HIGH":
            message = f"Warning. {label} detected {position}."
        else:
            message = f"{label} {position}."

        candidates.append(
            (
                item["priority_score"],
                item["ratio"],
                key,
                message,
                item["priority_level"],
                item["approaching"],
            )
        )

    candidates.sort(key=lambda v: (v[0], v[1]), reverse=True)
    chosen = candidates[:MAX_ALERTS_PER_BATCH]

    warnings = [(key, message) for _s, _r, key, message, _l, _a in chosen]
    urgent = any(level == "HIGH" or appr for _s, _r, _k, _m, level, appr in chosen)
    strongest = "NONE"

    if chosen:
        levels = [level for _s, _r, _k, _m, level, _a in chosen]
        if "HIGH" in levels:
            strongest = "HIGH"
        elif "MEDIUM" in levels:
            strongest = "MEDIUM"
        else:
            strongest = "LOW"

    return warnings, urgent, strongest


def path_is_blocked(detections):
    """True if a hazard object currently sits in the walking corridor."""
    for item in detections:
        if item["name"] not in HAZARDS:
            continue

        _label, minimum_ratio, _base = HAZARDS[item["name"]]

        if item["in_path"] and item["ratio"] >= minimum_ratio:
            return True

    return False




def main():
    global muted, volume

    print("=" * 60)
    print("VISIONGUIDE AI v5 - SMART HAZARD PRIORITY")
    print("Priority scores are experimental, not certified safety scores.")
    print("=" * 60)

    if not speak("VisionGuide AI is starting.", wait=True, force=True):
        print("WARNING: Check Windows speech settings.")

    setup_ocr()

    try:
        print("Loading YOLO model:", YOLO_MODEL_PATH, flush=True)
        model = YOLO(YOLO_MODEL_PATH)
        print("YOLO model loaded successfully.", flush=True)

    except Exception as error:
        print("YOLO ERROR:", repr(error), flush=True)
        print("Check that your YOLO model file is available.")
        return

    depth_model = None

    if ENABLE_DEPTH_VIEW:
        if not os.path.isfile(DEPTH_MODEL_PATH):
            print("Depth model file not found:", DEPTH_MODEL_PATH, flush=True)
        else:
            try:
                depth_model = YOLO(DEPTH_MODEL_PATH)
                print("Optional depth model loaded.", flush=True)
            except Exception as error:
                print("Depth model unavailable:", repr(error), flush=True)
                depth_model = None

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("ERROR: Camera could not be opened.", flush=True)
        speak("Camera could not be opened.", wait=True, force=True)
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_spoken = {}
    seen_counts = {}
    last_debug_time = 0.0
    frame_count = 0

    was_blocked = False
    last_blocked_time = 0.0
    last_path_clear = 0.0
    last_dark_warning = 0.0

    log_event("start", "VisionGuide v5 started")

    print(
        "\nCamera is active. Q quit | S read text | M mute | "
        "R repeat | +/- volume\n",
        flush=True,
    )

    speak("Camera is active. VisionGuide AI is ready.", wait=True, force=True)

    try:
        while True:
            success, frame = camera.read()

            if not success:
                time.sleep(0.1)
                continue

            frame_count += 1
            now = time.monotonic()
            result = None

            # Low-light check
            gray_small = cv2.cvtColor(
                cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY
            )
            if (
                gray_small.mean() < DARK_THRESHOLD
                and now - last_dark_warning > DARK_COOLDOWN
            ):
                speak("Camera view is too dark. Detection may be unreliable.")
                log_event("dark", f"mean={gray_small.mean():.1f}")
                last_dark_warning = now

            try:
                results = model.track(
                    source=frame,
                    persist=True,
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=640,
                    verbose=False,
                )
                result = results[0]

            except Exception as error:
                print("DETECTION ERROR:", repr(error), flush=True)

            if result is not None:
                detections = extract_detections(
                    result, frame.shape[1], frame.shape[0]
                )
            else:
                detections = []

            update_stability(detections, seen_counts)
            annotated = draw_detections(frame, detections)

            if now - last_debug_time >= 2.0:
                if detections:
                    parts = []

                    for item in detections:
                        details = assess_priority(item)
                        parts.append(
                            f"{item['name']} "
                            f"({item['confidence']:.2f}, "
                            f"{item['position']}, "
                            f"score {details['score']:.1f}"
                            f"{', approaching' if item['approaching'] else ''})"
                        )

                    print("DETECTED:", "; ".join(parts), flush=True)
                else:
                    print("No objects detected.", flush=True)

                last_debug_time = now

            # ---- Voice and beep alerts ----
            warnings, urgent, strongest = choose_warnings(
                detections, last_spoken, seen_counts
            )

            if warnings:
                if urgent or not is_speaking():
                    sentence = " ".join(m for _k, m in warnings)
                    spoken = speak(sentence, urgent=urgent)

                    if spoken:
                        stamp = time.monotonic()

                        for key, _m in warnings:
                            last_spoken[key] = stamp

                        log_event("alert", sentence)

                beep(strongest)

            # ---- Path clear message ----
            blocked = path_is_blocked(detections)

            if blocked:
                was_blocked = True
                last_blocked_time = now

            elif (
                was_blocked
                and now - last_blocked_time >= PATH_CLEAR_DELAY
            ):
                was_blocked = False

                if now - last_path_clear >= PATH_CLEAR_COOLDOWN:
                    if speak("Path clear."):
                        last_path_clear = now
                        log_event("path_clear")

            cv2.imshow("VisionGuide AI - Smart Hazard Priority", annotated)

            if depth_model is not None and frame_count % 5 == 0:
                try:
                    depth_results = depth_model.predict(
                        source=frame, imgsz=640, verbose=False
                    )
                    cv2.imshow(
                        "VisionGuide AI - Experimental Depth",
                        depth_results[0].plot(),
                    )

                except Exception as error:
                    print("Depth view error:", repr(error), flush=True)
                    depth_model = None

            
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("s"):
                read_text_from_frame(frame.copy())

            elif key == ord("m"):
                muted = not muted
                print("MUTED" if muted else "UNMUTED", flush=True)
                speak(
                    "Alerts muted." if muted else "Alerts on.",
                    force=True,
                )

            elif key == ord("r"):
                if last_message:
                    speak(last_message, force=True)

            elif key in (ord("+"), ord("=")):
                volume = min(100, volume + 10)
                speak(f"Volume {volume}.", force=True)

            elif key in (ord("-"), ord("_")):
                volume = max(10, volume - 10)
                speak(f"Volume {volume}.", force=True)

    except KeyboardInterrupt:
        print("Program interrupted.", flush=True)

    except Exception as error:
        print("MAIN PROGRAM ERROR:", repr(error), flush=True)
        log_event("error", repr(error))

    finally:
        camera.release()
        cv2.destroyAllWindows()

        if is_speaking():
            try:
                _speech_process.kill()
            except Exception:
                pass

        log_event("stop", "VisionGuide stopped")
        print("VisionGuide AI stopped.", flush=True)


if __name__ == "__main__":
    main()
