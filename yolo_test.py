import cv2
from ultralytics import YOLO

print("Loading YOLO model...")

model = YOLO("yolo26n.pt")

print("YOLO model loaded successfully.")
print("Starting camera...")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    results = model.predict(
        source=frame,
        conf=0.35,
        imgsz=640,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("VisionGuide AI - Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("VisionGuide object detection stopped.")