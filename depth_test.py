
import cv2
import numpy as np
from ultralytics import YOLO

print("Loading VisionGuide depth model...")
print("The first run may download model weights.")

model = YOLO("yolo26n-depth.pt")

print("Depth model loaded successfully.")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit

print("Camera started.")
print("Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break

        # Estimate depth from the camera image.
        results = model.predict(
            source=frame,
            imgsz=512,
            device="cpu",
            verbose=False
        )

        result = results[0]

        if result.depth is None:
            print("ERROR: The model returned no depth map.")
            break

        # Get the predicted depth map as a NumPy array.
        depth_map = result.depth.data.cpu().numpy()

        # Ensure the map is a 2D image.
        depth_map = np.squeeze(depth_map)

        # Create a visualization with a fixed 0-10 scale.
        # The colors are only a visual estimate.
        depth_map = np.nan_to_num(
            depth_map,
            nan=10.0,
            posinf=10.0,
            neginf=0.0
        )

        depth_display = np.clip(depth_map, 0, 10)
        depth_display = (
            depth_display / 10.0 * 255
        ).astype(np.uint8)

        depth_display = cv2.resize(
            depth_display,
            (frame.shape[1], frame.shape[0])
        )

        depth_colored = cv2.applyColorMap(
            depth_display,
            cv2.COLORMAP_TURBO
        )

        cv2.imshow("VisionGuide - Live Camera", frame)
        cv2.imshow("VisionGuide - Estimated Depth", depth_colored)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Depth test stopped.")