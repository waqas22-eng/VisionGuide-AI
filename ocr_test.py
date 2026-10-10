import cv2
import pytesseract
import pyttsx3
import os

# Set the Windows Tesseract executable path
tesseract_path = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if not os.path.exists(tesseract_path):
    print("ERROR: Tesseract was not found.")
    print("Check the installation location.")
    raise SystemExit

pytesseract.pytesseract.tesseract_cmd = tesseract_path

print("Tesseract version:", pytesseract.get_tesseract_version())

# Initialize voice
engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)

# Open camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit

print("OCR is ready.")
print("Show printed English text to the camera.")
print("Press S to read the text aloud.")
print("Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera.")
            break

        cv2.putText(
            frame,
            "Press S to read text | Q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.imshow("VisionGuide AI - OCR Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            print("Reading text...")

            # Prepare the captured camera frame
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.resize(
                gray,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_CUBIC
            )

            processed = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]

            # Recognize English printed text
            text = pytesseract.image_to_string(
                processed,
                lang="eng",
                config="--oem 3 --psm 6"
            )

            text = " ".join(text.split())

            if text:
                print("Recognized text:", text)

                # Keep the spoken message manageable
                message = text[:300]
                engine.say(message)
                engine.runAndWait()

            else:
                print("No readable text found.")
                engine.say("Sorry, no readable text found.")
                engine.runAndWait()

        elif key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("OCR test finished.")