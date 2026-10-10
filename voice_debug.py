
import pyttsx3

print("Initializing speech engine...")

try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    engine.setProperty("volume", 1.0)

    voices = engine.getProperty("voices")
    print("Available voices:", len(voices))

    engine.say("VisionGuide AI voice test. Can you hear me?")
    engine.runAndWait()

    print("Speech test finished.")

except Exception as error:
    print("VOICE ERROR:", repr(error))