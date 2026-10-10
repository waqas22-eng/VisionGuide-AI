import pyttsx3

engine = pyttsx3.init()

engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)

engine.say("VisionGuide AI is ready. I can see objects around you.")
engine.runAndWait()

print("Voice test completed.")