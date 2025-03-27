import os
import cv2
import time
import speech_recognition as sr
import google.generativeai as genai
import mediapipe as mp
import threading
from playsound import playsound
from elevenlabs.client import ElevenLabs
import keyboard

# Configure APIs
genai.configure(api_key="AIzaSyC_ghxOYuxZ8tJKzpFrnNzmnGMvHHckfVc")
client = ElevenLabs(api_key="sk_2f74c92623b7f2706479eec686d16304cd91f1ab38b314c9")

# MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# User chooses interview topic
topic = input("Enter the interview topic: ")

responses = []
video_running = True
stop_recording = False

def record_video():
    global video_running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter("interview_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
    print("\n🔴 Recording video...")
    while video_running:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow("Interview Recording", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def generate_question():
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"Generate one interview question for a {topic} role."
    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "No question generated."
    except:
        return "Error generating question."

def stop_audio_recording():
    global stop_recording
    input("Press Enter to stop recording early...")
    stop_recording = True

def record_audio():
    global stop_recording
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎙️ Recording audio... Speak now.")
        recognizer.adjust_for_ambient_noise(source)
        stop_recording = False
        stop_thread = threading.Thread(target=stop_audio_recording)
        stop_thread.start()
        
        try:
            audio = recognizer.listen(source, timeout=30)  # Default time is 30s
            print("🔄 Processing speech-to-text...")
            return recognizer.recognize_google(audio)
        except:
            return "Could not understand speech."

def play_question(question):
    VOICE_ID = "pNInz6obpgDQGcFmaJgB"
    filename = "question.mp3"
    try:
        audio_stream = client.text_to_speech.convert(voice_id=VOICE_ID, text=question, model_id="eleven_multilingual_v1")
        with open(filename, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)
        if os.path.exists(filename):
            playsound(filename)
            time.sleep(5)
            os.remove(filename)
    except:
        print("Error generating audio.")

video_thread = threading.Thread(target=record_video)
video_thread.start()

num_questions = 5
for i in range(1, num_questions + 1):
    print(f"\n🟢 Question {i}:")
    question = generate_question()
    print(f"📢 {question}")
    play_question(question)
    response_text = record_audio()
    responses.append({"question": question, "answer": response_text})
    input("Press Enter to proceed to the next question...")

video_running = False
video_thread.join()
print("\n✅ Interview Complete!")
