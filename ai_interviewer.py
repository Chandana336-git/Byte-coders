import os
import cv2
import time
import speech_recognition as sr
import google.generativeai as genai
import mediapipe as mp
import threading
import numpy as np
from playsound import playsound
from elevenlabs import ElevenLabs

# ✅ Set API Keys (Store these securely in environment variables)
GENAI_API_KEY = "Give your api key"  # Replace with your actual API key
ELEVENLABS_API_KEY = "give yor api key"  # Replace with your actual API key

# ✅ Configure APIs
genai.configure(api_key=GENAI_API_KEY)
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# ✅ Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def analyze_facial_expressions(video_filename="response.mp4"):
    """Analyzes facial expressions from the recorded video using MediaPipe."""
    cap = cv2.VideoCapture(video_filename)
    frame_count = 0
    face_presence = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            face_presence += 1

        frame_count += 1

    cap.release()
    
    if frame_count == 0:
        return "No video data available for analysis."
    
    face_presence_percentage = (face_presence / frame_count) * 100

    if face_presence_percentage > 80:
        return "Candidate maintained good eye contact and engagement."
    elif face_presence_percentage > 50:
        return "Candidate showed some engagement but was distracted at times."
    else:
        return "Candidate was mostly disengaged or looking away."

def generate_question(topic):
    """Generate a single interview question using Google Gemini AI."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"Generate one interview question for a {topic} role."

    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "No question generated."
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while generating the question."

def evaluate_response(question, answer, facial_analysis):
    """Evaluate the candidate's response and expressions using Gemini AI."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
    Interview question: {question}
    Candidate's response: {answer}
    
    Facial Expression & Behavior Analysis: {facial_analysis}
    
    Provide an overall evaluation, including strengths, weaknesses, and a final rating out of 10 considering both the response and expressions.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "No evaluation generated."
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while evaluating the response."

def record_audio():
    """Records candidate's voice response and converts it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\nPlease speak your response now...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=10)
            print("Processing response...")
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand your response."
        except sr.RequestError:
            return "Error connecting to the speech recognition service."

def record_video(filename="response.mp4", duration=10):
    """Records video while the candidate answers."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))

    print("\nRecording video... Please answer the question.")
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        cv2.imshow("Recording...", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def play_question(question):
    """Uses ElevenLabs to convert the question to speech and play it."""
    VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Replace with a valid ElevenLabs voice ID
    filename = "question.mp3"

    try:
        audio_stream = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=question,
            model_id="eleven_multilingual_v1"
        )

        with open(filename, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)

        if os.path.exists(filename):
            playsound(filename)
            os.remove(filename)
        else:
            print(f"Error: {filename} not found.")
    except Exception as e:
        print(f"Error generating audio: {e}")

# ✅ Interview Process
topic = "Software Engineer"
question = generate_question(topic)

if question:
    print(f"Interview Question: {question}")

    # Play the question using ElevenLabs
    play_question(question)

    # Start video & audio recording simultaneously
    video_filename = "response.mp4"
    
    video_thread = threading.Thread(target=record_video, args=(video_filename,))
    video_thread.start()

    audio_response = record_audio()  # Audio recording runs synchronously
    video_thread.join()  # Wait for video to finish

    # Analyze facial expressions from video
    facial_analysis = analyze_facial_expressions(video_filename)

    # Evaluate response with Gemini AI
    evaluation = evaluate_response(question, audio_response, facial_analysis)
    print("\nEvaluation of your response:")
    print(evaluation)

else:
    print("No question was generated.")
