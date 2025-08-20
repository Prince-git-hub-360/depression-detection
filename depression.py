import os
import sys
import time
import cv2
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
from transformers import pipeline
from deepface import DeepFace
from datetime import datetime
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import words
import csv

# Download NLTK dictionary
nltk.download('words')
english_words = set(words.words())

# Function to check if input text is meaningful
def is_meaningful(text):
    words_in_text = text.lower().split()
    if not words_in_text:
        return False
    matches = sum(1 for word in words_in_text if word in english_words)
    return matches / len(words_in_text) > 0.4

# ---------- 0. USER NAME ----------
name = input("👤 Please enter your name: ").strip().title()
print(f"\n📝 Hi {name}, how are you feeling today?")

# ---------- 1. TEXT INPUT ----------
text_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
user_text = input()

if not is_meaningful(user_text):
    print("❌ Please enter a valid sentence with meaningful English words.")
    exit()

text_result = text_classifier(user_text)[0]
sentiment_label = text_result["label"]
sentiment_score = text_result["score"]
print(f"✅ Text Sentiment: {sentiment_label} ({sentiment_score:.2f})")

# ---------- 2. VOICE ----------
print("\n🎙️ Recording voice for 10 seconds...")
fs = 44100
duration = 10
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
write("user_voice.wav", fs, audio)
print("✅ Voice recorded as 'user_voice.wav'")

y, sr = librosa.load("user_voice.wav")
pitch = librosa.yin(y, fmin=50, fmax=300).mean()
energy = np.mean(librosa.feature.rms(y=y))
print(f"🎧 Voice Features → Pitch: {pitch:.2f}, Energy: {energy:.5f}")

# ---------- 3. FACE ----------
print("\n📸 Capturing your image... Look at the camera.")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot access webcam")
    exit()

ret, frame = cap.read()
if not ret:
    print("❌ Failed to capture image.")
    exit()

cv2.imwrite("user_face.jpg", frame)
cap.release()
cv2.destroyAllWindows()
print("✅ Image saved as 'user_face.jpg'")

try:
    face_result = DeepFace.analyze(img_path="user_face.jpg", actions=["emotion"], enforce_detection=True)
    dominant_emotion = face_result[0]["dominant_emotion"]
    print(f"😐 Facial Emotion: {dominant_emotion}")
except:
    print("❌ No face detected! Please ensure your face is clearly visible and try again.")
    exit()

# ---------- 4. FUSION & SCORE ----------
text_class = 1 if sentiment_label == "NEGATIVE" else 0
emotion_class = 1 if dominant_emotion in ["sad", "fear", "angry", "disgust"] else 0
low_pitch = 1 if pitch < 135 else 0
low_energy = 1 if energy < 0.02 else 0

features = [pitch, energy, sentiment_score, text_class, emotion_class]
print(f"\n📈 Features → Pitch: {pitch:.2f}, Energy: {energy:.5f}, Text: {sentiment_label}, Face: {dominant_emotion}")

score = text_class + emotion_class + low_pitch + low_energy

# ---------- Adjusted Threshold Logic ----------
if score == 0:
    final_status = "Not Depressed"
elif score == 1:
    final_status = "Likely Not Depressed"
elif score == 2:
    final_status = "Borderline"
elif score == 3:
    final_status = "Likely Depressed"
else:
    final_status = "Depressed"

print(f"\n📊 Depression Score: {score}/4")
print(f"🔮 Final Prediction: {final_status}")

# ---------- 5. SAVE TO CSV ----------
row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, pitch, energy, sentiment_label, dominant_emotion, score, final_status]
file_exists = os.path.isfile("depression_logs.csv")

with open("depression_logs.csv", mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["Timestamp", "Name", "Pitch", "Energy", "Text Sentiment", "Facial Emotion", "Score", "Result"])
    writer.writerow(row)

# ---------- 6. STREAMLIT GUI ----------
df = pd.read_csv("depression_logs.csv")
st.title("🧠 Depression Detection Log")
st.dataframe(df.tail(10))

st.subheader("📉 Depression Score Over Time")
if "Score" in df.columns:
    st.line_chart(df["Score"])
else:
    st.warning("⚠️ No 'Score' column found in CSV.")
