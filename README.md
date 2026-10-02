#  Sign Language Recognition System

A real-time AI-powered Sign Language Recognition System that uses **MediaPipe** for hand landmark detection and **TensorFlow/Keras** for gesture classification.

The system captures hand gestures through a webcam, extracts 21 hand landmarks, processes the landmark coordinates, and predicts the corresponding gesture in real time.

---

##  Project Overview

Communication can be difficult between people who use sign language and those who do not understand it.

This project aims to provide a simple AI-based solution that can recognize predefined hand gestures and convert them into readable text. The system can also provide speech output for the recognized gestures.

### Main Pipeline

```text
Webcam
   ↓
MediaPipe Hand Detection
   ↓
21 Hand Landmarks
   ↓
Feature Normalization
   ↓
TensorFlow/Keras Model
   ↓
Gesture Prediction
   ↓
Text Output
   ↓
Optional Text-to-Speech
