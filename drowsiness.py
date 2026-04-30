# drowsiness_mediapipe.py

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

EAR_THRESHOLD = 0.2
COUNTER = 0

def calculate_ear(landmarks, eye_points):
    points = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_points])
    
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    
    return (A + B) / (2.0 * C)

def detect_drowsiness():
    global COUNTER
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    if not ret:
        return False
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # Left & Right eye points (MediaPipe indexes)
            left_eye = [33, 160, 158, 133, 153, 144]
            right_eye = [362, 385, 387, 263, 373, 380]

            leftEAR = calculate_ear(face_landmarks.landmark, left_eye)
            rightEAR = calculate_ear(face_landmarks.landmark, right_eye)

            ear = (leftEAR + rightEAR) / 2

            if ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER > 10:
                    return True
            else:
                COUNTER = 0

    return False