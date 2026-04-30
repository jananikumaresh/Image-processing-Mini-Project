zfrom flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import threading
import time

app = Flask(__name__)

# =========================
# MediaPipe Setup (FAST)
# =========================
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,   # 🔥 faster
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# Config
# =========================
EAR_THRESHOLD = 0.22
COUNTER_LIMIT = 5  # faster response

cap = cv2.VideoCapture(0)

# =========================
# Shared state (for /status)
# =========================
status_text = "Awake"
lock = threading.Lock()

# =========================
# EAR Calculation
# =========================
def calculate_ear(landmarks, eye_points):
    points = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_points])

    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])

    return (A + B) / (2.0 * C)

# =========================
# Processing thread (IMPORTANT SPEED BOOST)
# =========================
def process_frames():
    global status_text

    counter = 0
    frame_skip = 0

    while True:
        success, frame = cap.read()
        if not success:
            continue

        # 🔥 resize for speed
        frame = cv2.resize(frame, (640, 360))

        # 🔥 skip frames to reduce load
        frame_skip += 1
        if frame_skip % 2 != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        current_status = "Awake"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                left_eye = [33, 160, 158, 133, 153, 144]
                right_eye = [362, 385, 387, 263, 373, 380]

                leftEAR = calculate_ear(face_landmarks.landmark, left_eye)
                rightEAR = calculate_ear(face_landmarks.landmark, right_eye)

                ear = (leftEAR + rightEAR) / 2.0

                if ear < EAR_THRESHOLD:
                    counter += 1
                    if counter > COUNTER_LIMIT:
                        current_status = "DROWSY!"
                else:
                    counter = 0

        with lock:
            status_text = current_status

        time.sleep(0.01)

# Start processing thread
threading.Thread(target=process_frames, daemon=True).start()

# =========================
# Video stream
# =========================
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 360))

        with lock:
            label = status_text

        cv2.putText(
            frame,
            label,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )

# =========================
# Routes
# =========================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/status')
def status():
    with lock:
        return jsonify({"status": status_text})


# =========================
# Run server (IMPORTANT)
# =========================
if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False
    )