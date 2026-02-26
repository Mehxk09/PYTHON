import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from tensorflow.keras.models import load_model
import pickle
import os
import time
import threading
from collections import deque

app = Flask(__name__)

# Register dictionary blueprint (separate file)
from dicionario import dicionario_bp
app.register_blueprint(dicionario_bp)

# ---------------------------
# Load models (both letter and word models if available)
# ---------------------------
LETTER_MODEL_PATH = "model/letter_model.h5"
LETTER_LABEL_PATH = "model/label_encoder.pkl"
WORD_MODEL_PATH = "model/word_model.h5"
WORD_LABEL_PATH = "model/word_label_encoder.pkl"

letter_model = None
letter_label_encoder = None
word_model = None
word_label_encoder = None

# Load letter model
if os.path.exists(LETTER_MODEL_PATH) and os.path.exists(LETTER_LABEL_PATH):
    print("Loading letter model...")
    letter_model = load_model(LETTER_MODEL_PATH)
    with open(LETTER_LABEL_PATH, "rb") as f:
        letter_label_encoder = pickle.load(f)
    print("Letter model loaded")

# Load word model
if os.path.exists(WORD_MODEL_PATH) and os.path.exists(WORD_LABEL_PATH):
    print("Loading word model...")
    word_model = load_model(WORD_MODEL_PATH)
    with open(WORD_LABEL_PATH, "rb") as f:
        word_label_encoder = pickle.load(f)
    print("Word model loaded")

if letter_model is None and word_model is None:
    raise FileNotFoundError("No models found. Train a model first!")

# Default mode: always start with letters mode
DEFAULT_MODE = "letters"


# ---------------------------
# Prediction memory
# ---------------------------
current_mode = DEFAULT_MODE  # "letters" or "words"
current_letter = "-"
current_word = "-"
current_confidence = 0.0  # Store last real confidence
predicted_word = ""  # Accumulated text
detected_words = []  # List of detected words (for word mode)
last_letter = "-"
letter_buffer = []
BUFFER_SIZE = 5  # number of frames to confirm a letter

# Word detection settings
SEQUENCE_LENGTH = 20  # Reduced from 30 for faster detection
word_recording_buffer = deque(maxlen=SEQUENCE_LENGTH * 2)
last_word_prediction_time = 0
WORD_PREDICTION_COOLDOWN = 1.5  # Seconds between word predictions
MIN_RECORDING_FRAMES = 8  # Minimum frames needed before predicting

# Performance optimization
PREDICTION_THROTTLE = 0.12  # Predict every 0.12 seconds (~8 FPS for predictions)
last_prediction_time = 0

# Camera lock to prevent concurrent access
camera_lock = threading.Lock()
active_stream = False

# ---------------------------
# Mediapipe
# ---------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Support 2 hands for words
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------
# Extract landmarks
# ---------------------------
def extract_landmarks_for_letter(results):
    """Extract landmarks for letter prediction (single hand, normalized)"""
    if not results.multi_hand_landmarks:
        return None

    lm = results.multi_hand_landmarks[0]
    pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)

    wrist = pts[0].copy()
    pts -= wrist

    minxy = pts[:, :2].min(axis=0)
    maxxy = pts[:, :2].max(axis=0)
    box = (maxxy - minxy).max()
    if box == 0:
        box = 1.0

    pts[:, :2] /= box
    return pts.flatten()


def extract_landmarks_for_word(results):
    """Extract landmarks for word prediction (supports two hands).
    MUST match the preprocessing in train_word_model.py exactly (no normalization)."""
    if not results.multi_hand_landmarks:
        return None

    # Combine landmarks from all detected hands (raw coordinates)
    pts = []
    for hand_landmarks in results.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            pts.append([lm.x, lm.y, lm.z])

    if len(pts) == 0:
        return None

    pts = np.array(pts, dtype=np.float32)

    # Keep only first 21 landmarks (single hand) to match training
    if pts.shape[0] >= 21:
        pts = pts[:21]
    else:
        padded = np.zeros((21, 3), dtype=np.float32)
        padded[:pts.shape[0]] = pts
        pts = padded

    return pts.flatten()  # Returns (63,) — raw, no normalization


def prepare_word_sequence(buffer):
    """Prepare sequence from buffer for word prediction"""
    if len(buffer) < MIN_RECORDING_FRAMES:
        return None
    
    sequence = np.array(list(buffer), dtype=np.float32)
    
    # Pad or truncate to SEQUENCE_LENGTH
    if len(sequence) < SEQUENCE_LENGTH:
        last_frame = sequence[-1]
        while len(sequence) < SEQUENCE_LENGTH:
            sequence = np.vstack([sequence, last_frame])
    else:
        indices = np.linspace(0, len(sequence) - 1, SEQUENCE_LENGTH, dtype=int)
        sequence = sequence[indices]
    
    return sequence.reshape(1, SEQUENCE_LENGTH, -1)

# ---------------------------
# Webcam stream + prediction
# ---------------------------
def gen_frames():
    """
    Generate frames from the webcam in a resilient, smooth way.
    Supports both letter and word detection modes.
    """
    global current_letter, current_word, predicted_word, detected_words
    global last_letter, letter_buffer, word_recording_buffer, last_word_prediction_time
    global current_mode, last_prediction_time, active_stream, current_confidence

    # Prevent multiple streams from fighting over the camera
    with camera_lock:
        if active_stream:
            return
        active_stream = True

    cap = None
    consecutive_failures = 0
    MAX_FAILURES = 30  # After 30 failed reads, reopen camera
    frame_count = 0

    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow is more stable on Windows
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)  # Fallback

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce lag

        while True:
            # Reopen camera if too many failures
            if cap is None or not cap.isOpened() or consecutive_failures >= MAX_FAILURES:
                if cap is not None:
                    cap.release()
                    time.sleep(0.3)
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    consecutive_failures = 0
                else:
                    time.sleep(0.5)
                    continue

            success, frame = cap.read()
            if not success:
                consecutive_failures += 1
                time.sleep(0.01)
                continue

            consecutive_failures = 0  # Reset on successful read
            frame_count += 1
            frame = cv2.flip(frame, 1)

            # Only run MediaPipe every 2nd frame for performance
            run_mediapipe = (frame_count % 2 == 0)
            results = None

            if run_mediapipe:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

            # Draw landmarks (use cached results if skipping)
            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1)
                    )

            # ---- Word detection mode (processing) ----
            if current_mode == "words" and word_model is not None and results:
                landmarks = extract_landmarks_for_word(results)

                if landmarks is not None:
                    word_recording_buffer.append(landmarks)

                    # Predict when enough frames are collected
                    if len(word_recording_buffer) >= SEQUENCE_LENGTH:
                        now = time.time()
                        if now - last_word_prediction_time > WORD_PREDICTION_COOLDOWN:
                            sequence = prepare_word_sequence(word_recording_buffer)
                            if sequence is not None:
                                probs = word_model.predict(sequence, verbose=0)[0]
                                idx = int(np.argmax(probs))
                                word = word_label_encoder.classes_[idx]
                                conf = float(probs[idx])

                                if conf > 0.55:
                                    current_word = word
                                    current_confidence = conf
                                    if not detected_words or detected_words[-1] != word:
                                        detected_words.append(word)
                                        predicted_word += word + " "
                                        print(f"Detected word: {word} ({conf:.2f})")
                                    last_word_prediction_time = now
                                    word_recording_buffer.clear()

            # ---- Letter detection mode (processing) ----
            elif current_mode == "letters" and letter_model is not None and results:
                landmarks = extract_landmarks_for_letter(results)
                if landmarks is not None:
                    now = time.time()
                    if now - last_prediction_time >= PREDICTION_THROTTLE:
                        x = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                        probs = letter_model.predict(x, verbose=0)[0]
                        idx = int(np.argmax(probs))

                        letter = letter_label_encoder.classes_[idx]
                        conf = float(probs[idx])
                        current_letter = letter
                        current_confidence = conf  # Store real confidence
                        last_prediction_time = now

                    # Debounce logic (use current values)
                    letter = current_letter if current_letter != "-" else "-"
                    conf = current_confidence

                    if conf > 0.5 and letter != "-":
                        letter_buffer.append(letter)
                        if len(letter_buffer) > BUFFER_SIZE:
                            letter_buffer.pop(0)

                        if len(letter_buffer) >= 3 and letter_buffer.count(letter) >= 3:
                            if letter != last_letter:
                                predicted_word += letter
                                last_letter = letter
                                print(f"Letter confirmed: {letter} | Word: {predicted_word}")
                                letter_buffer = []
                    else:
                        if len(letter_buffer) > 0:
                            letter_buffer.pop(0)

            # ---- Always draw text overlay (every frame, no flicker) ----
            frame_w = frame.shape[1]

            if current_mode == "words" and word_model is not None:
                status_text = f"Word: {current_word}"
                buf_len = len(word_recording_buffer)
                if buf_len > 0:
                    status_text += f" ({buf_len}/{SEQUENCE_LENGTH})"

                if len(status_text) > 35:
                    status_text = status_text[:32] + "..."

                (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                tx = max(10, (frame_w - tw) // 2)
                cv2.putText(frame, status_text, (tx, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                recent = " ".join(detected_words[-3:])
                if recent:
                    if len(recent) > 25:
                        recent = recent[:22] + "..."
                    rtxt = f"Recent: {recent}"
                    (rw, _), _ = cv2.getTextSize(rtxt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    rx = max(10, (frame_w - rw) // 2)
                    cv2.putText(frame, rtxt, (rx, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            elif current_mode == "letters" and letter_model is not None:
                letter = current_letter if current_letter != "-" else "-"
                conf = current_confidence
                text = f"{letter} ({conf:.2f})" if letter != "-" else "--"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                tx = max(10, (frame_w - tw) // 2)
                cv2.putText(frame, text, (tx, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # Mode indicator at bottom center
            frame_w = frame.shape[1]
            mode_text = f"Mode: {current_mode.upper()}"
            (mw, _), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            mx = max(10, (frame_w - mw) // 2)
            cv2.putText(frame, mode_text, (mx, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # Encode JPEG — quality 80 for speed
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
                )

    finally:
        active_stream = False
        if cap is not None and cap.isOpened():
            cap.release()

# ---------------------------
# Flask Routes
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_prediction')
def get_prediction():
    return jsonify({
        "success": True,
        "current_letter": current_letter if current_mode == "letters" else "-",
        "current_word": current_word if current_mode == "words" else "-",
        "predicted_word": predicted_word,
        "detected_words": detected_words,
        "mode": current_mode,
        "confidence": round(current_confidence, 2),
        "has_letter_model": letter_model is not None,
        "has_word_model": word_model is not None
    })


@app.route('/reset_word', methods=['POST'])
def reset_word():
    global predicted_word, current_letter, current_word, last_letter
    global letter_buffer, detected_words, word_recording_buffer, current_confidence
    predicted_word = ""
    current_letter = "-"
    current_word = "-"
    current_confidence = 0.0
    last_letter = "-"
    letter_buffer = []
    detected_words = []
    word_recording_buffer.clear()
    return jsonify({"success": True})


@app.route('/set_mode', methods=['POST'])
def set_mode():
    global current_mode
    data = request.get_json()
    new_mode = data.get('mode', 'letters')
    
    if new_mode == "words" and word_model is None:
        return jsonify({"success": False, "error": "Word model not available"})
    elif new_mode == "letters" and letter_model is None:
        return jsonify({"success": False, "error": "Letter model not available"})
    
    current_mode = new_mode
    return jsonify({"success": True, "mode": current_mode})


@app.route('/get_mode')
def get_mode():
    return jsonify({
        "mode": current_mode,
        "has_letter_model": letter_model is not None,
        "has_word_model": word_model is not None
    })

@app.route('/add_space', methods=['POST'])
def add_space():
    global predicted_word
    predicted_word += " "
    return jsonify({"success": True})


@app.route('/backspace', methods=['POST'])
def backspace():
    """
    Remove the last character from the predicted word (if any).
    """
    global predicted_word
    if len(predicted_word) > 0:
        predicted_word = predicted_word[:-1]
    return jsonify({"success": True})
# ---------------------------
# Run
# ---------------------------
if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)