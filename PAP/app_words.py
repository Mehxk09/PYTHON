import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify
from tensorflow.keras.models import load_model
import pickle
import os
import time
from collections import deque

app = Flask(__name__)

# ---------------------------
# Configuration
# ---------------------------
WORD_MODEL_PATH = "model/word_model.h5"
WORD_LABEL_PATH = "model/word_label_encoder.pkl"
LETTER_MODEL_PATH = "model/letter_model.h5"
LETTER_LABEL_PATH = "model/label_encoder.pkl"

SEQUENCE_LENGTH = 30  # Must match training
RECORDING_FPS = 30
MIN_RECORDING_FRAMES = 10  # Minimum frames to make a prediction

# ---------------------------
# Load models
# ---------------------------
word_model = None
word_label_encoder = None
letter_model = None
letter_label_encoder = None
mode = "letters"  # "letters" or "words"

# Try to load word model first
if os.path.exists(WORD_MODEL_PATH) and os.path.exists(WORD_LABEL_PATH):
    print("Loading word model...")
    word_model = load_model(WORD_MODEL_PATH)
    with open(WORD_LABEL_PATH, "rb") as f:
        word_label_encoder = pickle.load(f)
    mode = "words"
    print(f"✓ Word model loaded. Mode: WORDS")
    print(f"  Classes: {list(word_label_encoder.classes_)}")
elif os.path.exists(LETTER_MODEL_PATH) and os.path.exists(LETTER_LABEL_PATH):
    print("Loading letter model...")
    letter_model = load_model(LETTER_MODEL_PATH)
    with open(LETTER_LABEL_PATH, "rb") as f:
        letter_label_encoder = pickle.load(f)
    mode = "letters"
    print(f"✓ Letter model loaded. Mode: LETTERS")
else:
    raise FileNotFoundError("No model found. Train a model first!")

# ---------------------------
# Prediction memory
# ---------------------------
current_prediction = "-"
detected_words = []  # List of detected words
is_recording = False
recording_buffer = deque(maxlen=SEQUENCE_LENGTH * 2)  # Store more frames than needed
last_prediction_time = 0
PREDICTION_COOLDOWN = 2.0  # Seconds between predictions

# ---------------------------
# Mediapipe
# ---------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Support two hands for words
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
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
    """Extract landmarks for word prediction (supports two hands)"""
    if not results.multi_hand_landmarks:
        return None
    
    # Combine landmarks from all hands
    all_landmarks = []
    for hand_landmarks in results.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            all_landmarks.append([lm.x, lm.y, lm.z])
    
    if len(all_landmarks) == 0:
        return None
    
    # Normalize
    landmarks = np.array(all_landmarks, dtype=np.float32)
    if len(landmarks) > 0:
        wrist = landmarks[0].copy()
        landmarks = landmarks - wrist
        
        # Scale normalization
        min_vals = landmarks.min(axis=0)
        max_vals = landmarks.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0
        landmarks = landmarks / ranges
    
    # Pad or truncate to 21 landmarks (single hand standard)
    if len(landmarks) < 21:
        padded = np.zeros((21, 3))
        padded[:len(landmarks)] = landmarks[:len(landmarks)]
        landmarks = padded
    elif len(landmarks) > 21:
        landmarks = landmarks[:21]
    
    return landmarks.flatten()  # Returns (63,)


def prepare_sequence(buffer):
    """Prepare sequence from buffer for word prediction"""
    if len(buffer) < MIN_RECORDING_FRAMES:
        return None
    
    # Convert buffer to array
    sequence = np.array(list(buffer), dtype=np.float32)
    
    # Pad or truncate to SEQUENCE_LENGTH
    if len(sequence) < SEQUENCE_LENGTH:
        # Pad with last frame
        last_frame = sequence[-1]
        while len(sequence) < SEQUENCE_LENGTH:
            sequence = np.vstack([sequence, last_frame])
    else:
        # Sample evenly or take last SEQUENCE_LENGTH frames
        indices = np.linspace(0, len(sequence) - 1, SEQUENCE_LENGTH, dtype=int)
        sequence = sequence[indices]
    
    # Reshape for model: (1, sequence_length, features_per_frame)
    return sequence.reshape(1, SEQUENCE_LENGTH, -1)


# ---------------------------
# Webcam stream + prediction
# ---------------------------
def gen_frames():
    global current_prediction, detected_words, is_recording, recording_buffer, last_prediction_time
    
    cap = None
    
    def _open_capture():
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("Could not open webcam. Retrying...")
            return None
        return capture
    
    try:
        while True:
            if cap is None or not cap.isOpened():
                cap = _open_capture()
                if cap is None:
                    time.sleep(0.5)
                    continue
            
            success, frame = cap.read()
            if not success:
                cap.release()
                cap = None
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
            
            # Extract landmarks and make predictions
            if mode == "words" and word_model is not None:
                # Word detection mode
                landmarks = extract_landmarks_for_word(results)
                
                if landmarks is not None:
                    # Add to recording buffer
                    recording_buffer.append(landmarks)
                    
                    # Auto-detect when user performs a sign (buffer fills up)
                    if len(recording_buffer) >= SEQUENCE_LENGTH:
                        current_time = time.time()
                        # Only predict if cooldown has passed
                        if current_time - last_prediction_time > PREDICTION_COOLDOWN:
                            sequence = prepare_sequence(recording_buffer)
                            if sequence is not None:
                                probs = word_model.predict(sequence, verbose=0)[0]
                                idx = int(np.argmax(probs))
                                word = word_label_encoder.classes_[idx]
                                conf = float(probs[idx])
                                
                                if conf > 0.3:  # Lower threshold for words
                                    current_prediction = word
                                    if word not in detected_words[-5:]:  # Avoid duplicates
                                        detected_words.append(word)
                                        print(f"✓ Detected word: {word} (confidence: {conf:.2f})")
                                    last_prediction_time = current_time
                                    
                                    # Clear buffer after prediction
                                    recording_buffer.clear()
                
                # Display current prediction
                status_text = f"Word: {current_prediction}"
                if len(recording_buffer) > 0:
                    status_text += f" (Recording: {len(recording_buffer)}/{SEQUENCE_LENGTH})"
                
                cv2.putText(frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Show recent words
                recent_words = " ".join(detected_words[-3:])
                if recent_words:
                    cv2.putText(frame, f"Recent: {recent_words}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            elif mode == "letters" and letter_model is not None:
                # Letter detection mode (original functionality)
                landmarks = extract_landmarks_for_letter(results)
                if landmarks is not None:
                    x = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                    probs = letter_model.predict(x, verbose=0)[0]
                    idx = int(np.argmax(probs))
                    letter = letter_label_encoder.classes_[idx]
                    conf = float(probs[idx])
                    current_prediction = letter
                    
                    cv2.putText(frame, f"{letter} ({conf:.2f})", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "No model loaded", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Mode indicator
            cv2.putText(frame, f"Mode: {mode.upper()}", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )
    finally:
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
        "current_prediction": current_prediction,
        "detected_words": detected_words,
        "mode": mode
    })


@app.route('/reset_words', methods=['POST'])
def reset_words():
    global detected_words, current_prediction, recording_buffer
    detected_words = []
    current_prediction = "-"
    recording_buffer.clear()
    return jsonify({"success": True})


@app.route('/get_mode')
def get_mode():
    return jsonify({"mode": mode, "success": True})


# ---------------------------
# Run
# ---------------------------
if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"Starting app in {mode.upper()} mode")
    print(f"{'='*60}\n")
    app.run(debug=True, port=5000)

