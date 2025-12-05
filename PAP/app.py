import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from tensorflow.keras.models import load_model
import pickle
import os
import time
from collections import deque

app = Flask(__name__)

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
    print("✓ Letter model loaded")

# Load word model
if os.path.exists(WORD_MODEL_PATH) and os.path.exists(WORD_LABEL_PATH):
    print("Loading word model...")
    word_model = load_model(WORD_MODEL_PATH)
    with open(WORD_LABEL_PATH, "rb") as f:
        word_label_encoder = pickle.load(f)
    print("✓ Word model loaded")

if letter_model is None and word_model is None:
    raise FileNotFoundError("No models found. Train a model first!")

# Default mode: use word model if available, otherwise letter model
DEFAULT_MODE = "words" if word_model is not None else "letters"


# ---------------------------
# Prediction memory
# ---------------------------
current_mode = DEFAULT_MODE  # "letters" or "words"
current_letter = "-"
current_word = "-"
predicted_word = ""  # Accumulated text
detected_words = []  # List of detected words (for word mode)
last_letter = "-"
letter_buffer = []
BUFFER_SIZE = 5  # number of frames to confirm a letter

# Word detection settings
SEQUENCE_LENGTH = 30
word_recording_buffer = deque(maxlen=SEQUENCE_LENGTH * 2)
last_word_prediction_time = 0
WORD_PREDICTION_COOLDOWN = 2.0  # Seconds between word predictions
MIN_RECORDING_FRAMES = 10

# ---------------------------
# Mediapipe
# ---------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Support 2 hands for words
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
    Generate frames from the webcam in a resilient way.
    Supports both letter and word detection modes.
    """
    global current_letter, current_word, predicted_word, detected_words
    global last_letter, letter_buffer, word_recording_buffer, last_word_prediction_time
    global current_mode

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

            # Word detection mode
            if current_mode == "words" and word_model is not None:
                landmarks = extract_landmarks_for_word(results)
                
                if landmarks is not None:
                    word_recording_buffer.append(landmarks)
                    
                    # Auto-detect when user performs a sign
                    if len(word_recording_buffer) >= SEQUENCE_LENGTH:
                        current_time = time.time()
                        if current_time - last_word_prediction_time > WORD_PREDICTION_COOLDOWN:
                            sequence = prepare_word_sequence(word_recording_buffer)
                            if sequence is not None:
                                probs = word_model.predict(sequence, verbose=0)[0]
                                idx = int(np.argmax(probs))
                                word = word_label_encoder.classes_[idx]
                                conf = float(probs[idx])
                                
                                if conf > 0.3:
                                    current_word = word
                                    if word not in detected_words[-5:]:
                                        detected_words.append(word)
                                        predicted_word += word + " "
                                        print(f"✓ Detected word: {word} (confidence: {conf:.2f})")
                                    last_word_prediction_time = current_time
                                    word_recording_buffer.clear()
                
                # Display with better positioning - no background
                status_text = f"Word: {current_word}"
                if len(word_recording_buffer) > 0:
                    status_text += f" (Recording: {len(word_recording_buffer)}/{SEQUENCE_LENGTH})"
                
                cv2.putText(frame, status_text, (15, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # Black color
                
                recent_words = " ".join(detected_words[-3:])
                if recent_words:
                    cv2.putText(frame, f"Recent: {recent_words}", 
                               (15, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Black color

            # Letter detection mode
            elif current_mode == "letters" and letter_model is not None:
                landmarks = extract_landmarks_for_letter(results)
                if landmarks is not None:
                    x = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                    probs = letter_model.predict(x, verbose=0)[0]
                    idx = int(np.argmax(probs))

                    letter = letter_label_encoder.classes_[idx]
                    conf = float(probs[idx])
                    current_letter = letter

                    # Debounce logic
                    if conf > 0.5:
                        letter_buffer.append(letter)
                        if len(letter_buffer) > BUFFER_SIZE:
                            letter_buffer.pop(0)

                        if len(letter_buffer) >= 3 and letter_buffer.count(letter) >= 3:
                            if letter != last_letter:
                                predicted_word += letter
                                last_letter = letter
                                print(f"✓ Letter confirmed: {letter} | Word so far: {predicted_word}")
                                letter_buffer = []
                    else:
                        if len(letter_buffer) > 0:
                            letter_buffer.pop(0)

                    # Position text in upper-left with padding to avoid cutoff
                    text_x = 15
                    text_y = 50
                    
                    # Display text in black, no background
                    text = f"{letter} ({conf:.2f})"
                    cv2.putText(
                        frame,
                        text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 0, 0),  # Black color
                        3
                    )

            # Mode indicator at bottom - no background
            mode_text = f"Mode: {current_mode.upper()}"
            if current_mode == "words" and word_model is None:
                mode_text += " (No model)"
            elif current_mode == "letters" and letter_model is None:
                mode_text += " (No model)"
            
            cv2.putText(frame, mode_text, (15, frame.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # Black color

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
        "current_letter": current_letter if current_mode == "letters" else "-",
        "current_word": current_word if current_mode == "words" else "-",
        "predicted_word": predicted_word,
        "detected_words": detected_words,
        "mode": current_mode,
        "has_letter_model": letter_model is not None,
        "has_word_model": word_model is not None
    })


@app.route('/reset_word', methods=['POST'])
def reset_word():
    global predicted_word, current_letter, current_word, last_letter
    global letter_buffer, detected_words, word_recording_buffer
    predicted_word = ""
    current_letter = "-"
    current_word = "-"
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
    app.run(debug=True)