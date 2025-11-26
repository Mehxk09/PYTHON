import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify
from tensorflow.keras.models import load_model
import pickle
import os

app = Flask(__name__)

# ---------------------------
# Load model + label encoder
# ---------------------------
MODEL_PATH = "model/letter_model.h5"
LABEL_PATH = "model/label_encoder.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_PATH):
    print("Loading model and label encoder...")
    model = load_model(MODEL_PATH)
    with open(LABEL_PATH, "rb") as f:
        label_encoder = pickle.load(f)
else:
    raise FileNotFoundError("Model or label encoder not found.")

# ---------------------------
# Prediction memory
# ---------------------------
current_letter = "-"
predicted_word = ""
last_letter = "-"
letter_buffer = []
BUFFER_SIZE = 5  # number of frames to confirm a letter

# ---------------------------
# Mediapipe
# ---------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ---------------------------
# Extract landmarks
# ---------------------------
def extract_landmarks(results):
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

# ---------------------------
# Webcam stream + prediction
# ---------------------------
def gen_frames():
    global current_letter, predicted_word, last_letter, letter_buffer

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = extract_landmarks(results)
            if landmarks is not None:
                x = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                probs = model.predict(x, verbose=0)[0]
                idx = int(np.argmax(probs))

                letter = label_encoder.classes_[idx]
                conf = float(probs[idx])
                current_letter = letter

                # -----------------------
                # Debounce logic
                # -----------------------
                if conf > 0.5:  # Lowered from 0.90 to catch more predictions
                    letter_buffer.append(letter)
                    if len(letter_buffer) > BUFFER_SIZE:
                        letter_buffer.pop(0)

                    # Only add if at least 3 out of 5 frames are the same letter (was BUFFER_SIZE=5, too strict!)
                    if len(letter_buffer) >= 3 and letter_buffer.count(letter) >= 3:
                        if letter != last_letter:
                            predicted_word += letter
                            last_letter = letter
                            print(f"✓ Letter confirmed: {letter} | Word so far: {predicted_word}")
                            letter_buffer = []  
                else:
                    if len(letter_buffer) > 0:
                        letter_buffer.pop(0)  

                cv2.putText(
                    frame,
                    f"{letter} ({conf:.2f})",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    3
                )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

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
        "current_letter": current_letter,
        "predicted_word": predicted_word
    })


@app.route('/reset_word', methods=['POST'])
def reset_word():
    global predicted_word, current_letter, last_letter, letter_buffer
    predicted_word = ""
    current_letter = "-"
    last_letter = "-"
    letter_buffer = []
    return jsonify({"success": True})

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