import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import glob
import pickle

DATASET_DIR = os.path.join("dataset", "psl_words")
MODEL_OUT = "model/word_model.h5"
LABEL_OUT = "model/word_label_encoder.pkl"
SEQUENCE_LENGTH = 20  # Must match app.py

mp_hands = mp.solutions.hands


def extract_landmarks_from_frame(frame, hands):
    """Extract raw landmarks from a single frame (no normalization)."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    pts = []
    for hand_landmarks in results.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            pts.append([lm.x, lm.y, lm.z])

    pts = np.array(pts, dtype=np.float32)

    # Keep only first 21 landmarks (one hand)
    if pts.shape[0] >= 21:
        pts = pts[:21]
    else:
        padded = np.zeros((21, 3), dtype=np.float32)
        padded[:pts.shape[0]] = pts
        pts = padded

    return pts.flatten()  # (63,)


def extract_landmarks_from_video(video_path):
    """Extract a sequence of landmark vectors from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    sequence = []
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm = extract_landmarks_from_frame(frame, hands)
        if lm is not None:
            sequence.append(lm)

    cap.release()
    hands.close()

    if len(sequence) == 0:
        return None

    # Pad or uniformly sample to SEQUENCE_LENGTH
    if len(sequence) < SEQUENCE_LENGTH:
        last = sequence[-1]
        while len(sequence) < SEQUENCE_LENGTH:
            sequence.append(last)
    else:
        idxs = np.linspace(0, len(sequence) - 1, SEQUENCE_LENGTH, dtype=int)
        sequence = [sequence[i] for i in idxs]

    return np.array(sequence, dtype=np.float32)


def build_dataset():
    """Build training dataset from all word video folders."""
    X, y = [], []

    if not os.path.exists(DATASET_DIR):
        print(f"Dataset directory not found: {DATASET_DIR}")
        return None, None, None

    word_folders = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )

    # Skip empty folders
    word_folders = [w for w in word_folders
                    if glob.glob(os.path.join(DATASET_DIR, w, "*.mp4"))]

    print(f"Found {len(word_folders)} word classes: {word_folders}")

    for word in word_folders:
        word_path = os.path.join(DATASET_DIR, word)
        label = word.replace("_", " ").title()

        videos = glob.glob(os.path.join(word_path, "*.mp4"))
        print(f"  {word}: {len(videos)} videos")
        for video in videos:
            seq = extract_landmarks_from_video(video)
            if seq is not None:
                X.append(seq)
                y.append(label)

    if not X:
        return None, None, None

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)

    print(f"\nDataset: {len(X)} samples, {len(le.classes_)} classes")
    print(f"Classes: {list(le.classes_)}")
    return X, y_cat, le


def make_model(input_shape, num_classes):
    """Build LSTM model for sequence classification."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    print("=" * 60)
    print("Training Word Recognition Model (LSTM)")
    print(f"Sequence Length: {SEQUENCE_LENGTH}")
    print("=" * 60)

    X, y, le = build_dataset()
    if X is None:
        print("No data found. Collect word samples first.")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_shape = (X.shape[1], X.shape[2])
    num_classes = y.shape[1]

    model = make_model(input_shape, num_classes)
    model.summary()

    # Callbacks for better training
    os.makedirs("model", exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint(MODEL_OUT, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    model.fit(
        X_train, y_train,
        epochs=80,
        batch_size=8,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    # Save final model (best weights already saved via checkpoint)
    model.save(MODEL_OUT)

    with open(LABEL_OUT, "wb") as f:
        pickle.dump(le, f)

    print(f"\nModel saved to {MODEL_OUT}")
    print(f"Label encoder saved to {LABEL_OUT}")
    print(f"Classes: {list(le.classes_)}")


if __name__ == "__main__":
    main()
