# model/train_model.py
import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATASET_DIR = "dataset"  # expects dataset/A, dataset/B, ...
MODEL_OUT = "model/letter_model.h5"

mp_hands = mp.solutions.hands

def extract_landmarks_from_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.6) as hands:
        res = hands.process(img_rgb)
        if not res.multi_hand_landmarks:
            return None
        lm = res.multi_hand_landmarks[0]
        pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
        # normalize relative to wrist (lm 0)
        wrist = pts[0].copy()
        pts -= wrist
        # scale by bbox
        minxy = pts[:, :2].min(axis=0)
        maxxy = pts[:, :2].max(axis=0)
        box = (maxxy - minxy).max()
        if box == 0:
            box = 1.0
        pts[:, :2] /= box
        return pts.flatten()

def build_dataset(dataset_dir=DATASET_DIR):
    X = []
    y = []
    letters = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    if not letters:
        print("No dataset folders found in", dataset_dir)
        return None, None, None
    print("Found classes:", letters)
    for lbl in letters:
        folder = os.path.join(dataset_dir, lbl)
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(folder, fname)
            lm = extract_landmarks_from_image(path)
            if lm is None:
                print("Skipping (no hand):", path)
                continue
            X.append(lm)
            y.append(lbl.upper())
    if not X:
        print("No valid samples extracted.")
        return None, None, None
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc, num_classes=len(le.classes_))
    return X, y_cat, le

def make_model(input_dim, num_classes):
    model = Sequential([
        Dense(128, input_shape=(input_dim,), activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    X, y, le = build_dataset()
    if X is None:
        print("No data to train on. Collect samples first.")
        return
    input_dim = X.shape[1]
    num_classes = y.shape[1]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_model(input_dim, num_classes)
    print("Training model on", X_train.shape[0], "samples...")
    model.fit(X_train, y_train, epochs=40, batch_size=16, validation_data=(X_val, y_val))
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    print("Saved model to", MODEL_OUT)
    # Save label mapping
    import pickle
    with open("model/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print("Saved label encoder.")

if __name__ == "__main__":
    main()
