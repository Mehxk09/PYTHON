# model/train_word_model.py
import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Reshape
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import glob
import pickle

DATASET_DIR = os.path.join("dataset", "psl_words")
MODEL_OUT = "model/word_model.h5"
LABEL_OUT = "model/word_label_encoder.pkl"
SEQUENCE_LENGTH = 30  # Number of frames to use per word (will pad/truncate)

mp_hands = mp.solutions.hands

def extract_landmarks_from_frame(frame, hands):
    """Extract landmarks from a single frame"""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    
    if not result.multi_hand_landmarks:
        return None
    
    # Combine landmarks from all hands (up to 2 hands)
    all_landmarks = []
    for hand_landmarks in result.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            all_landmarks.append([lm.x, lm.y, lm.z])
    
    # Normalize: use first hand's wrist as reference
    if len(all_landmarks) > 0:
        wrist = np.array(all_landmarks[0])
        landmarks = np.array(all_landmarks)
        landmarks = landmarks - wrist  # Center around wrist
        
        # Scale normalization
        if len(landmarks) > 0:
            min_vals = landmarks.min(axis=0)
            max_vals = landmarks.max(axis=0)
            ranges = max_vals - min_vals
            ranges[ranges == 0] = 1.0  # Avoid division by zero
            landmarks = landmarks / ranges
    
    # Pad or truncate to fixed size (21 landmarks per hand, max 2 hands = 42)
    # For simplicity, use first 21 landmarks (single hand) or pad
    if len(all_landmarks) < 21:
        # Pad with zeros
        padded = np.zeros((21, 3))
        padded[:len(all_landmarks)] = landmarks[:len(all_landmarks)]
        landmarks = padded
    elif len(all_landmarks) > 21:
        # Take first 21 (or could average/combine)
        landmarks = landmarks[:21]
    
    return landmarks.flatten()  # Returns shape (63,)


def extract_landmarks_from_video(video_path):
    """Extract landmark sequences from a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    landmarks_sequence = []
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks = extract_landmarks_from_frame(frame, hands)
        if landmarks is not None:
            landmarks_sequence.append(landmarks)
    
    cap.release()
    hands.close()
    
    if len(landmarks_sequence) == 0:
        return None
    
    # Pad or truncate to SEQUENCE_LENGTH
    if len(landmarks_sequence) < SEQUENCE_LENGTH:
        # Pad with last frame
        last_frame = landmarks_sequence[-1]
        while len(landmarks_sequence) < SEQUENCE_LENGTH:
            landmarks_sequence.append(last_frame)
    else:
        # Truncate or sample evenly
        indices = np.linspace(0, len(landmarks_sequence) - 1, SEQUENCE_LENGTH, dtype=int)
        landmarks_sequence = [landmarks_sequence[i] for i in indices]
    
    return np.array(landmarks_sequence, dtype=np.float32)


def extract_landmarks_from_npy_folder(folder_path):
    """Extract landmarks from .npy files in a folder"""
    npy_files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
    if not npy_files:
        return None
    
    landmarks_sequence = []
    for npy_file in npy_files:
        landmarks = np.load(npy_file)
        # Flatten if needed
        if landmarks.ndim > 1:
            landmarks = landmarks.flatten()
        landmarks_sequence.append(landmarks)
    
    if len(landmarks_sequence) == 0:
        return None
    
    # Pad or truncate to SEQUENCE_LENGTH
    if len(landmarks_sequence) < SEQUENCE_LENGTH:
        last_frame = landmarks_sequence[-1]
        while len(landmarks_sequence) < SEQUENCE_LENGTH:
            landmarks_sequence.append(last_frame)
    else:
        indices = np.linspace(0, len(landmarks_sequence) - 1, SEQUENCE_LENGTH, dtype=int)
        landmarks_sequence = [landmarks_sequence[i] for i in indices]
    
    return np.array(landmarks_sequence, dtype=np.float32)


def build_dataset(dataset_dir=DATASET_DIR):
    """Build dataset from video files and npy folders"""
    X = []
    y = []
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return None, None, None
    
    word_folders = sorted([d for d in os.listdir(dataset_dir) 
                          if os.path.isdir(os.path.join(dataset_dir, d))])
    
    if not word_folders:
        print("No word folders found in", dataset_dir)
        return None, None, None
    
    print(f"Found {len(word_folders)} word classes: {word_folders}")
    
    for word_folder in word_folders:
        word_path = os.path.join(dataset_dir, word_folder)
        word_label = word_folder.replace('_', ' ').title()  # "Bom_Dia" -> "Bom Dia"
        
        # Process video files
        video_files = glob.glob(os.path.join(word_path, "*.mp4"))
        for video_file in video_files:
            print(f"Processing video: {video_file}")
            sequence = extract_landmarks_from_video(video_file)
            if sequence is not None:
                X.append(sequence)
                y.append(word_label)
        
        # Process npy landmark folders
        npy_folders = [d for d in os.listdir(word_path) 
                      if os.path.isdir(os.path.join(word_path, d)) and "landmarks" in d.lower()]
        for npy_folder in npy_folders:
            folder_path = os.path.join(word_path, npy_folder)
            print(f"Processing landmarks folder: {folder_path}")
            sequence = extract_landmarks_from_npy_folder(folder_path)
            if sequence is not None:
                X.append(sequence)
                y.append(word_label)
    
    if not X:
        print("No valid samples extracted.")
        return None, None, None
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    print(f"\nDataset summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Sequence shape: {X.shape}")
    print(f"  Classes: {sorted(set(y))}")
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc, num_classes=len(le.classes_))
    
    return X, y_cat, le


def make_model(input_shape, num_classes):
    """Create LSTM model for sequence classification"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def make_simple_model(input_shape, num_classes):
    """Create simpler model that flattens sequence (alternative approach)"""
    model = Sequential([
        Reshape((input_shape[0] * input_shape[1],), input_shape=input_shape),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    print("=" * 60)
    print("Training Word Recognition Model")
    print("=" * 60)
    
    X, y, le = build_dataset()
    if X is None:
        print("\nNo data to train on. Collect word samples first using:")
        print("  python collect_words.py <WORD>")
        return
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    input_shape = (X.shape[1], X.shape[2])  # (sequence_length, features_per_frame)
    num_classes = y.shape[1]
    
    print(f"\nModel configuration:")
    print(f"  Input shape: {input_shape}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Validation samples: {X_val.shape[0]}")
    
    # Choose model type based on dataset size
    if len(X_train) < 50:
        print("\nUsing simple model (small dataset)...")
        model = make_simple_model(input_shape, num_classes)
    else:
        print("\nUsing LSTM model (larger dataset)...")
        model = make_model(input_shape, num_classes)
    
    model.summary()
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=8,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    print(f"\n✓ Saved model to {MODEL_OUT}")
    
    # Save label encoder
    with open(LABEL_OUT, "wb") as f:
        pickle.dump(le, f)
    print(f"✓ Saved label encoder to {LABEL_OUT}")
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Accuracy: {val_acc:.2%}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

