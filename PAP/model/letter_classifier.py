# model/letter_classifier.py
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model


class LetterClassifier:
    def __init__(self, model_path="model/letter_model.h5",
                 label_path="model/label_encoder.pkl"):
        self.model_path = model_path
        self.label_path = label_path
        self.model = None
        self.labels = None

        # Load model
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print("Loaded model:", self.model_path)
            except Exception as e:
                print("Failed to load model:", e)
                self.model = None
        else:
            print("Model file not found at", self.model_path)

        # Load label encoder
        if os.path.exists(self.label_path):
            try:
                with open(self.label_path, "rb") as f:
                    le = pickle.load(f)
                self.labels = list(le.classes_)
                print("Loaded labels:", self.labels)
            except Exception as e:
                print("Failed to load label encoder:", e)
                self.labels = None
        else:
            print("Label encoder not found at", self.label_path)

    def predict(self, landmarks_vector):
        """
        landmarks_vector: flat numpy array shape (63,)
        returns (letter, confidence) or (None, 0.0)
        """
        if self.model is None or self.labels is None:
            return (None, 0.0)
        x = np.array(landmarks_vector, dtype=np.float32).reshape(1, -1)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        if idx < len(self.labels):
            letter = self.labels[idx]
        else:
            letter = None
            conf = 0.0
        return (letter, conf)
