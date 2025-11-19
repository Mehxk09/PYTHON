# model/letter_classifier.py
import numpy as np
import string
from tensorflow.keras.models import load_model
import os

class LetterClassifier:
    def __init__(self, model_path="model/letter_model.h5"):
        self.model_path = model_path
        self.model = None
        self.labels = list(string.ascii_uppercase)  # A..Z
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print("Loaded model:", self.model_path)
            except Exception as e:
                print("Failed to load model:", e)
                self.model = None
        else:
            print("Model file not found at", self.model_path)

    def predict(self, landmarks_vector):
        """
        landmarks_vector: flat numpy array shape (63,)
        returns (letter, confidence) or (None, 0.0)
        """
        if self.model is None:
            return (None, 0.0)
        x = np.array(landmarks_vector, dtype=np.float32).reshape(1, -1)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        letter = self.labels[idx]
        return (letter, conf)
