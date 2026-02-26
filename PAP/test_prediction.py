from model.letter_classifier import LetterClassifier
import numpy as np

# Initialize classifier with model and label encoder
clf = LetterClassifier("model/letter_model.h5", "model/label_encoder.pkl")

# Example fake input (63 landmarks = 21 points * 3 coordinates)
fake_input = np.random.rand(63)  # just for testing
letter, confidence = clf.predict(fake_input)

print(f"Predicted letter: {letter}, confidence: {confidence:.2f}")
