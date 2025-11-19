from model.letter_classifier import LetterClassifier
import numpy as np

# Initialize classifier
clf = LetterClassifier("model/letter_model.h5")

# Example fake input (63 landmarks = 21 points * 3 coordinates)
fake_input = np.random.rand(63)  # just for testing
letter, confidence = clf.predict(fake_input)

print(f"Predicted letter: {letter}, confidence: {confidence:.2f}")
