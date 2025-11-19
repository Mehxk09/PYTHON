# collect_samples.py
import cv2
import os
import sys
import time

if len(sys.argv) < 2:
    print("Usage: python collect_samples.py <LETTER>")
    sys.exit(1)

LETTER = sys.argv[1].upper()
OUT_DIR = os.path.join("dataset", LETTER)
os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
print("Press SPACE to save a sample image for letter", LETTER)
print("Press q to quit.")

count = len([f for f in os.listdir(OUT_DIR) if f.lower().endswith(('.png','.jpg'))])
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not available")
        break
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f"Letter: {LETTER}  Count: {count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Collect Samples", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # space to save
        fname = f"{LETTER}_{int(time.time())}.jpg"
        path = os.path.join(OUT_DIR, fname)
        cv2.imwrite(path, frame)
        count += 1
        print("Saved", path)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
