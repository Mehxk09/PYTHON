import cv2
import os
import time
import sys
import mediapipe as mp

# Configuration
FPS = 30
MAX_RECORDING_TIME = 5

if len(sys.argv) < 2:
    print("Usage: python collect_words.py <WORD_OR_PHRASE>")
    print("Example: python collect_words.py 'bom dia'")
    print("Example: python collect_words.py obrigado")
    sys.exit(1)

# Handle multi-word phrases (e.g., "bom dia" -> "Bom_Dia")
WORD = sys.argv[1].replace(' ', '_').title()
SAVE_DIR = os.path.join("dataset", "psl_words", WORD)
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    sys.exit(1)

# Camera properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_writer = None
recording = False
frame_count = 0
sample_count = len([f for f in os.listdir(SAVE_DIR) if f.endswith('.mp4')])

print(f"\n{'='*50}")
print(f"Recording: {WORD}")
print(f"{'='*50}")
print("Instructions:")
print("  - Press SPACE to START/STOP recording")
print("  - Press 's' to stop + save recording")
print("  - Press 'q' to quit")
print(f"  - Current sample count: {sample_count}")
print(f"{'='*50}\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    frame = cv2.flip(frame, 1)

    # Detect hands (for drawing only)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw landmarks on the preview
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
    else:
        cv2.putText(frame, "NO HAND DETECTED", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Status text
    status_text = "RECORDING..." if recording else "Ready - Press SPACE to record"
    status_color = (0, 0, 255) if recording else (0, 255, 0)

    cv2.putText(frame, f"Word: {WORD}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, f"Sample: {sample_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if recording:
        recording_time = frame_count / FPS
        cv2.putText(frame, f"Time: {recording_time:.1f}s / {MAX_RECORDING_TIME}s", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.circle(frame, (width - 30, 30), 15, (0, 0, 255), -1)

    key = cv2.waitKey(1) & 0xFF

    # SPACE: toggle start/stop
    if key == ord(' '):
        if not recording:
            recording = True
            frame_count = 0

            timestamp = int(time.time())
            video_path = os.path.join(SAVE_DIR, f"{WORD}_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))

            print("Recording started...")
        else:
            recording = False
            if video_writer:
                video_writer.release()
                video_writer = None

            sample_count += 1
            print(f"Recording stopped. Sample count: {sample_count}")

    # s: stop + save (same as stopping)
    elif key == ord('s') and recording:
        recording = False
        if video_writer:
            video_writer.release()
            video_writer = None

        sample_count += 1
        print(f"Saved recording. Sample count: {sample_count}")

    elif key == ord('q'):
        break

    # Write frames to video while recording
    if recording and video_writer:
        video_writer.write(frame)
        frame_count += 1

        # Auto stop after max time
        if frame_count >= MAX_RECORDING_TIME * FPS:
            recording = False
            video_writer.release()
            video_writer = None
            sample_count += 1
            print(f"Auto-stopped after {MAX_RECORDING_TIME} seconds. Sample count: {sample_count}")

    cv2.imshow("Record Moving Hand Signs (MP4 only)", frame)

# Cleanup
if video_writer:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()
hands.close()
print(f"\nDone! Collected {sample_count} samples for {WORD}.")
