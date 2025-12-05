import cv2
import os
import time
import sys
import numpy as np
import mediapipe as mp

# Configuration
FPS = 30  # Frames per second for recording
MAX_RECORDING_TIME = 5  # Maximum recording time in seconds

if len(sys.argv) < 2:
    print("Usage: python collect_words.py <WORD_OR_PHRASE>")
    print("Example: python collect_words.py 'bom dia'")
    print("Example: python collect_words.py obrigado")
    sys.exit(1)

# Handle multi-word phrases (e.g., "bom dia" -> "Bom_Dia")
WORD = sys.argv[1].replace(' ', '_').title()  # Convert "bom dia" to "Bom_Dia"
SAVE_DIR = os.path.join("dataset", "psl_words", WORD)

# Create folder if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,  # Allow two hands for some signs
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    sys.exit(1)

# Get camera properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer for saving video files
video_writer = None
recording = False
frame_count = 0
sample_count = len([f for f in os.listdir(SAVE_DIR) if f.endswith('.mp4')])

print(f"\n{'='*50}")
print(f"Recording: {WORD}")
print(f"{'='*50}")
print("Instructions:")
print("  - Press SPACE to START/STOP recording")
print("  - Press 's' to save current recording")
print("  - Press 'q' to quit")
print(f"  - Current sample count: {sample_count}")
print(f"{'='*50}\n")

frame_buffer = []  # Store frames for the current recording
landmarks_buffer = []  # Store landmarks for the current recording

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw landmarks on frame
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
    else:
        # Draw warning if no hands detected
        cv2.putText(frame, "NO HAND DETECTED", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display status
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
        
        # Draw red circle to indicate recording
        cv2.circle(frame, (width - 30, 30), 15, (0, 0, 255), -1)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):  # Space to start/stop recording
        if not recording:
            # Start recording
            recording = True
            frame_count = 0
            frame_buffer = []
            landmarks_buffer = []
            
            # Initialize video writer
            timestamp = int(time.time())
            video_path = os.path.join(SAVE_DIR, f"{WORD}_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))
            
            print(f"Recording started...")
        else:
            # Stop recording
            recording = False
            if video_writer:
                video_writer.release()
                video_writer = None
            
            # Save landmarks if we have any
            if landmarks_buffer:
                timestamp = int(time.time())
                landmarks_dir = os.path.join(SAVE_DIR, f"{WORD}_{timestamp}_landmarks")
                os.makedirs(landmarks_dir, exist_ok=True)
                
                for i, landmarks in enumerate(landmarks_buffer):
                    np.save(os.path.join(landmarks_dir, f"frame_{i:03d}.npy"), landmarks)
                
                print(f"Saved {len(landmarks_buffer)} landmark frames to {landmarks_dir}")
            
            sample_count += 1
            frame_count = 0
            frame_buffer = []
            landmarks_buffer = []
            print(f"Recording stopped. Sample count: {sample_count}")
    
    elif key == ord('s') and recording:  # Save current recording
        recording = False
        if video_writer:
            video_writer.release()
            video_writer = None
        sample_count += 1
        print(f"Saved recording. Sample count: {sample_count}")
    
    elif key == ord('q'):  # Quit
        break

    # Record frames when recording
    if recording:
        # Save frame to video
        if video_writer:
            video_writer.write(frame)
        
        # Store frame and landmarks
        frame_buffer.append(frame.copy())
        
        if result.multi_hand_landmarks:
            # Combine landmarks from all hands
            all_landmarks = []
            for hand_landmarks in result.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    all_landmarks.append([lm.x, lm.y, lm.z])
            
            # If two hands, pad or combine appropriately
            if len(result.multi_hand_landmarks) == 1:
                # Single hand: 21 landmarks
                landmarks = np.array(all_landmarks)
            else:
                # Two hands: 42 landmarks (or pad to fixed size)
                landmarks = np.array(all_landmarks[:42])  # Take first 42 if more
            
            landmarks_buffer.append(landmarks)
        else:
            # No hand detected, append zeros or skip
            landmarks_buffer.append(np.zeros((21, 3)))
        
        frame_count += 1
        
        # Auto-stop after max recording time
        if frame_count >= MAX_RECORDING_TIME * FPS:
            recording = False
            if video_writer:
                video_writer.release()
                video_writer = None
            sample_count += 1
            print(f"Auto-stopped after {MAX_RECORDING_TIME} seconds. Sample count: {sample_count}")
            frame_count = 0
            frame_buffer = []
            landmarks_buffer = []

    cv2.imshow("Record Moving Hand Signs", frame)

# Cleanup
if video_writer:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()
hands.close()
print(f"\nDone! Collected {sample_count} samples for {WORD}.")
