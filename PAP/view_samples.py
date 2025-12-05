import cv2
import os
import sys
import glob

def view_video(video_path):
    """Play a video file"""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / fps) if fps > 0 else 30
    
    print(f"\nPlaying: {os.path.basename(video_path)}")
    print("Press 'q' to quit, SPACE to pause/resume")
    
    paused = False
    frame_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video. Press 'q' to exit or 'r' to replay.")
                key = cv2.waitKey(0) & 0xFF
                if key == ord('r'):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    continue
                elif key == ord('q'):
                    break
                else:
                    break
            
            frame_count += 1
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Add info overlay
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit, SPACE to pause", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Video Player", frame)
        
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")
    
    cap.release()
    cv2.destroyAllWindows()


def list_samples(word=None):
    """List all available samples"""
    base_dir = os.path.join("dataset", "psl_words")
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        return []
    
    if word:
        # List samples for specific word
        word_dir = os.path.join(base_dir, word.replace(' ', '_').title())
        if not os.path.exists(word_dir):
            print(f"No samples found for: {word}")
            return []
        
        videos = glob.glob(os.path.join(word_dir, "*.mp4"))
        return sorted(videos)
    else:
        # List all words and their sample counts
        words = {}
        for word_folder in os.listdir(base_dir):
            word_path = os.path.join(base_dir, word_folder)
            if os.path.isdir(word_path):
                videos = glob.glob(os.path.join(word_path, "*.mp4"))
                words[word_folder] = len(videos)
        
        return words


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python view_samples.py <WORD>              # View all samples for a word")
        print("  python view_samples.py <WORD> <INDEX>      # View specific sample by index")
        print("  python view_samples.py list                # List all available words")
        print("\nExamples:")
        print("  python view_samples.py obrigado")
        print("  python view_samples.py 'bom dia'")
        print("  python view_samples.py obrigado 0          # View first sample")
        sys.exit(1)
    
    if sys.argv[1].lower() == "list":
        words = list_samples()
        if not words:
            print("No samples found.")
            return
        
        print("\nAvailable words and sample counts:")
        print("-" * 40)
        for word, count in sorted(words.items()):
            print(f"  {word:20s} : {count} sample(s)")
        print("-" * 40)
        return
    
    word = sys.argv[1]
    videos = list_samples(word)
    
    if not videos:
        print(f"No video samples found for: {word}")
        return
    
    print(f"\nFound {len(videos)} sample(s) for '{word}':")
    for i, video in enumerate(videos):
        print(f"  [{i}] {os.path.basename(video)}")
    
    # If index specified, play that specific video
    if len(sys.argv) >= 3:
        try:
            index = int(sys.argv[2])
            if 0 <= index < len(videos):
                view_video(videos[index])
            else:
                print(f"Error: Index {index} out of range. Use 0-{len(videos)-1}")
        except ValueError:
            print(f"Error: '{sys.argv[2]}' is not a valid index number")
    else:
        # Play all videos in sequence
        print("\nPlaying all samples. Press 'q' to quit, 'n' for next video.")
        for i, video in enumerate(videos):
            print(f"\n[{i+1}/{len(videos)}] Playing: {os.path.basename(video)}")
            view_video(video)
            
            # Ask if user wants to continue
            if i < len(videos) - 1:
                print("\nPress 'n' for next video, 'q' to quit...")
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break


if __name__ == "__main__":
    main()

