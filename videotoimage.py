import cv2
import os
# Define input video path and output folder
video_path = 'C:\\Intel 2024\\alone and sleep 1.mp4'
output_folder = 'C:\\Intel 2024\\alone and sleep frame'

def video_to_frames(video_path, output_folder, fps=1):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            saved_frame_count += 1
            # Save frame as image
            frame_filename = os.path.join(output_folder, f'frame_{saved_frame_count:04d}.png')
            cv2.imwrite(frame_filename, frame)
            print(f'Saved {frame_filename}')
        
        frame_count += 1

    cap.release()
    print(f'Total frames saved: {saved_frame_count}')

# Call the function
video_to_frames(video_path, output_folder, fps=1)