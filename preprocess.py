import os
import cv2

def extract_frames_from_video(video_path, dest_directory, format='jpg'):
    """
    Extracts frames from a given video and saves them as images.
    
    Args:
    - video_path (str): Path to the video file.
    - dest_directory (str): Directory where image frames will be saved.
    - format (str): Format to save the images in ('jpg' or 'png').
    
    Returns:
    - None
    """
    
    # Make sure the destination directory exists
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    
    # Read the video using OpenCV
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        frame_file = os.path.join(dest_directory, f"frame_{count:04d}.{format}")
        cv2.imwrite(frame_file, image) 
        success, image = vidcap.read()
        count += 1


video_file = ""
dest_directory = ""

extract_frames_from_video(video_file, dest_directory)
