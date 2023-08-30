# import os
# import cv2
#
# def extract_frames_from_video(video_path, dest_directory, format='jpg'):
#     """
#     Extracts frames from a given video and saves them as images.
#
#     Args:
#     - video_path (str): Path to the video file.
#     - dest_directory (str): Directory where image frames will be saved.
#     - format (str): Format to save the images in ('jpg' or 'png').
#
#     Returns:
#     - None
#     """
#
#     # Make sure the destination directory exists
#     if not os.path.exists(dest_directory):
#         os.makedirs(dest_directory)
#
#     # Read the video using OpenCV
#     vidcap = cv2.VideoCapture(video_path)
#     success, image = vidcap.read()
#     count = 0
#     while success:
#         frame_file = os.path.join(dest_directory, f"frame_{count:04d}.{format}")
#         cv2.imwrite(frame_file, image)
#         success, image = vidcap.read()
#         count += 1
#
#
# video_file = "/home/yeongjun/DEV/e2e/data/videos_3d/ttl=7d"
# dest_directory = "/home/yeongjun/DEV/e2e/data/preprocessed"
#
# extract_frames_from_video(video_file, dest_directory)

import os
import cv2
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('input_path', 'data/videos_3d', 'Path to the directory containing the videos.')
flags.DEFINE_string('output_path', 'data/frames', 'Path to save the extracted frames.')
flags.DEFINE_string('file_format', 'jpg', 'Format to save frames (jpg or png).')
flags.DEFINE_integer('frame_interval', 1, 'Interval to save frames. E.g., 1 means every frame, 2 means every second frame, and so on.')

def extract_frames(video_path, output_folder, file_format='jpg', interval=1):
    """Extract frames from video and save as images."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % interval == 0:
            file_name = f"frame_{count:04d}.{file_format}"
            cv2.imwrite(os.path.join(output_folder, file_name), image)
        success, image = vidcap.read()
        count += 1

def main(_):
    input_folder = FLAGS.input_path
    output_folder = FLAGS.output_path
    file_format = FLAGS.file_format
    interval = FLAGS.frame_interval

    for video_file in os.listdir(input_folder):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(input_folder, video_file)
            extract_output_folder = os.path.join(output_folder, os.path.splitext(video_file)[0])
            extract_frames(video_path, extract_output_folder, file_format, interval)

if __name__ == "__main__":
    app.run(main)
