import argparse
import random

import cv2

parser = argparse.ArgumentParser(description="YOLO Video Tracking")

parser.add_argument("-v", "--video", type=str, default="../videos/chosen-videos/Scenario-6-04.mp4", help="Path to input video")
parser.add_argument("-f", "--frames", nargs="+", type=int, help="Start and end frames for interval of userful frames")
parser.add_argument("-e", "--exclude", nargs="+", type=int, help="List of frames to exclude")
parser.add_argument("-n", "--number", type=int, default=4, help="Number of frames to save")

args = parser.parse_args()

def save_images():
    videos = [
        args.video,
        f"{args.video.split('.mp4')[0]}-blur.mp4",
        f"{args.video.split('.mp4')[0]}-black_box.mp4",
    ]

    caps = {video: cv2.VideoCapture(video) for video in videos}

    exclude = set(args.exclude) if args.exclude else set()

    frames = [x for x in range(args.frames[0], args.frames[1] + 1) if x not in exclude]

    frame_numbers = random.sample(frames, args.number)

    for frame_number in frame_numbers:
        for video, cap in caps.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ret, frame = cap.read()

            if not ret:
                print(f"Could not read frame {frame_number} from {video}")
                continue

            output_name = (
                f"{video.split('.mp4')[0]}-frame_{frame_number}.jpg"
            )

            cv2.imwrite(output_name, frame)

    for cap in caps.values():
        cap.release()


if len(args.frames) != 2:
    raise ValueError("There must be a start frame and an end frame")

save_images()