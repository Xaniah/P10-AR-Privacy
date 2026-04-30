import argparse
import random

import cv2

parser = argparse.ArgumentParser(description="YOLO Video Tracking")

parser.add_argument("-v", "--video", type=str, default="../videos/20260212_124301_f04acdba.mp4", help="Path to input video")
parser.add_argument("-f", "--frames", nargs="+", type=int, help="Start and end frames for interval of userful frames")
parser.add_argument("-e", "--exclude", nargs="+", type=int, help="List of frames to exclude")
parser.add_argument("-n", "--number", type=int, default=4, help="Number of frames to save")

args = parser.parse_args()



def save_images():
    cap = cv2.VideoCapture(args.video)

    exclude = set(args.exclude) if args.exclude else set()

    frames = [x for x in range(args.frames[0], args.frames[1] + 1) if x not in exclude]

    frame_numbers = random.sample(frames, args.number)

    for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ret, frame = cap.read()

        if not ret:
            print(f"Could not read frame {frame_number}")
            continue

        cv2.imwrite(f"{args.video.split(".mp4")[0]}-frame_{frame_number}.jpg", frame)

    cap.release()


if (len(args.frames) != 2):
    raise ValueError("There must be a start frame and an end frame")

save_images()