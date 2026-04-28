import argparse

import cv2

parser = argparse.ArgumentParser(description="YOLO Video Tracking")

parser.add_argument("-v", "--video", type=str, default="../videos/20260212_124301_f04acdba.mp4", help="Path to input video")
parser.add_argument("-f", "--frames", nargs="+", type=int, help="List of frames to save to image")

args = parser.parse_args()

def save_images():
    cap = cv2.VideoCapture(args.video)

    for frame_number in args.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ret, frame = cap.read()

        if not ret:
            print(f"Could not read frame {frame_number}")
            continue

        cv2.imwrite(f"{args.video.split(".mp4")[0]}-frame_{frame_number}.jpg", frame)

    cap.release()


save_images()