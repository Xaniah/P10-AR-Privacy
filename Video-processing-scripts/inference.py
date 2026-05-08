import argparse
from pathlib import Path

import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch
from diffusers import DiffusionPipeline
from PIL import Image

parser = argparse.ArgumentParser(description="YOLO Video Tracking")

parser.add_argument("-m", "--model", type=str, default="v2.pt", help="Path to YOLO model")
parser.add_argument("-v", "--video", type=str, default="../videos/chosen-videos/", help="Path to input video")
parser.add_argument("-o", "--output", type=str, default="../videos/chosen-videos/", help="Path to output video")
parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="Enable debug mode: Display bounding boxes, tracking IDs and FPS on the video output"
)

args = parser.parse_args()

ALLOWED = {
    "Face": {
        "prompt": "Replace the face of the person with a different realistic human face, "
            "keeping the same pose, lighting, camera angle, hairstyle, and background. "
            "Maintain natural skin texture and consistent shadows. The new face should blend "
            "seamlessly with the body and environment, photorealistic, high detail, no artifacts.", 
        "bbx_scale": 2.2
    },

    "License plate": {
        "prompt": "Replace the license plate with a different one, maintaining the same font, color, and style. "
            "Replace the text with random characters i.e. 'ER 21 531' can become 'AB 34 789', 'CD 56 123', etc. "
            "Ensure the new plate is clearly visible and matches the vehicle's make and model.",
        "bbx_scale": 1.5
    },

    "Traffic sign": {
        "prompt": "Replace the traffic sign with a different privatized one, maintaining the same shape and color. Scramble the text."
            "Ensure the new sign is clearly visible and matches the location and context.",
        "bbx_scale": 1.5
    },

    "Laptop": {
        "prompt": "Replace the laptop with a different one displaying a privatized screen and keyboard, maintaining the same shape and color.",
        "bbx_scale": 1.5
    }
}

model = YOLO(args.model)

def process_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    stem = video_path.stem
    out_bboxes = f"{args.output}/{stem}-bboxes.csv"
    out_video  = f"{args.output}/{stem}-bboxes.mp4"

    out = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    open(out_bboxes, "w").close()  # wipe file once

    prev_time = time.time()
    frame_number = 1

    with open(out_bboxes, "a") as f:
        while cap.isOpened():
            success, cap_frame = cap.read()

            if not success:
                break

            # Run YOLO tracking
            results = model.track(cap_frame, persist=True, imgsz=1280)
            r = results[0]

            frame = r.orig_img
            boxes = r.boxes

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])

                    if model.names[cls] not in ALLOWED:
                        continue

                    track_id = int(box.id[0]) if box.id is not None else -1

                    label = f"{model.names[cls]} ID:{track_id}"

                    f.write(
                        f"{frame_number};{model.names[cls]};"
                        f"{x1};{y1};{x2};{y2}\n"
                    )

                    if args.debug:
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2
                        )

            if args.debug:
                current_time = time.time()
                fps_display = 1 / (current_time - prev_time)
                prev_time = current_time

                cv2.putText(
                    frame,
                    f"FPS: {fps_display:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("YOLO Tracking", frame)

            out.write(frame)
            frame_number += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()


def inference():
    video_dir = Path(args.video)

    # Find all Scenario-x-yy.mp4 files
    videos = sorted(video_dir.glob("Scenario-*-*.mp4"))

    if not videos:
        print(f"No matching videos found in {video_dir}")
        return

    for video_path in videos:
        print(f"Processing: {video_path}")
        process_video(video_path)

    cv2.destroyAllWindows()
inference()