import argparse

import cv2
import csv
import time
import numpy as np
from ultralytics import YOLO
import torch
from diffusers import DiffusionPipeline
from PIL import Image

parser = argparse.ArgumentParser(description="YOLO Video Tracking")

parser.add_argument("-m", "--model", type=str, default="best.pt", help="Path to YOLO model")
parser.add_argument("-v", "--video", type=str, default="../videos/20260212_124301_f04acdba.mp4", help="Path to input video")
parser.add_argument("-b", "--bboxes", type=str, default="../videos/results-bboxes.csv", help="Path to input videos bounding boxes")
parser.add_argument("-o", "--output", type=str, default="../videos/results", help="Path to output video")

parser.add_argument(
    "-c",
    "--censoring-method",
    choices=["black-box", "blur"],
    help=(
        "Method to use for censoring detected objects. Options:\n"
        "  black-box : fill the detected area with a black rectangle\n"
        "  blur      : apply a Gaussian blur to anonymize the region\n"
    )
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
    }
}

model = YOLO(args.model)

def read_bboxes(frame_number):
    bboxes = []

    with open(args.bboxes, newline="") as f:
        bboxes = [row for row in csv.reader(f, delimiter=";") if row and int(row[0]) == frame_number]

    return bboxes

def black_box(frame, _, x1, y1, x2, y2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), -1)

def blur(frame, _, x1, y1, x2, y2):
    roi = frame[y1:y2, x1:x2]
    roi = cv2.GaussianBlur(roi, (25,25), 30)
    frame[y1:y2, x1:x2] = roi

def censor(censoring_method):
    cap = cv2.VideoCapture(args.video)

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"{args.output}-{censoring_method.__name__}.mp4", fourcc, fps, (width, height))

    prev_time = time.time()
    frame_number = 1

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, cap_frame = cap.read()

        if success:
            bboxes = read_bboxes(frame_number)
            for _, cls, x1, y1, x2, y2 in bboxes:
                cls, x1, y1, x2, y2 = map(int, [cls, x1, y1, x2, y2]) # Convert from str to int
                censoring_method(cap_frame, cls, x1, y1, x2, y2)

            out.write(cap_frame)

            cv2.imshow("YOLO Tracking", cap_frame)

            frame_number += 1

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    cap.release()
    out.release()

# Map string -> function for easy dispatch
methods = {
    "black-box": black_box,
    "blur": blur,
}

if args.censoring_method is None:
    for censoring_method in methods.values():
        censor(censoring_method)
else:
    censor(methods[args.censoring_method])

cv2.destroyAllWindows()