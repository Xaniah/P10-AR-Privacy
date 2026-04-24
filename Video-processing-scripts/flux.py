import argparse

import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch
from diffusers import DiffusionPipeline
from PIL import Image

parser = argparse.ArgumentParser(description="YOLO Video Tracking")

parser.add_argument("-m", "--model", type=str, default="best_WIDER.pt", help="Path to YOLO model")
parser.add_argument("-v", "--video", type=str, default="../videos/20260212_124301_f04acdba.mp4", help="Path to input video")
parser.add_argument("-o", "--output", type=str, default="../videos/results-flux.mp4", help="Path to output video")

parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="Enable debug mode: Display bounding boxes, tracking IDs and FPS on the video output"
)

args = parser.parse_args()

PATH_TO_FRAMES = "../frames/"

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

pipe = None

model = YOLO(args.model)

def expand_box(x1, y1, x2, y2, frame_shape, scale=1.5):
    """
    Expand the bounding box by `scale` (1.0 = original size)
    frame_shape = (height, width, channels)
    """
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w // 2
    cy = y1 + h // 2

    new_w = int(w * scale)
    new_h = int(h * scale)

    new_x1 = max(cx - new_w // 2, 0)
    new_y1 = max(cy - new_h // 2, 0)
    new_x2 = min(cx + new_w // 2, frame_shape[1])
    new_y2 = min(cy + new_h // 2, frame_shape[0])

    return new_x1, new_y1, new_x2, new_y2

def morph(frame, cls, x1, y1, x2, y2):
    global pipe
    class_config = ALLOWED[model.names[cls]]

    if pipe is None:
        print("Loading FLUX.2-klein-4B model…")
        pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        pipe.enable_model_cpu_offload()

    x1_exp, y1_exp, x2_exp, y2_exp = expand_box(x1, y1, x2, y2, frame.shape, scale=class_config["bbx_scale"])

    crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]

    # Ensure minimum size 64x64 (FLUX.2-klein-4B has a minimum size requirement)
    min_size = 64
    h, w = crop.shape[:2]
    if h < min_size or w < min_size:
        scale = max(min_size / h, min_size / w)
        new_w, new_h = int(w*scale), int(h*scale)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    output_image = pipe(image=pil_crop, prompt=class_config["prompt"]).images[0]

    # Convert back and resize to original tight bounding box
    output_cv2 = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
    #frame[y1:y2, x1:x2] = cv2.resize(output_cv2, (x2 - x1, y2 - y1))
    frame[y1_exp:y2_exp, x1_exp:x2_exp] = cv2.resize(output_cv2, (x2_exp - x1_exp, y2_exp - y1_exp))

    if args.debug:
        cv2.rectangle(frame, (x1_exp, y1_exp), (x2_exp, y2_exp), (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

