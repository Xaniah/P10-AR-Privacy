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
parser.add_argument("-v", "--video", type=str, default="videos/20260212_124301_f04acdba.mp4", help="Path to input video")
parser.add_argument("-o", "--output", type=str, default="videos/results.mp4", help="Path to output video")

parser.add_argument(
    "-c",
    "--censoring-method",
    choices=["black-box", "blur", "morph"],
    default="black-box",
    help=(
        "Method to use for censoring detected objects. Options:\n"
        "  black-box : fill the detected area with a black rectangle\n"
        "  blur      : apply a Gaussian blur to anonymize the region\n"
        "  morph     : pixelate / mosaic the detected area"
    )
)

parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="Enable debug mode: Display bounding boxes, tracking IDs and FPS on the video output"
)

args = parser.parse_args()

ALLOWED = {"person", "Face", "face", "Machine printed", "Handwritten", "Other text", "license plate", "Traffic sign"}

def black_box(frame, x1, y1, x2, y2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), -1)

def blur(frame, x1, y1, x2, y2):
    roi = frame[y1:y2, x1:x2]
    roi = cv2.GaussianBlur(roi, (25,25), 30)
    frame[y1:y2, x1:x2] = roi


pipe = None
MORPH_PROMPT = (
    "Replace the face of the person with a different realistic human face, "
    "keeping the same pose, lighting, camera angle, hairstyle, and background. "
    "Maintain natural skin texture and consistent shadows. The new face should blend "
    "seamlessly with the body and environment, photorealistic, high detail, no artifacts."
)

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

def morph(frame, x1, y1, x2, y2):
    global pipe

    if pipe is None:
        print("Loading FLUX.2-klein-4B model…")
        pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        pipe.enable_model_cpu_offload()

    x1_exp, y1_exp, x2_exp, y2_exp = expand_box(x1, y1, x2, y2, frame.shape, scale=2.2)

    crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]

    # Ensure minimum size 64x64 (FLUX.2-klein-4B has a minimum size requirement)
    min_size = 64
    h, w = crop.shape[:2]
    if h < min_size or w < min_size:
        scale = max(min_size / h, min_size / w)
        new_w, new_h = int(w*scale), int(h*scale)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    output_image = pipe(image=pil_crop, prompt=MORPH_PROMPT).images[0]

    # Convert back and resize to original tight bounding box
    output_cv2 = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
    #frame[y1:y2, x1:x2] = cv2.resize(output_cv2, (x2 - x1, y2 - y1))
    frame[y1_exp:y2_exp, x1_exp:x2_exp] = cv2.resize(output_cv2, (x2_exp - x1_exp, y2_exp - y1_exp))

    if args.debug:
        cv2.rectangle(frame, (x1_exp, y1_exp), (x2_exp, y2_exp), (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

# Map string -> function for easy dispatch
methods = {
    "black-box": black_box,
    "blur": blur,
    "morph": morph
}

model = YOLO(args.model)
cap = cv2.VideoCapture(args.video)

# Get video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

prev_time = time.time()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, cap_frame = cap.read()

    if success:
        # Run YOLO26 tracking on the frame, persisting tracks between frames
        results = model.track(cap_frame, persist=True)
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

                methods[args.censoring_method](frame, x1, y1, x2, y2)
                if args.debug:
                    cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        if args.debug:
            # ---- FPS calculation ----
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        out.write(frame)

        cv2.imshow("YOLO Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

cap.release()
out.release()
cv2.destroyAllWindows()