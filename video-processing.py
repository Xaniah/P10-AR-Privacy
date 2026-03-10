import cv2
import time
from ultralytics import YOLO

VIDEO_PATH = "videos/20260212_124301_f04acdba.mp4"
OUTPUT_PATH = "videos/results.mp4"
ALLOWED = {"person", "face", "Machine printed", "Handwritten", "Other text", "license plate", "Traffic sign"}

model = YOLO("best_WIDER.pt")
cap = cv2.VideoCapture(0)

# Get video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

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

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), -1)
                cv2.putText(frame, label, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

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