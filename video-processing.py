import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("runs/detect/WIDER_train/weights/best.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    # Convert to supervision Rectangles
    
    rectangles = sv.Rect.from_xyxy(detections.xyxy)

    # Draw filled black rectangles
    output = sv.draw_filled_rectangle(
        frame.copy(),
        rectangles=rectangles,
        thickness=-1,            # filled
        color=(0, 0, 0),         # black
        alpha=1.0                # opaque
    )

    return output

sv.process_video(
    source_path="videos/20260212_124301_f04acdba.mp4",
    target_path="videos/result_20260212_124301_f04acdba.mp4",
    callback=callback
)