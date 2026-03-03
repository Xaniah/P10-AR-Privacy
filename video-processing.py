import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolo26s.pt")
tracker = sv.ByteTrack()
color_annotator = sv.ColorAnnotator(opacity=1, color=sv.Color.BLACK)

# Choose classes here (by name and/or id). Example: {"person", "car"} or {0, 2}
# Set to None or empty set to draw all classes.
SELECTED_CLASSES = {"person", "face", "Machine printed", "Handwritten", "Other text", "license plate", "Traffic sign"}

def _resolve_selected_class_ids(selected_classes, names):
    if not selected_classes:
        return None

    # names can be dict{id: name} or list[name]
    if isinstance(names, dict):
        name_to_id = {name: cid for cid, name in names.items()}
    else:
        name_to_id = {name: i for i, name in enumerate(names)}

    class_ids = []
    for c in selected_classes:
        if isinstance(c, int):
            class_ids.append(c)
        elif isinstance(c, str) and c in name_to_id:
            class_ids.append(name_to_id[c])
        else:
            print(f"Warning: class '{c}' not found in model names, skipping.")

    return np.array(sorted(set(class_ids)), dtype=np.int32) if class_ids else None

SELECTED_CLASS_IDS = _resolve_selected_class_ids(SELECTED_CLASSES, model.names)

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    if len(detections.xyxy) == 0:
        return frame

    # Filter detections by selected classes
    if SELECTED_CLASS_IDS is not None and detections.class_id is not None:
        mask = np.isin(detections.class_id.astype(np.int32), SELECTED_CLASS_IDS)
        detections = detections[mask]

    annotated_frame = color_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    return annotated_frame

sv.process_video(
    source_path="videos/20260212_124518_bcd950d6.mp4",
    target_path="videos/result_20260212_124518_bcd950d6.mp4",
    callback=callback
)