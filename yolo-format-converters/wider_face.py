# Source: https://medium.com/@ahmetcan7754/i-converted-the-wider-face-detection-dataset-to-yolo12-format-8b47d1ec8c88
import os
from PIL import Image

# Paths
source_label_file = "wider_face_train_bbx_gt.txt"
images_base_dir = "WIDER_train/images"
target_dir = "labels"

# Create target directory if not exists
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Read label file
with open(source_label_file, "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    if "/" in line or line.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = line
        i += 1
    else:
        i += 1
        continue

    # Get bounding boxes count
    bbox_count = int(lines[i].strip()) if lines[i].strip().isdigit() else 0
    i += 1

    bboxes = []
    for _ in range(bbox_count):
        bbox = list(map(float, lines[i].split()[:4]))
        bboxes.append(bbox)
        i += 1

    print(bboxes)
    # Get image dimensions
    image_full_path = os.path.join(images_base_dir, image_path)
    with Image.open(image_full_path) as img:
        img_width, img_height = img.size


    # Convert to YOLO format
    yolo_lines = [f"0 {(x + w / 2.0) / img_width} {(y + h / 2.0) / img_height} {w / img_width} {h / img_height}" 
                  for x, y, w, h in bboxes]

    # Write to label file
    output_file = os.path.join(target_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
    with open(output_file, "w") as out_f:
        out_f.write("\n".join(yolo_lines))

    print(f"Created label file: {output_file}")