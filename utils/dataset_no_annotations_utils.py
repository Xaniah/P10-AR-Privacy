
from pathlib import Path

# GTSDB and Roboflow LP have more images than annotations
def find_image_with_no_labels(path="./datasets/GTSDB/GTSDB_Train_and_Test/Train"):
    train_dir = Path(path)
    image_dir = train_dir / "images"
    label_dir = train_dir / "labels"

    images = list(image_dir.glob("*.jpg")) + \
                list(image_dir.glob("*.png")) + \
                list(image_dir.glob("*.jpeg"))
    
    labels = [p.name for p in label_dir.glob("*.txt")]

    no_labels = []

    for image in images:
        occurences = labels.count(f"{image.stem}.txt")
        if occurences == 0:
            no_labels.append(image)

    return no_labels

no_labels = find_image_with_no_labels()
print(no_labels)