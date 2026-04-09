import warnings

from ultralytics.utils import LOGGER, SETTINGS, Path
from ultralytics.utils.checks import check_requirements

check_requirements("fiftyone")

import fiftyone as fo
import fiftyone.zoo as foz

def download_open_images_v7():
    classes = ["Traffic sign", "Vehicle registration plate", "Human face"]
    name = "open-images-v7"
    fo.config.dataset_zoo_dir = Path(SETTINGS["datasets_dir"]) / "fiftyone" / name
    fraction = 1.0  # fraction of full dataset to use
    for split in "train", "validation":  # 1743042 train, 41620 val images
        train = split == "train"

        # Load Open Images dataset
        dataset = foz.load_zoo_dataset(
            name,
            split=split,
            classes=classes,
            label_types=["detections"],
            max_samples=round((1743042 if train else 41620) * fraction),
        )

        # Export to YOLO format
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="fiftyone.utils.yolo")
            dataset.export(
                export_dir=str(Path(SETTINGS["datasets_dir"]) / name),
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth",
                split="val" if split == "validation" else split,
                classes=classes,
                overwrite=False,
            )

if __name__ == "__main__":
    download_open_images_v7()