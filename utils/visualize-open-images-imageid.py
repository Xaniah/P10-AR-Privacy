import argparse

import fiftyone as fo
import fiftyone.zoo as foz

parser = argparse.ArgumentParser(description="Visualize Open Images v7 dataset using FiftyOne")
parser.add_argument("-d", "--dataset", type=str, help="Path to Open Images v7 dataset\nIt is recommended to use a path different from the one used by the dataset downloader to avoid conflicts with the database link.")
args = parser.parse_args()

fo.config.dataset_zoo_dir = args.dataset

image_ids = ["0a0a00b2fbe89a47"]

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    image_ids=image_ids,
    dataset_name="my-open-images-v2",
    drop_existing_dataset=True,
)

session = fo.launch_app(dataset)
session.wait()