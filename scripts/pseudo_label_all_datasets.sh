# Assumes you run this from the project root and have best_WIDER.pt located in the project root as well. This script will pseudo label all datasets with the WIDER face model.

set -e

# GTSDB
python dataset_postprocessors/pseudo_label_dataset.py -m best_WIDER.pt -i 3 -ip GTSDB/GTSDB_Train_and_Test/Train/images -lp GTSDB/GTSDB_Train_and_Test/Train/labels -c full
python dataset_postprocessors/pseudo_label_dataset.py -m best_WIDER.pt -i 3 -ip GTSDB/GTSDB_Train_and_Test/Test/images -lp GTSDB/GTSDB_Train_and_Test/Test/labels -c full

# Roboflow LP
python dataset_postprocessors/pseudo_label_dataset.py -m best_WIDER.pt -i 3 -ip License-Plate-Recognition-13/test/images -lp License-Plate-Recognition-13/test/labels -c full
python dataset_postprocessors/pseudo_label_dataset.py -m best_WIDER.pt -i 3 -ip License-Plate-Recognition-13/train/images -lp License-Plate-Recognition-13/train/labels -c full
python dataset_postprocessors/pseudo_label_dataset.py -m best_WIDER.pt -i 3 -ip License-Plate-Recognition-13/valid/images -lp License-Plate-Recognition-13/valid/labels -c full

# Open images V7
python dataset_postprocessors/pseudo_label_dataset.py -m best_WIDER.pt -i 3 -ip open-images-v7/images/train -lp open-images-v7/labels/train -c full
python dataset_postprocessors/pseudo_label_dataset.py -m best_WIDER.pt -i 3 -ip open-images-v7/images/val -lp open-images-v7/labels/val -c full

# UC3M-LP
python dataset_postprocessors/pseudo_label_dataset.py -m best_WIDER.pt -i 3 -ip UC3M-LP/images/train -lp UC3M-LP/labels/train -c full
python dataset_postprocessors/pseudo_label_dataset.py -m best_WIDER.pt -i 3 -ip UC3M-LP/images/val -lp UC3M-LP/labels/val -c full