# GTSDB
python ./dataset_postprocessors/pseudo_label_dataset.py -m best_WIDER.pt -i 3 -ip GTSDB/GTSDB_Train_and_Test/Train/images -lp GTSDB/GTSDB_Train_and_Test/Train/labels -c full
python ./dataset_postprocessors/pseudo_label_dataset.py -m best_WIDER.pt -i 3 -ip GTSDB/GTSDB_Train_and_Test/Test/images -lp GTSDB/GTSDB_Train_and_Test/Test/labels -c full

