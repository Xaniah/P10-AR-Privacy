# P10-AR-Privacy

## YOLO Configuration
### Update dataset directory
```bash
yolo settings datasets_dir="C:\Programming\Uni\P10-AR-Privacy\datasets"
```

## Training
```bash
python ./train.py
```

## Dataset utilities
### Count annotations for every subfolder
Used for the paper when we display number of annotations in the dataset table.
```bash
./scripts/count_lines.sh path/to/dir
```
The script will scan all subdirectories, count the number of lines in each file and output the total lines for all files.

### Download all datasets
```bash
python ./dataset_downloaders/download_all.py
```

### Pseudo-labelling
Pseudo-labelling involves using a model to annotate a dataset. For example a dataset containing traffic signs can also contain faces. The dataset itself only has annotations for traffic signs, so we must add the face annotations using the model. Depending on the dataset's size this can be crucial for model performance as having missing annotations can penalize the model during training.

```bash
 python ./dataset_postprocessors/pseudo_label_dataset.py -m best_WIDER.pt -i 3 -ip GTSDB/GTSDB_Train_and_Test/Train/images -lp GTSDB/GTSDB_Train_and_Test/Train/labels -c full
```
```bash
 python ./dataset_postprocessors/pseudo_label_dataset.py -m best_WIDER.pt -i 3 -ip GTSDB/GTSDB_Train_and_Test/Test/images -lp GTSDB/GTSDB_Train_and_Test/Test/labels -c full
```

**Parameters:**
- `-m` — Path to the model
- `-i` — ID to use (Must also match what is configured in the `dataset-config.yaml`, if you are going to train using this dataset)
- `-ip` — The path to the dataset images directory
- `-lp` — The path to the dataset labels directory
- `-c` — The configuration to use. Options: label, merge, full (label + merge)

The command pseudo-labels the GTSDB dataset, which is a dataset for traffic signs. We use a YOLO model training on WIDER face and when annotating we use ID 3. Note. In this example the `Test` split is used as the `Val` split.

## Server commands
The ucloud GPU server is where we perform training

### Copy a model to your local machine
```bash
scp -P <PORT> ucloud@ssh.cloud.sdu.dk:/work/P10-AR-Privacy/runs/detect/<TRAIN_NO>/weights/best.pt <LOCAL_PATH>
```

**Parameters:**
- `<PORT>` — SSH port (e.g. `2699`)
- `<TRAIN_NO>` — training run folder (e.g. `train5`)
- `<LOCAL_PATH>` — local destination path (e.g. `C:\Users\Frederik\Downloads\`)