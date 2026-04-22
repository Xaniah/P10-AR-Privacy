# Source: https://github.com/ramajoballester/UC3M-LP/blob/main/scripts/labels2yolo.py
import os
import json
import shutil
import cv2
import argparse
from tqdm import tqdm

from utils.dataset_config_utils import get_dataset_class_id_by_name

def poly2bbox(poly_coord):
    x_coords = [coord[0] for coord in poly_coord]
    y_coords = [coord[1] for coord in poly_coord]
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    return [[x_min, y_min], [x_max, y_max]]


def create_txt_file(input_directory):
    # Read files in lp_directory
    train_files = os.listdir(os.path.join(input_directory, 'images', 'train'))
    val_files = os.listdir(os.path.join(input_directory, 'images', 'val'))
    train_files.sort()
    val_files.sort()

    # Save train and val files in train.txt and val.txt
    with open(os.path.join(input_directory, 'train.txt'), 'w') as f:
        for filename in train_files:
            f.write(filename.split('.')[0] + '\n')
    
    with open(os.path.join(input_directory, 'val.txt'), 'w') as f:
        for filename in val_files:
            f.write(filename.split('.')[0] + '\n')

def create_yolo_bbox_string(class_id, bbox, img_width, img_height):
    x_center = (bbox[0][0] + bbox[1][0]) / (2 * img_width)
    y_center = (bbox[0][1] + bbox[1][1]) / (2 * img_height)
    width = (bbox[1][0] - bbox[0][0]) / img_width
    height = (bbox[1][1] - bbox[0][1]) / img_height
    return f'{class_id} {x_center} {y_center} {width} {height}'


def transform_dataset(input_directory, lp_size):
    train_txt_path = os.path.join(input_directory, 'train.txt')
    test_txt_path = os.path.join(input_directory, 'test.txt')

    # Normalize input directory
    input_directory = os.path.normpath(input_directory)

    output_directory = input_directory
    
    # Create directories if not exist
    os.makedirs(os.path.join(output_directory, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_directory, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_directory, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_directory, 'labels', 'val'), exist_ok=True)

    for txt_file in [train_txt_path, test_txt_path]:
        split = 'train' if 'train' in txt_file else 'test'
        yolo_split = 'train' if 'train' in txt_file else 'val'

        with open(txt_file, 'r') as f:
            filenames = f.read().splitlines()

        print(f'[UC3M-LP]: Processing {split} split')
        for filename in tqdm(filenames):
            # Load image
            img_path = os.path.join(input_directory, split, filename + '.jpg')
            img = cv2.imread(img_path)
            img_height, img_width, _ = img.shape

            # Load JSON label
            json_path = os.path.join(input_directory, split, filename + '.json')
            with open(json_path, 'r') as f:
                data = json.load(f)

            # License Plate Detection Dataset
            for lp_data in data['lps']:
                lp_img = img.copy()
                poly_coord = lp_data['poly_coord']

                # Convert polygonal annotation to rectangular bbox
                lp_bbox = poly2bbox(poly_coord)

                # Write license plate image
                lp_output_path = os.path.join(output_directory, 'images', yolo_split, f'{filename}.jpg')
                if not os.path.isfile(lp_output_path):
                    rescale_factor_lp = lp_size / max(img_height, img_width)
                    lp_img_resized = cv2.resize(
                        lp_img,
                        (int(img_width * rescale_factor_lp), int(img_height * rescale_factor_lp)),
                    )
                    cv2.imwrite(lp_output_path, lp_img_resized)

                # Write YOLO bbox annotation for license plate
                lp_yolo_path = os.path.join(output_directory, 'labels', yolo_split, f'{filename}.txt')
                append_write_lp = 'a' if os.path.exists(lp_yolo_path) else 'w'
                with open(lp_yolo_path, append_write_lp) as lp_f:
                    lp_f.write(create_yolo_bbox_string(get_dataset_class_id_by_name("License plate"), lp_bbox, img_width, img_height) + '\n')

    create_txt_file(output_directory)
    
    # Cleanup
    os.remove(train_txt_path)
    os.remove(test_txt_path)
    os.remove(input_directory + '/val.txt')
    shutil.rmtree(input_directory + '/train')
    shutil.rmtree(input_directory + '/test')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_directory', type=str, help='Path to input dataset')
    parser.add_argument('lp_size', type=int, help='YOLO input size for LP detection')
    args = parser.parse_args()
    
    transform_dataset(args.input_directory, args.lp_size)