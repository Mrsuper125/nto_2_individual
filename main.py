import os
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from output import inference
from preprocessing import yolo_files_preprocess

import yaml

from visualisation import visualization_by_idx_prediction

data_df_path = "data/train.csv"
yolo_data_dir = "yolo/"
images_data_dir = "data/train"

Path("yolo/images/train").mkdir(exist_ok=True, parents=True)
Path("yolo/images/val").mkdir(exist_ok=True, parents=True)
Path("yolo/labels/train").mkdir(exist_ok=True, parents=True)
Path("yolo/labels/val").mkdir(exist_ok=True, parents=True)

yolo_files_preprocess(data_df_path, yolo_data_dir, images_data_dir)

data = {
    "path": "../",
    "train": "yolo/images/train",
    "val": "yolo/images/val",
    "nc": 10,
    "save_period": 1,
    "names": ['Заяц', 'Кабан', 'Кошки', 'Куньи', 'Медведь', 'Оленевые', 'Пантеры', 'Полорогие', 'Собачие', 'Сурок']
}

with open('yolo/yolo_config.yaml', 'w') as file:
    yaml.dump(data, file, default_flow_style=False, allow_unicode=True)

os.environ['WANDB_MODE'] = 'disabled'


def train(model_info, config_file):
    model = YOLO(model_info)
    training_results = model.train(
        data=config_file,
        epochs=25,  # число эпох для обучения
        imgsz=640,  # размер изображения для обучения
        batch=4,  # размер батча для обучения
        device=0,  # номер девайса для обучения
        #resume = True,
        single_cls=False  # для обучения с учетом классов на основании data.yaml
    )


train("./yolo11x.pt", "./yolo/yolo_config.yaml")

"""

path_detection_model_cpt = Path("runs/detect/train2/weights/best.pt")
output_path = Path("submission.csv")
dir_test_images = Path("data/check/images")

detection_model = YOLO(path_detection_model_cpt)
inference(detection_model, dir_test_images, output_path)

MAPPER = ['Заяц', 'Кабан', 'Кошки', 'Куньи', 'Медведь', 'Оленевые', 'Пантеры', 'Полорогие', 'Собачие', 'Сурок']

idx = 3
submission_df = pd.read_csv("submission.csv")
visualization_by_idx_prediction(dir_test_images, submission_df, idx, MAPPER)

"""
