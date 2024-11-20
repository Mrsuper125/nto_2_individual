import os
from shutil import copy

import pandas as pd

from sklearn.model_selection import train_test_split

from tqdm import tqdm

def make_folders(path, split):
    labels_directory = os.path.join(path, "labels", split)
    images_directory = os.path.join(path, "images", split)
    os.makedirs(labels_directory, exist_ok=True)
    os.makedirs(images_directory, exist_ok=True)  # Create folder <path>/labels/<split> and <saving_path>/images/<split>

    return labels_directory, images_directory


def image_preprocess(labels_directory, images_directory, source_images_path, source_file_name,
                     markup):  # Last 3 arguments are image and predictions on it. Write the image into needed directory and parse&save labels
    yolo_markups = []

    for detail in markup.split(";"):  # Split line by semicolons and add elements to an array
        yolo_markups.append(f"{detail}\n")

    label_file_path = os.path.join(labels_directory, source_file_name[
                                                     :-4] + ".txt")  # For file ABC.JPG create path <labels_directory>/ABC.txt
    with open(label_file_path, "w") as file:
        file.writelines(yolo_markups)  # Write all markup elements to the file

    image_source_path = os.path.join(source_images_path, source_file_name)  # Create the whole path to the source image
    image_destination_path = os.path.join(images_directory,
                                          source_file_name)  # Create the whole path to the target image

    copy(image_source_path, image_destination_path)  # write the file to the targeted destination


def yolo_files_preprocess(data_df_path, yolo_data_directory, source_images_directory):
    data_df: pd.DataFrame = pd.read_csv(data_df_path)

    data_df_train, data_df_val = train_test_split(data_df, test_size=0.3, random_state=0)  # split datasets

    labels_directory, images_directory = make_folders(yolo_data_directory, "train")
    for index, row in tqdm(data_df_train.iterrows()):
        image_preprocess(labels_directory, images_directory, source_images_directory, row["image_name"],
                         row["detection"])

    labels_directory, images_directory = make_folders(yolo_data_directory, "val")
    for index, row in tqdm(data_df_val.iterrows()):
        image_preprocess(labels_directory, images_directory, source_images_directory, row["image_name"],
                         row["detection"])
