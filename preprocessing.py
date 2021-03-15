import os
import shutil
import zipfile
import urllib
import pandas as pd
import numpy as np


TRAIN_ZIP_URL = "https://aicrowd-practice-challenges.s3.us-west-002.backblazeb2.com/public/roverclassification/v0.1/train.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000003%2F20210314%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20210314T174808Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=80345413f4a175614f71638c6d391e7b8b138ddf2e29e26bf051971e12135022"
TRAIN_ZIP_FILE_PATH = "./src/train.zip"
TRAIN_LABELS_URL = "https://aicrowd-practice-challenges.s3.us-west-002.backblazeb2.com/public/roverclassification/v0.1/train.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000003%2F20210314%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20210314T174808Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=8be7fd371905eeb62c717e8556f8375cbd2f3a26908f2188745ad93b6fa93d0c"
TRAIN_LABELS_FILE = "./src/train.csv"
TRAIN_FINAL_PATH = "./dataset/train"

VAL_ZIP_URL = "https://aicrowd-practice-challenges.s3.us-west-002.backblazeb2.com/public/roverclassification/v0.1/val.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000003%2F20210314%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20210314T174808Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=f7f4c1bf48c0ad0c65874313f16f4f6a3202c5e2c68a3f7ccf83c22673e85677"
VAL_ZIP_FILE_PATH = "./src/val.zip"
VAL_LABELS_URL = "https://aicrowd-practice-challenges.s3.us-west-002.backblazeb2.com/public/roverclassification/v0.1/val.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000003%2F20210314%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20210314T174808Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=6308899139a0b56c7c68d5b0eb4e873de80282baa88b2c00a9c517744b7435bf"
VAL_LABELS_FILE = "./src/val.csv"
VAL_FINAL_PATH = "./dataset/val"

TEST_ZIP_URL = "https://aicrowd-practice-challenges.s3.us-west-002.backblazeb2.com/public/roverclassification/v0.1/test.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=002ae2491b744be0000000003%2F20210314%2Fus-west-002%2Fs3%2Faws4_request&X-Amz-Date=20210314T174808Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=b9982fb8479136219648afc020d924cf9bdac8904bdcc7e3e0c9a84b7aff23a4"
TEST_ZIP_FILE_PATH = "./src/test.zip"
TEST_LABELS_FILE = "./src/test.csv"
TEST_FINAL_PATH = "./dataset/test"

URLS = [TRAIN_ZIP_URL, TRAIN_LABELS_URL, VAL_ZIP_URL, VAL_LABELS_URL, TEST_ZIP_URL]
SRC_FILES = [TRAIN_ZIP_FILE_PATH, TRAIN_LABELS_FILE, VAL_ZIP_FILE_PATH, VAL_LABELS_FILE, TEST_ZIP_FILE_PATH]
ZIP_FILES_PATHS = [TRAIN_ZIP_FILE_PATH, VAL_ZIP_FILE_PATH, TEST_ZIP_FILE_PATH]
LABELS_FILES = [TRAIN_LABELS_FILE, VAL_LABELS_FILE, TEST_LABELS_FILE]
FINAL_PATHS = [TRAIN_FINAL_PATH, VAL_FINAL_PATH, TEST_FINAL_PATH]

IMAGE_FORMATS = ["jpeg", "jpg", "png"]

def __create_directory_structure():
    print("Adjusting directory structure...")
    try:
        os.mkdir("./src")
        os.mkdir("./dataset")
        os.mkdir("./log")
        os.mkdir("./output")
        os.mkdir("./utils")
    except:
        pass
    print("Directory structure OK!!!\n")

    print("Downloading training, validation and test sets...")
    for url, filepath in zip(URLS, SRC_FILES):
        urllib.request.urlretrieve(url, filepath)
    print("Files succesfully downloaded!!!\n")

def __save_images_and_labels():
    print("Organizing images and labels according to ImageGenerator input...")
    for zip_file, label_file, final_path in zip(ZIP_FILES_PATHS, LABELS_FILES, FINAL_PATHS):
        try:
            os.mkdir(final_path)
        except:
            pass
        zip_ref = zipfile.ZipFile(zip_file, 'r')
        zip_ref.extractall(final_path)
        try:
            df = pd.read_csv(label_file)
            labels = sorted(df.label.unique())
            for label in labels:
                try:
                    os.mkdir(final_path+f"/{label}")
                except:
                    pass
            id_to_label = df.set_index("ImageID").to_dict()["label"]
            images = [f for f in os.listdir(final_path) if f.split(".")[-1] in IMAGE_FORMATS]
            for image in images:
                label = id_to_label[int(image.split(".")[0])]
                shutil.move(os.path.join(final_path, image), os.path.join(final_path+f"/{label}", image))
        except:
            try:
                os.mkdir(f"{final_path}/test")
            except:
                pass
            images = [f for f in os.listdir(final_path) if f.split(".")[-1] in IMAGE_FORMATS]
            for image in images:
                shutil.move(os.path.join(final_path, image), os.path.join(final_path+"/test", image))
    print("Preprocessing completed!!!\n")

def get_and_preprocess_data():
    __create_directory_structure()
    __save_images_and_labels()