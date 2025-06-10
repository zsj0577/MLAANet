import os
import time
import random
import numpy as np
import cv2
import torch
import shutil
import gdown
from zipfile import ZipFile
from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


""" Clearing the Directory """
def clear_directory(path):
    for elem_path in glob(path+"/*"):
        if os.path.isfile(elem_path):
            os.remove(elem_path)
        elif os.path.isdir(elem_path):
            shutil.rmtree(elem_path)
    print(path + " cleared !!")


def download_dataset(url, filepath):
    gdown.download(url, output=filepath, quiet=None)


def extract_dataset(filepath, unzip_path, remove=False):
    print("Extracting...")
    with ZipFile(file=filepath) as zip_file:
        for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
            zip_file.extract(member=file, path=unzip_path)
    print("File Unzipped Succesfully...")
    if remove:
        os.remove(filepath)
    print("Removed the Zipped File...")


def prepare_dataset(root):
    train_path = os.path.join(root, 'train')
    test_path = os.path.join(root, 'test')
    # Check if Directory Exists:
    print("Preparing Dataset")
    create_dir(root)
    print("Checking for Discrepancies in Dataset")
    # if not np.all([
    #     os.path.exists(train_path),
    #     os.path.exists(test_path),
    #     len(glob(os.path.join(train_path, 'imgs', '*'))) == 85,
    #     len(glob(os.path.join(train_path, 'masks', '*'))) == 85,
    #     len(glob(os.path.join(test_path, 'imgs', '*'))) == 80,
    #     len(glob(os.path.join(test_path, 'masks', '*'))) == 80,
    # ]):
    #     print("Dataset Incomplete Downlading it again !!!")
    #     filepath = os.path.join(os.getcwd(), root, 'GlasDataset.zip')
    #     clear_directory(root)
    #     download_dataset(
    #         'https://drive.google.com/u/0/uc?id=1HeVtHSrT-sWyFBiB2rx5tORMYSI4nwD0&confirm=1', filepath)
    #     extract_dataset(filepath, unzip_path=root, remove=True)


def plot(scores, name):
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(scores["train"])),
             scores["train"], label=f'train {name}')
    plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
    plt.title(f'{name} plot')
    plt.xlabel('Epoch')
    plt.ylabel(f'{name}')
    plt.legend()
    model_name = name.split(" ")[0]
    plt.savefig(f'results/model_pre_result/{model_name}_Files/{name}.png')
    # plt.show()


def combine_img_target_pred(rgb, gray, pred):
    rows_rgb, cols_rgb, channels = rgb.shape
    rows_gray, cols_gray, channels_gray = gray.shape
    rows_pred, cols_pred, channels_pred = gray.shape
    rows_comb = max(rows_rgb, rows_gray, rows_pred)
    cols_comb = cols_rgb + cols_gray + cols_pred
    comb = np.zeros(shape=(rows_comb, cols_comb, channels), dtype=np.uint8)
    comb[:rows_rgb,  0:cols_rgb] = rgb
    comb[:rows_gray, cols_rgb: cols_rgb*2] = np.atleast_3d(gray)
    comb[:rows_pred, cols_rgb*2:cols_rgb*3] = np.atleast_3d(pred)
    return comb
