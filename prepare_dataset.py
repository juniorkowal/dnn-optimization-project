from genericpath import isfile
import gdown
import os
import zipfile
from constants import DATASET_DIR
import shutil
import json
from rembg import remove
from PIL import Image
import io
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
from tqdm.autonotebook import tqdm
from pathlib import Path


def download_and_extract_dataset():
    '''Use gdown to download dataset. Original source from kaggle:
        https://www.kaggle.com/datasets/wanderdust/coin-images/data'''
    url = 'https://drive.google.com/uc?export=download&id=10TA69iEb1X2-IEvo_Os68YRBsrwgV2kP'
    zip_file = 'coins.zip'
    extract_dir = DATASET_DIR

    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"'{extract_dir}' folder already exists and is not empty. Skipping download and extraction.")
    else:
        if not os.path.exists(zip_file):
            print(f"Downloading '{zip_file}'...")
            gdown.download(url, zip_file, quiet=False)
        else:
            print(f"'{zip_file}' already exists. Skipping download.")

        print(f"Unzipping '{zip_file}' to '{extract_dir}'...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        print(f"Extraction complete. Files are now available in the '{extract_dir}' folder.")
    os.remove(zip_file)


def dataset_to_32_classes():
    json_file = f"{DATASET_DIR}/coins/data/cat_to_name.json"
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)

    class_to_currency = {}
    for class_num, description in cat_to_name.items():
        currency = description.split(",")[1].strip()
        class_to_currency[class_num] = currency

    dir_pairs = [
        (f"{DATASET_DIR}/coins_cropped/validation",
            f"{DATASET_DIR}/coins_cropped_categorized_by_currency/validation"),
        (f"{DATASET_DIR}/coins_cropped/test",
            f"{DATASET_DIR}/coins_cropped_categorized_by_currency/test"),
        (f"{DATASET_DIR}/coins_cropped/train",
            f"{DATASET_DIR}/coins_cropped_categorized_by_currency/train")
    ]
    for src_dir, dest_dir in dir_pairs:
        os.makedirs(dest_dir, exist_ok=True)

        for class_folder in os.listdir(src_dir):
            class_path = os.path.join(src_dir, class_folder)
            if os.path.isdir(class_path):
                currency = class_to_currency.get(class_folder)
                if currency:
                    currency_dir = os.path.join(dest_dir, currency)
                    os.makedirs(currency_dir, exist_ok=True)
                    img_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
                    with tqdm(total=len(img_files), desc=f'Copying images for {class_folder}', unit='file') as pbar:
                        for img_file in os.listdir(class_path):
                            src_img_path = os.path.join(class_path, img_file)
                            dest_img_path = os.path.join(currency_dir, img_file)
                            shutil.copy(src_img_path, dest_img_path)
                            print(f"Copied {img_file} to {currency_dir}")
                            pbar.set_postfix({'Copied': img_file})
                            pbar.update(1)


def preprocess_images_in_folder(folder_path):
    '''Removing background from images and splitting image if there is more than 1 coin in the image.'''

    for split, _, files in os.walk(folder_path):
        image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Processing split: {split}")
        with tqdm(total=len(image_files), desc='Processing images', unit='file') as pbar:
            for file in image_files:
                file_path = os.path.join(split, file)
                process_image(file_path)
                pbar.set_postfix({'Processing': file})
                pbar.update(1)

def process_image(image_path):
    with open(image_path, 'rb') as input_file:
        image_data = input_file.read()
    result = remove(image_data)

    image_path_to_save = image_path.replace('coins/data', 'coins_cropped')
    dir_to_save = os.path.dirname(image_path_to_save)
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)

    img = Image.open(io.BytesIO(result)).convert("RGBA")
    cropped_img = img.crop(img.getbbox())

    width, height = cropped_img.size

    if width > 1.5 * height:
        half_width = width // 2

        left_half = cropped_img.crop((0, 0, half_width, height))
        left_half_path = image_path_to_save.replace('.png', '_left_half.png').replace('.jpg', '_left_half.png').replace('.jpeg', '_left_half.png')
        left_half.save(left_half_path)
        right_half = cropped_img.crop((half_width, 0, width, height))
        right_half_path = image_path_to_save.replace('.png', '_right_half.png').replace('.jpg', '_right_half.png').replace('.jpeg', '_right_half.png')
        right_half.save(right_half_path)
    else:
        cropped_img_path = image_path_to_save.replace('.png', '_cropped.png').replace('.jpg', '_cropped.png').replace('.jpeg', '_cropped.png')
        cropped_img.save(cropped_img_path)


def save_image(tensor_image, path):
    img = transforms.ToPILImage()(tensor_image)
    img.save(path)

def balance_dataset():
    train_dir = f'{DATASET_DIR}/coins_cropped/train'  # Path to training folder
    output_dir = f'{DATASET_DIR}/coins_cropped_aug/train'  # Where to save original and augmented images
    target_count = 120  # target number of images per class

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform_augment = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
    ])

    transform_original = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(train_dir, transform=None)
    class_names = dataset.classes

    for class_name in class_names:
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(train_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)

        class_images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        print(f"Processing class: {class_name}")
        with tqdm(total=len(class_images), desc=f'Processing {class_name}', unit='file') as pbar:
            for img_file in class_images:
                img_path = os.path.join(class_dir, img_file)
                img = Image.open(img_path)
                img_tensor = transform_original(img)
                save_image(img_tensor, os.path.join(output_class_dir, f'original_{img_file}'))
                pbar.update(1)

        current_count = len(class_images)
        if current_count < target_count:
            print(f"Augmenting images for class: {class_name}")
            with tqdm(total=target_count - current_count, desc=f'Augmenting {class_name}', unit='file') as pbar:
                while current_count < target_count:
                    random_img = random.choice(class_images)
                    img_path = os.path.join(class_dir, random_img)
                    img = Image.open(img_path)
                    img_aug = transform_augment(img)
                    aug_img_name = f'augmented_{current_count}_{random_img}'
                    save_image(img_aug, os.path.join(output_class_dir, aug_img_name))
                    current_count += 1
                    pbar.update(1)


def preprocess_all():
    download_and_extract_dataset() # just download
    folder_path = f'{DATASET_DIR}/coins/data'
    preprocess_images_in_folder(folder_path) # first preprocessing for baseline model
    dataset_to_32_classes() # mapping dataset to 32 classes coins -> currency
    balance_dataset() # augmenting dataset to have 120 images of each class


if __name__ == "__main__":
    download_and_extract_dataset() # just download
    folder_path = f'{DATASET_DIR}/coins/data'
    preprocess_images_in_folder(folder_path) # first preprocessing for baseline model
    dataset_to_32_classes() # mapping dataset to 32 classes coins -> currency
    balance_dataset() # augmenting dataset to have 120 images of each class
