from constants import DATASET_DIR
from . import basic_funcs
import torch
from torchvision import models, transforms
from prodigyopt import Prodigy
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import torch.optim as optim

from constants import DEVICE


def add_noise_to_labels(labels, noise_percentage, num_classes):
    """
    Adds noise to the labels by randomly changing them based on a specified percentage.

    Args:
        labels (torch.Tensor): Original labels.
        noise_percentage (float): Percentage of labels to be noisy.
        num_classes (int): Number of classes in the classification problem.

    Returns:
        torch.Tensor: Labels with added noise.
    """
    labels_noisy = labels.clone()
    num_labels = len(labels)
    num_noisy_labels = int(num_labels * noise_percentage)
    noisy_indices = np.random.choice(num_labels, num_noisy_labels, replace=False)

    for idx in noisy_indices:
        current_label = labels[idx].item()
        noisy_label = current_label

        while noisy_label == current_label: # so that we don't pick original class
            noisy_label = np.random.randint(0, num_classes)

        labels_noisy[idx] = noisy_label

    return labels_noisy


def train(model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        noise_percentage: float = 0.1,
        num_classes: int = 32):
    model.to(DEVICE)
    model.train()
    running_loss = 0.0
    running_corrects = 0
    len_dataset = len(dataloader.dataset)
    total_noisy_labels = 0
    total_labels = 0

    with tqdm(total=len(dataloader), desc='Training', unit='batch', leave=False) as pbar:
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            labels_noisy = add_noise_to_labels(labels, noise_percentage, num_classes)

            # num_noisy_labels = torch.sum(labels != labels_noisy).item()
            # total_noisy_labels += num_noisy_labels
            # total_labels += len(labels)

            # if num_noisy_labels > 0:
            #     print(f"Batch: {pbar.n}/{len(dataloader)} - Noisy labels: {num_noisy_labels}/{len(labels)} ({num_noisy_labels / len(labels) * 100:.2f}%)")

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels_noisy)  # use noisy labels

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels_noisy.data)

            pbar.set_postfix({'loss': running_loss / len_dataset,
                'acc': running_corrects.double().item() / len_dataset})

            pbar.update(1)

    train_loss = running_loss / len_dataset
    train_acc = running_corrects.double() / len_dataset

    # print(f"Total noisy labels: {total_noisy_labels}/{total_labels} ({total_noisy_labels / total_labels * 100:.2f}%)")

    return train_loss, train_acc


def remapped_training(baseline_model_path: str, noise_percentage: float = 0.1, epochs: int = 50):
    model = models.resnet50(weights=None)

    num_classes = 211
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(DEVICE)

    optimizer = Prodigy(params=model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    basic_funcs.load_model(model, optimizer, scheduler, model_path=baseline_model_path)
    print(f"Loaded model from: {baseline_model_path}")

    for i, param in enumerate(model.parameters()):
        if i < 143:
            param.requires_grad = False

    num_classes = 32
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transforms_ = (transform, transform, transform)
    folder = 'coins_cropped_categorized_by_currency'
    data_dirs = (
        f'{DATASET_DIR}/{folder}/train',
        f'{DATASET_DIR}/{folder}/validation',
        f'{DATASET_DIR}/{folder}/test',
    )

    basic_funcs.train = lambda model, dataloader, criterion, optimizer: train(
            model, dataloader, criterion, optimizer, noise_percentage, num_classes
        ) # replace with noisy train function

    basic_funcs.train_loop(model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        transforms=transforms_,
        data_dirs = data_dirs,
        use_wandb=False,
        model_name='ResNet50_REMAPPED_CLASSES',
        epochs=epochs)
