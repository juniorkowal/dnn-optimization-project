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
from rigl_torch.RigL import RigLScheduler
from typing import Callable
import os

from constants import DEVICE, MODELS_DIR


def save_model(model: nn.Module,
                optimizer: optim.Optimizer,
                scheduler: optim.lr_scheduler.LRScheduler,
                model_name: str,
                epochs: int,
                prefix: str = '',
                pruner: Callable = None
            ):
    model.to(DEVICE)
    model_dir = os.path.join(MODELS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_file_name = f"{prefix}{model_name}_epochs_{epochs}.pth"
    save_path = os.path.join(model_dir, model_file_name)
    torch.save({
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'pruner': pruner.state_dict(),
    }, save_path)


def train(model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        pruner: Callable = None):
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

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            loss.backward()

            if pruner():
                optimizer.step()
            print(pruner)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            pbar.set_postfix({'loss': running_loss / len_dataset,
                'acc': running_corrects.double().item() / len_dataset})

            pbar.update(1)

    train_loss = running_loss / len_dataset
    train_acc = running_corrects.double() / len_dataset

    return train_loss, train_acc


def rigl_training(baseline_model_path: str):
    model = models.resnet50(weights=None)

    epochs = 200
    optimizer = Prodigy(params=model.parameters(), lr=1.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    basic_funcs.load_model(model, optimizer, scheduler, model_path=baseline_model_path)
    print(f"Loaded model from: {baseline_model_path}")

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

    dataloader, _, _ = basic_funcs.get_dataloaders(data_dirs, transforms_)
    total_iterations = len(dataloader) * epochs
    T_end = int(0.75 * total_iterations)

    pruner = RigLScheduler(model,                           # model you created
                           optimizer,                       # optimizer (recommended = SGD w/ momentum)
                           dense_allocation=0.1,            # a float between 0 and 1 that designates how sparse you want the network to be
                                                              # (0.1 dense_allocation = 90% sparse)
                           sparsity_distribution='uniform', # distribution hyperparam within the paper, currently only supports `uniform`
                           T_end=T_end,                     # T_end hyperparam within the paper (recommended = 75% * total_iterations)
                           delta=100,                       # delta hyperparam within the paper (recommended = 100)
                           alpha=0.3,                       # alpha hyperparam within the paper (recommended = 0.3)
                           grad_accumulation_n=1,           # new hyperparam contribution (not in the paper)
                                                              # for more information, see the `Contributions Beyond the Paper` section
                           static_topo=False,               # if True, the topology will be frozen, in other words RigL will not do it's job
                                                              # (for debugging)
                           ignore_linear_layers=False,      # if True, linear layers in the network will be kept fully dense
                           state_dict=None)

    basic_funcs.train = lambda model, dataloader, criterion, optimizer: train(
            model, dataloader, criterion, optimizer, pruner
        ) # add pruner

    basic_funcs.save_model = lambda model, optimizer, scheduler, model_name, epochs, prefix: save_model(
            model, optimizer, scheduler, model_name, epochs, prefix, pruner
        )

    basic_funcs.train_loop(model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        transforms=transforms_,
        data_dirs = data_dirs,
        use_wandb=False,
        model_name='ResNet50_RIGL',
        epochs=epochs)
