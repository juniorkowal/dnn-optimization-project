from constants import DATASET_DIR
from .basic_funcs import train_loop
import torch
from torchvision import models, transforms
from prodigyopt import Prodigy
import torch.nn as nn


def baseline_training():
    model = models.resnet50(weights=None)
    num_classes = 211
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    epochs = 200
    optimizer = Prodigy(params=model.parameters(), lr=1.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet input
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    transforms_ = (transform, transform, transform)
    data_dirs = (
        f'{DATASET_DIR}/coins/data/train',
        f'{DATASET_DIR}/coins/data/validation',
        f'{DATASET_DIR}/coins/data/test',
    )
    train_loop(model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        transforms=transforms_,
        data_dirs = data_dirs,
        use_wandb=False,
        model_name='ResNet50_BASELINE',
        epochs=epochs)
