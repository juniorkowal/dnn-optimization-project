from constants import DATASET_DIR, MODELS_DIR, SET_SEED
from prepare_dataset import preprocess_all
from training.basic_funcs import train_loop
import torch
from torchvision import models, transforms
from prodigyopt import Prodigy
import torch.nn as nn
from training.train_baseline import baseline_training
from training.train_remapped import remapped_training


if __name__ == "__main__":
    SET_SEED(0)
    # preprocess_all() # or download all
    # baseline_training()
    remapped_training(f'{MODELS_DIR}/resnet_baseline/LAST_resnet_baseline_epochs_2.pth', noise_percentage=0.2) # training with remapped classes and noisy labels
