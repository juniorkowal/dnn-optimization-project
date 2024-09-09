from constants import DATASET_DIR, SET_SEED
from prepare_dataset import preprocess_all
from training.basic_funcs import train_loop
import torch
from torchvision import models, transforms
from prodigyopt import Prodigy
import torch.nn as nn
from training.train_baseline import baseline_training


if __name__ == "__main__":
    SET_SEED(0)
    # preprocess_all() # or download all
    baseline_training()
