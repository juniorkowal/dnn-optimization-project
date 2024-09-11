from torchvision.models import resnet50
import torch.quantization
from training.basic_funcs import load_model, save_model, get_dataloaders, validate
from tqdm import tqdm
import torch
from torchvision import models, transforms, datasets
from prodigyopt import Prodigy
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from constants import MODELS_DIR, DEVICE, WORKERS, DATASET_DIR


# Step 1: Define custom compression and decompression methods
def compress_model(model: nn.Module, num_bits: int = 8):
    """
    Compress the model using weight sharing (quantization-based).
    Args:
        model (nn.Module): The model to be compressed.
        num_bits (int): The number of bits to use for quantization.
    Returns:
        dict: A dictionary with compressed weights and the original scale factors.
    """
    compressed_model = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_data = param.detach().cpu().numpy()
            flat_param = param_data.flatten()
            min_val, max_val = flat_param.min(), flat_param.max()

            quantized_param = np.round((flat_param - min_val) / (max_val - min_val) * (2**num_bits - 1))
            quantized_param = quantized_param.astype(np.uint8)

            scale = (max_val - min_val) / (2**num_bits - 1)

            compressed_model[name] = {
                'quantized_param': quantized_param,
                'scale': scale,
                'min_val': min_val
            }
    return compressed_model

def decompress_model(model: nn.Module, compressed_model: dict):
    """
    Decompress the model from the compressed representation.
    Args:
        model (nn.Module): The original model architecture.
        compressed_model (dict): The compressed model dictionary.
    Returns:
        nn.Module: The decompressed model with original weights recovered.
    """
    for name, param in model.named_parameters():
        if name in compressed_model:
            quantized_param = compressed_model[name]['quantized_param']
            scale = compressed_model[name]['scale']
            min_val = compressed_model[name]['min_val']

            decompressed_param = (quantized_param.astype(np.float32) * scale) + min_val
            decompressed_param = torch.tensor(decompressed_param).reshape(param.shape)

            param.data.copy_(decompressed_param)
    return model

def model_compression(model_path):
    model = models.resnet50(weights=None)
    num_classes = 32
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    epochs = 50
    optimizer = Prodigy(params=model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    load_model(model, optimizer, scheduler, model_path=model_path)
    print(f"Loaded model from: {model_path}")

    # Compress the model using 8-bit quantization
    compressed_model = compress_model(model, num_bits=8)

    # Step 3: Save the compressed model
    model_name = 'ResNet50_COMPRESSED'
    os.makedirs('trained_models/ResNet50_COMPRESSED/')
    torch.save(compressed_model, 'trained_models/ResNet50_COMPRESSED/COMPRESS_ResNet50.pth')

    # Step 4: Load and decompress the model
    loaded_compressed_model = torch.load('trained_models/ResNet50_COMPRESSED/COMPRESS_ResNet50.pth')
    # Create a new instance of ResNet-50
    decompressed_model = models.resnet50(weights=None)
    decompressed_model.fc = nn.Linear(decompressed_model.fc.in_features, num_classes)
    decompressed_model.to(DEVICE)

    load_model(decompressed_model, optimizer, scheduler, model_path=model_path)
    decompressed_model.eval()

    # Decompress the model weights
    decompressed_model = decompress_model(decompressed_model, loaded_compressed_model)

    criterion = nn.CrossEntropyLoss()
    data_dirs = (
        f'{DATASET_DIR}/coins_cropped_categorized_by_currency/train',
        f'{DATASET_DIR}/coins_cropped_categorized_by_currency/validation',
        f'{DATASET_DIR}/coins_cropped_categorized_by_currency/test',
    )
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transforms_ = (transform, transform, transform)
    _, _, test_loader = get_dataloaders(data_dirs, transforms_)

    save_dir = os.path.join(MODELS_DIR, model_name)

    test_loss, test_acc = validate(decompressed_model, test_loader, criterion, desc='Testing on BEST model')
    content = f"Test Results:\n\nLoss: {test_loss}\nAccuracy: {test_acc}\nCriterion: {criterion.__class__.__name__}\n"
    results_path = os.path.join(save_dir, 'test_results_best.txt')
    with open(results_path, 'w') as file:
        file.write(content)
