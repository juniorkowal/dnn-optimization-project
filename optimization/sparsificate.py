from constants import DATASET_DIR
from training import basic_funcs
import torch
from torchvision import models
from prodigyopt import Prodigy
import torch.nn.utils.prune as prune
import torch.nn as nn
import os
from torchvision import transforms

from constants import MODELS_DIR, DEVICE, WORKERS


def prune_model(model_path: str, amount: float = 0.15):
    model_name = 'ResNet50_SPARSIFIED'
    model = models.resnet50(weights=None)

    num_classes = 32
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    optimizer = Prodigy(params=model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    basic_funcs.load_model(model, optimizer, scheduler, model_path=model_path)
    print(f"Loaded model from: {model_path}")

    # Przykład przerzedzania wybranych warstw konwolucyjnych i w pełni połączonych
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            # Przerzedzanie wag w 50% (można to dostosować)
            prune.l1_unstructured(module, name='weight', amount=amount)

    # Sprawdzenie, które warstwy zostały przerzedzone
    print("Przerzedzone warstwy:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            print(f"Layer: {name}, Pruned: {hasattr(module, 'weight_mask')}")

    # Usuwanie masek przerzedzających, aby model był finalnie przerzedzony
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')

    basic_funcs.save_model(model=model, optimizer=optimizer, scheduler=scheduler, model_name=model_name,
                           epochs=50, prefix='SPARSE_')

    # Count total and non-zero parameters
    total_params = sum(p.numel() for p in model.parameters())
    non_zero_params = sum((p != 0).sum().item() for p in model.parameters())

    # Calculate sparsity
    pruned_params = total_params - non_zero_params
    sparsity = pruned_params / total_params * 100

    content = (f"Total number of parameters: {total_params}" + '\n' +
               f"Number of non-zero parameters: {non_zero_params}" + '\n' +
               f"Number of pruned (zeroed) parameters: {pruned_params}" + '\n' +
               f"Sparsity: {sparsity:.2f}%" + '\n\n')

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

    _, _, test_loader = basic_funcs.get_dataloaders(data_dirs, transforms_)

    save_dir = os.path.join(MODELS_DIR, model_name)

    model.to(DEVICE)

    test_loss, test_acc = basic_funcs.validate(model, test_loader, criterion, desc='Testing on BEST model')
    content += f"Test Results:\n\nLoss: {test_loss}\nAccuracy: {test_acc}\nCriterion: {criterion.__class__.__name__}\n"
    results_path = os.path.join(save_dir, 'test_results_best.txt')
    with open(results_path, 'w') as file:
        file.write(content)