import torch
import torch.nn as nn
import torch.optim as optim
import os
import re
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from ptflops import get_model_complexity_info
from tqdm.autonotebook import tqdm
import wandb
from torchinfo import summary
from typing import Tuple
import matplotlib.pyplot as plt
import glob

from constants import MODELS_DIR, DEVICE, WORKERS


def save_model(model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler, model_name: str, epochs: int, prefix: str = ''):
    model_dir = os.path.join(MODELS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_file_name = f"{prefix}{model_name}_epochs_{epochs}.pth"
    save_path = os.path.join(model_dir, model_file_name)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }, save_path)

def load_model(model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler, model_path: str):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

def load_last_model(model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler, model_name: str):
    model_folder = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_folder):
        model_files = [f for f in os.listdir(model_folder) if f.endswith(".pth")]
        if model_files:
            model_files.sort(key=lambda x: int(re.findall(r"epochs_(\d+)", x)[0])) # Sort by epochs in filename
            last_model_file = model_files[-1] # Load the model with the most epochs
            last_epochs = int(re.findall(r"epochs_(\d+)", last_model_file)[0])
            model_path = os.path.join(MODELS_DIR, model_name, last_model_file)
            load_model(model, optimizer, scheduler, model_path)
            print(f"Loaded model with {last_epochs} epochs.")
            return last_epochs
    return 0

def delete_last_best_model(model_name: str):
    """Deletes the last saved best model file that contains 'BEST' in its name."""
    model_dir = os.path.join(MODELS_DIR, model_name)
    for file in os.listdir(model_dir):
        if 'BEST' in file:
            epoch_num = int(file.split('.')[0].split('_')[-1])
            os.remove(os.path.join(model_dir, file))

def get_dataloaders(dirs, transforms):
    train_dir, valid_dir, test_dir = dirs
    train_transform, val_transform, test_transform = transforms
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=WORKERS)
    val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=WORKERS)
    return train_loader, val_loader, test_loader

def save_gflops(model_name, model, input_shape):
    model_dir = os.path.join(MODELS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    gflops_file_name = "gflops.txt"
    save_path = os.path.join(model_dir, gflops_file_name)

    with open(save_path, "w") as file:
        macs, params = get_model_complexity_info(
            model,
            input_shape,
            as_strings=True,
            backend='pytorch',
            print_per_layer_stat=True,
            verbose=True,
            ost=file
        )
    return macs, params

def plot_results(train_losses, val_losses, train_accs, val_accs, save_dir, show = False):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    if show:
        plt.show()
    save_path = os.path.join(save_dir, 'training_results.png')
    plt.savefig(save_path)
    plt.close()


def train(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    len_dataset = len(dataloader.dataset)

    with tqdm(total=len(dataloader), desc='Training', unit='batch', leave=False) as pbar:
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            pbar.set_postfix({'loss': running_loss / len_dataset,
                'acc': running_corrects.double().item() / len_dataset})

            pbar.update(1)
    train_loss = running_loss / len_dataset
    train_acc = running_corrects.double() / len_dataset
    return train_loss, train_acc


@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, desc: str = 'Validation'):
    model.eval()
    valid_loss = 0.0
    valid_corrects = 0
    len_dataset = len(dataloader.dataset)

    with tqdm(total=len(dataloader), desc='Validation', unit='batch', leave=False) as pbar:
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            valid_loss += loss.item() * inputs.size(0)
            valid_corrects += torch.sum(preds == labels.data)

            pbar.update(1)
    valid_loss = valid_loss / len_dataset
    valid_acc = valid_corrects.double() / len_dataset
    return valid_loss, valid_acc



def train_loop(model: nn.Module,
                optimizer: optim.Optimizer,
                scheduler: optim.lr_scheduler.LRScheduler,
                transforms: Tuple,
                data_dirs:Tuple,
                use_wandb: bool = False,
                model_name: str = 'ResNet50',
                epochs: int = 200,):
    macs, params = save_gflops(model_name, model, input_shape=(3, 224, 224))

    config = {}

    config['macs'] = '{:<30}  {:<8}'.format('Computational complexity: ', macs)
    config['params'] = '{:<30}  {:<8}'.format('Number of parameters: ', params)
    config['optimizer'] = optimizer.__class__.__name__
    config['architecture']: model.__class__.__name__
    config['scheduler'] = scheduler.__class__.__name__
    config['scheduler_params'] = scheduler.state_dict()

    if use_wandb:
        wandb.init(project = 'osiowsn', name = model_name, group='studies', config = config)

    summary(model=model, input_size=(1, 3, 224, 224)) # B C H W
    model.to(DEVICE)

    train_loader, val_loader, test_loader = get_dataloaders(data_dirs, transforms)
    last_epoch = load_last_model(model, optimizer, scheduler, model_name)

    best_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    with tqdm(range(last_epoch, epochs), desc='Epochs', initial=last_epoch, total=epochs) as pbar:
        for epoch in pbar:
            save_best_model = False

            train_loss, train_acc = train(model, train_loader, criterion, optimizer)
            val_loss, val_acc = validate(model, val_loader, criterion)

            if val_acc > best_acc:
                best_acc = val_acc
                delete_last_best_model(model_name=model_name)
                save_model(model=model, optimizer=optimizer, scheduler=scheduler, model_name=model_name, epochs=epoch, prefix='BEST_')
                save_best_model = True

            scheduler.step()

            formatted_metrics = {
                "train/accuracy": train_acc,
                "train/loss": train_loss,
                "validate/accuracy": val_acc,
                "validate/loss": val_loss,
            }
            pbar.set_postfix_str("; ".join(f"{key}: {value:.4f}" for key, value in formatted_metrics.items()))
            torch.cuda.empty_cache()

            train_losses.append(train_loss); train_accs.append(train_acc.item())
            val_losses.append(val_loss); val_accs.append(val_acc.item())
            wandb.log(formatted_metrics, step=epoch) if use_wandb else None
    save_model(model=model, optimizer=optimizer, scheduler=scheduler, model_name=model_name, epochs=epochs, prefix='LAST_')
    wandb.finish() if use_wandb else None

    save_dir = os.path.join(MODELS_DIR, model_name)
    plot_results(train_losses, val_losses, train_accs, val_accs, save_dir)

    test_loss, test_acc = validate(model, test_loader, criterion, desc='Testing on LAST model')
    criterion_name = criterion.__class__.__name__
    content = f"Test Results:\n\nLoss: {test_loss}\nAccuracy: {test_acc}\nCriterion: {criterion_name}\n"
    results_path = os.path.join(save_dir, 'test_results_last.txt')
    with open(results_path, 'w') as file:
        file.write(content)

    best_model_files = glob.glob(os.path.join(save_dir, '*BEST*'))
    best_model_path = ''
    if best_model_files:
        best_model_path = best_model_files[0]
    load_model(model, optimizer, scheduler, best_model_path)
    test_loss, test_acc = validate(model, test_loader, criterion, desc='Testing on BEST model')
    content = f"Test Results:\n\nLoss: {test_loss}\nAccuracy: {test_acc}\nCriterion: {criterion_name}\n"
    results_path = os.path.join(save_dir, 'test_results_best.txt')
    with open(results_path, 'w') as file:
        file.write(content)
