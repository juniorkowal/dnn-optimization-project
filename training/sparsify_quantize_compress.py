import torch
import torch.nn.utils.prune as prune
from .basic_funcs import load_model
import torch.nn as nn
import torch.optim as optim
import torch.quantization as quantization
import pickle
import numpy as np


def pruning(model: nn.Module, layers_to_prune: dict, pruning_method: str = 'unstructured', **kwargs):
    """
    Prune specified layers of the model according to the given dictionary.

    Args:
        model (nn.Module): The model to be pruned.
        layers_to_prune (dict): A dictionary where keys are layer names (strings) and values are the pruning amounts (floats).
        pruning_method (str): The method of pruning ('unstructured' or 'structured'). Default is 'unstructured'.
        **kwargs: Additional keyword arguments to pass to the pruning method (e.g., 'dim' for structured pruning).

    Returns:
        nn.Module: The pruned model.
    """

    for layer_name, amount in layers_to_prune.items():
        layer = dict(model.named_modules()).get(layer_name)

        if layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in the model.")

        if pruning_method == 'unstructured':
            prune.l1_unstructured(layer, name='weight', amount=amount)
        elif pruning_method == 'structured':
            prune.ln_structured(layer, name='weight', amount=amount, dim=kwargs.get('dim', 0), n=kwargs.get('n', 2))
        else:
            raise ValueError(f"Pruning method '{pruning_method}' is not recognized. Use 'unstructured' or 'structured'.")

        if kwargs.get('remove_pruning', False):
            prune.remove(layer, 'weight')

    return model


def quantize_model(model: nn.Module, dtype=torch.qint8):
    """
    Apply quantization to the model.

    Args:
        model (nn.Module): The model to be quantized.
        dtype (torch.dtype): The quantization data type. Default is torch.qint8.

    Returns:
        nn.Module: The quantized model.
    """
    model.eval()

    model.qconfig = quantization.default_qconfig
    quantization.prepare(model, inplace=True)

    with torch.no_grad():
        input_data = torch.randn(1, 3, 224, 224)
        model(input_data)

    quantized_model = quantization.convert(model, inplace=True)

    return quantized_model


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
