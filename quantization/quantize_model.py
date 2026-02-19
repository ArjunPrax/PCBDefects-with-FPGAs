"""
8-bit Quantization for FPGA Deployment
Supports both Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model import create_model
from models.dataset import DefectDataset, get_transforms


class QuantizedConv2d(nn.Module):
    """Quantized Conv2d layer for QAT"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.zero_point = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        # Simulate quantization during training
        x_quantized = self.fake_quantize(x)
        out = self.conv(x_quantized)
        return out
    
    def fake_quantize(self, x):
        """Fake quantization (for training)"""
        x_scaled = x / self.scale
        x_rounded = torch.round(x_scaled + self.zero_point)
        x_clipped = torch.clamp(x_rounded, -128, 127)
        x_dequantized = (x_clipped - self.zero_point) * self.scale
        return x_dequantized


def quantize_tensor(tensor, num_bits=8):
    """
    Quantize a tensor to int8
    
    Args:
        tensor: Input tensor (weights or activations)
        num_bits: Number of bits (8 for int8)
    
    Returns:
        quantized_tensor: Quantized values (int8)
        scale: Scaling factor
        zero_point: Zero point
    """
    # Calculate min and max    
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        tensor = np.array(tensor)

    t_min = tensor.min()
    t_max = tensor.max()
    
    # Calculate scale and zero point
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    
    scale = (t_max - t_min) / (qmax - qmin)
    zero_point = qmin - t_min / scale
    
    # Quantize
    quantized = np.round(tensor / scale + zero_point)
    quantized = np.clip(quantized, qmin, qmax).astype(np.int8)
    
    return quantized, scale, zero_point


def dequantize_tensor(quantized_tensor, scale, zero_point):
    """Dequantize tensor back to float32"""
    return (quantized_tensor.astype(np.float32) - zero_point) * scale


def quantize_model(model, dataloader, device, num_batches=100):
    """
    Post-Training Quantization (PTQ)
    
    Calibrate with sample data and quantize all conv layers
    
    Args:
        model: Trained PyTorch model
        dataloader: Calibration data loader
        device: Device to run on
        num_batches: Number of calibration batches
    
    Returns:
        quantized_params: Dictionary of quantized parameters
    """
    model.eval()
    
    # Collect activation statistics
    activations = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            if isinstance(output, torch.Tensor):
                activations[name].append(output.detach().cpu().numpy())
            elif isinstance(output, np.ndarray):
                activations[name].append(output)
            else:
                activations[name].append(np.array(output))
        return hook
    
    # Register hooks on conv layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # Run calibration
    print("Running calibration...")
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(dataloader)):
            if i >= num_batches:
                break
            images = images.to(device)
            _ = model(images)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Quantize weights and collect activation ranges
    quantized_params = {}
    
    print("\nQuantizing parameters...")
    for name, module in tqdm(model.named_modules()):
        if isinstance(module, nn.Conv2d):
            # Quantize weights
            weight = module.weight.data
            weight_q, weight_scale, weight_zp = quantize_tensor(weight.cpu().numpy())
            
            # Quantize bias if exists
            if module.bias is not None:
                bias = module.bias.data
                bias_q, bias_scale, bias_zp = quantize_tensor(bias.cpu().numpy())
            else:
                bias_q, bias_scale, bias_zp = None, None, None
            
            # Get activation range
            if name in activations:
                act_tensor = torch.cat(activations[name], dim=0)
                act_q, act_scale, act_zp = quantize_tensor(act_tensor.cpu().numpy())
            else:
                act_scale, act_zp = 1.0, 0.0
            
            quantized_params[name] = {
                'weight': weight_q,
                'weight_scale': weight_scale,
                'weight_zero_point': weight_zp,
                'bias': bias_q,
                'bias_scale': bias_scale,
                'bias_zero_point': bias_zp,
                'activation_scale': act_scale,
                'activation_zero_point': act_zp,
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding
            }
    
    return quantized_params


def export_quantized_weights(quantized_params, output_dir):
    """
    Export quantized weights to .npy and .bin formats for FPGA
    
    Args:
        quantized_params: Quantized parameters dictionary
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nExporting quantized weights to {output_dir}...")
    
    for layer_name, params in quantized_params.items():
        # Clean layer name for filename
        clean_name = layer_name.replace('.', '_')
        
        # Save weights
        weight = params['weight']
        np.save(os.path.join(output_dir, f'{clean_name}_weight.npy'), weight)
        weight.tofile(os.path.join(output_dir, f'{clean_name}_weight.bin'))
        
        # Save bias
        if params['bias'] is not None:
            bias = params['bias']
            np.save(os.path.join(output_dir, f'{clean_name}_bias.npy'), bias)
            bias.tofile(os.path.join(output_dir, f'{clean_name}_bias.bin'))
        
        # Save metadata (scales and zero points)
        metadata = {
            'weight_scale': float(params['weight_scale']),
            'weight_zero_point': float(params['weight_zero_point']),
            'bias_scale': float(params['bias_scale']) if params['bias_scale'] is not None else None,
            'bias_zero_point': float(params['bias_zero_point']) if params['bias_zero_point'] is not None else None,
            'activation_scale': float(params['activation_scale']),
            'activation_zero_point': float(params['activation_zero_point']),
            'shape': {
                'in_channels': params['in_channels'],
                'out_channels': params['out_channels'],
                'kernel_size': params['kernel_size'],
                'stride': params['stride'],
                'padding': params['padding']
            }
        }
        
        import json
        with open(os.path.join(output_dir, f'{clean_name}_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Exported {len(quantized_params)} layers")


def evaluate_quantized_model(model, quantized_params, dataloader, device):
    """
    Evaluate accuracy drop due to quantization
    
    Args:
        model: Original model
        quantized_params: Quantized parameters
        dataloader: Test dataloader
        device: Device to run on
    
    Returns:
        accuracy: Accuracy with quantized inference
    """
    model.eval()
    
    # Replace conv layers with quantized versions
    # For simplicity, we'll measure accuracy drop by comparing with original
    correct = 0
    total = 0
    
    print("\nEvaluating quantized model...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # First, load dataset to get actual num_classes
    _, val_transform = get_transforms(96)  #Temporary
    temp_dataset = DefectDataset(
        root=args.dataset,
        split='train',
        transform=val_transform
    )
    
    # Get model configuration from dataset (more reliable than checkpoint)
    num_classes = temp_dataset.num_classes
    input_size = checkpoint.get('args', {}).get('input_size', 96)
    
    print(f"Detected {num_classes} classes from dataset")
    print(f"Input size: {input_size}")
    
    # Create model
    model = create_model(
        model_type=args.model_type,
        num_classes=num_classes,
        input_size=input_size
    )
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    
    # Load calibration data
    _, val_transform = get_transforms(input_size)
    
    calib_dataset = DefectDataset(
        root=args.dataset,
        split='train',  # Use training data for calibration
        transform=val_transform
    )
    
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Post-training quantization
    print("\n" + "=" * 60)
    print("POST-TRAINING QUANTIZATION")
    print("=" * 60)
    
    quantized_params = quantize_model(model, calib_loader, device, num_batches=args.calib_batches)
    
    # Export weights
    export_quantized_weights(quantized_params, args.output_dir)
    
    # Evaluate accuracy drop (simplified)
    test_dataset = DefectDataset(
        root=args.dataset,
        split='val',
        transform=val_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Original FP32 accuracy
    print("\nEvaluating original FP32 model...")
    fp32_acc = evaluate_quantized_model(model, None, test_loader, device)
    
    # Simulate INT8 quantized inference
    print("\nSimulating INT8 quantized inference...")
    # Create a copy with quantized weights loaded
    quant_model = create_model(
        model_type=args.model_type,
        num_classes=num_classes,
        input_size=input_size
    )
    
    # Load quantized and immediately dequantized weights (simulates FPGA behavior)
    with torch.no_grad():
        for name, module in quant_model.named_modules():
            if isinstance(module, nn.Conv2d) and name in quantized_params:
                qp = quantized_params[name]
                # Dequantize weights
                weight_dequant = dequantize_tensor(qp['weight'], qp['weight_scale'], qp['weight_zero_point'])
                module.weight.data = torch.from_numpy(weight_dequant).to(device)
                
                if module.bias is not None and qp['bias'] is not None:
                    bias_dequant = dequantize_tensor(qp['bias'], qp['bias_scale'], qp['bias_zero_point'])
                    module.bias.data = torch.from_numpy(bias_dequant).to(device)
    
    quant_model = quant_model.to(device)
    int8_acc = evaluate_quantized_model(quant_model, None, test_loader, device)
    
    accuracy_drop = fp32_acc - int8_acc
    
    print("\n" + "=" * 60)
    print("QUANTIZATION RESULTS")
    print("=" * 60)
    print(f"FP32 Accuracy: {fp32_acc:.2f}%")
    print(f"INT8 Simulated Accuracy: {int8_acc:.2f}%")
    print(f"Accuracy Drop: {accuracy_drop:.2f}%")
    print(f"\nQuantized weights exported to: {args.output_dir}")
    print(f"Layers quantized: {len(quantized_params)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantize model for FPGA')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='../datasets/deeppcb',
                        help='Path to dataset for calibration')
    parser.add_argument('--model-type', type=str, default='classifier',
                        choices=['classifier', 'detector'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--calib-batches', type=int, default=100,
                        help='Number of batches for calibration')
    parser.add_argument('--output-dir', type=str, default='quantized_weights',
                        help='Output directory for quantized weights')
    
    args = parser.parse_args()
    main(args)
