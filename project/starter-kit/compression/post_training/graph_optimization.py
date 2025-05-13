"""
Graph optimization utilities for PyTorch models.
Supports TorchScript and TorchFX optimizations with unified evaluation pipeline.
"""
import os
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.experimental.optimization import (
    fuse,
    remove_dropout, 
    optimize_for_inference
)
from typing import Dict, Any, Optional, Tuple, Literal, List, Union, Callable
import warnings
import json
import time

def optimize_model(
    model: nn.Module,
    optimization_method: Literal["torchscript", "torch_fx"] = "torchscript",
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: torch.device = torch.device('cuda'),
    custom_options: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Optimize a model using PyTorch graph optimization techniques.
    
    Args:
        model: Model to optimize
        optimization_method: Optimization method to use
        input_shape: Input shape for tracing (e.g., (1, 3, 224, 224) for a single RGB image)
        device: Device to set the input and model on
        custom_options: Custom optimization options (optional)
        
    Returns:
        Optimized model
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create a sample input tensor for tracing
    dummy_input = torch.randn(input_shape).to(device)
    model = model.to(device)
    
    # Apply optimization based on method
    if optimization_method == "torchscript":
        optimized_model = _optimize_with_torchscript(model, dummy_input, custom_options)    
    elif optimization_method == "torch_fx":
        optimized_model = _optimize_with_torch_fx(model, dummy_input, custom_options)
    else:
        raise ValueError(f"Unsupported optimization method: {optimization_method}")
        
    return optimized_model


# TODO: Implement the graph optimization techniques of your choice
# Use built-in torchscript functionalities
def _optimize_with_torchscript(
    model: nn.Module,
    dummy_input: torch.Tensor,
    custom_options: Optional[Dict[str, Any]] = None
) -> torch.jit.ScriptModule:
    """
    Optimize model with TorchScript (JIT).
    
    Args:
        model: Model to optimize
        dummy_input: Sample input for tracing
        custom_options: Custom optimization options
        
    Returns:
        TorchScript optimized model
    """
    # Extract custom options with defaults
    if custom_options is None:
        custom_options = {}
    
    pass


# TODO: Implement the graph optimization techniques of your choice
# Use built-in torch fx functionalities
def _optimize_with_torch_fx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    custom_options: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Optimize model with Torch FX.
    
    Args:
        model: Model to optimize
        dummy_input: Sample input for tracing
        custom_options: Custom optimization options
        device: str
        
    Returns:
        FX-optimized model
    """
    # Extract custom options with defaults
    if custom_options is None:
        custom_options = {}
    
    pass


def verify_model_equivalence(
    original_model: nn.Module,
    optimized_model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: torch.device = torch.device('cpu'),
    rtol: float = 1e-3,
    atol: float = 1e-3
) -> bool:
    """
    Verify equivalence between original and optimized models.
    
    Args:
        original_model: Original PyTorch model
        optimized_model: Optimized model
        input_shape: Shape of input tensor
        device: Device to run verification on
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        
    Returns:
        True if models are equivalent, False otherwise
    """
    # Set models to evaluation mode
    original_model.eval()
    optimized_model.eval()
    
    # Create a random input tensor
    torch.manual_seed(0)  # For reproducibility
    input_tensor = torch.randn(input_shape, device=device)
    
    # Run inference with both models
    with torch.no_grad():
        original_output = original_model(input_tensor)
        optimized_output = optimized_model(input_tensor)
    
    # Compare outputs
    if isinstance(original_output, tuple):
        original_output = original_output[0]  # Use first output if model returns multiple
    
    if isinstance(optimized_output, tuple):
        optimized_output = optimized_output[0]
    
    # Check if the outputs are close
    is_close = torch.allclose(original_output, optimized_output, rtol=rtol, atol=atol)
    
    if is_close:
        print("Original and optimized models produce equivalent outputs.")
    else:
        max_diff = torch.max(torch.abs(original_output - optimized_output))
        mean_diff = torch.mean(torch.abs(original_output - optimized_output))
        print(f"Models differ. Max difference: {max_diff.item():.6f}, Mean difference: {mean_diff.item():.6f}")
    
    return is_close