"""
UdaciSense Project: Evaluation Utilities Module

This module provides functions for evaluating model performance, optimizations, 
and comparing models against baselines and requirements.
"""

import json
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

from utils import MAX_ALLOWED_ACCURACY_DROP, TARGET_INFERENCE_SPEEDUP,TARGET_MODEL_COMPRESSION 
from utils.evaluation import (evaluate_model_metrics, compare_models, 
                             evaluate_requirements_met, calculate_confusion_matrix)
from utils.model import count_parameters, MobileNetV3_Household
from utils.visualization import (plot_confusion_matrix, plot_training_history, 
                                plot_weight_distribution, plot_model_comparison)


# ------ EXPERIMENTATION UTILITIES ------ #

def evaluate_optimized_model(
    optimized_model: nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    technique_name: str, 
    class_names: List[str],
    input_size: Tuple[int, ...],
    is_in_training_technique: bool = False, 
    training_stats: Optional[Dict[str, Any]] = None, 
    device: torch.device = torch.device('cuda')
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Evaluate an optimized model and generate metrics and visualizations.
    
    Args:
        optimized_model: The optimized model to evaluate
        data_loader: DataLoader for test/validation data
        technique_name: Name of the optimization technique
        class_names: List of class names
        input_size: Input tensor shape for the model
        is_in_training_technique: Whether the technique involves training
        training_stats: Training statistics (if applicable)
        device: Device to run evaluation on
        
    Returns:
        Tuple of (metrics, confusion_matrix)
    """
    print(f"\nEvaluating performance of optimized model...")
    n_classes = len(class_names)

    # Evaluate model metrics
    metrics = evaluate_model_metrics(
        optimized_model, 
        data_loader, 
        device,
        n_classes,
        class_names,
        input_size,
        save_path=f"../results/{technique_name}/metrics.json"
    )
    
    # Calculate and save confusion matrix
    confusion_matrix = calculate_confusion_matrix(
        optimized_model, data_loader, device, n_classes)
    _ = plot_confusion_matrix(
        confusion_matrix, class_names, f"../results/{technique_name}/confusion_matrix.png")
    
    # Generate additional visualizations for training techniques
    if is_in_training_technique:
        # Plot weight distribution
        _ = plot_weight_distribution(
            optimized_model, output_path=f"../results/{technique_name}/weight_distribution.png")

        if training_stats:
            # Plot training history
            _ = plot_training_history(
                training_stats, f"../results/{technique_name}/training_history.png")
    
    # Print summary metrics
    print(f"\nOptimized Model Metrics ({technique_name}):")
    print(f"Accuracy: {metrics['accuracy']['top1_acc']:.2f}%")
    print(f"Model Size: {metrics['size']['model_size_mb']:.2f} MB")
    
    # Print inference time for both CPU and CUDA (if available)
    if metrics['timing'].get('cpu'):
        print(f"CPU Inference Time: {metrics['timing']['cpu']['avg_time_ms']:.2f} ms ({metrics['timing']['cpu']['fps']:.2f} FPS)")
    if metrics['timing'].get('cuda'):
        print(f"CUDA Inference Time: {metrics['timing']['cuda']['avg_time_ms']:.2f} ms ({metrics['timing']['cuda']['fps']:.2f} FPS)")
    
    return metrics, confusion_matrix
    

def compare_optimized_model_to_baseline(
    baseline_model: nn.Module,
    optimized_model: nn.Module, 
    technique_name: str, 
    data_loader: torch.utils.data.DataLoader, 
    class_names: List[str],
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    device: torch.device = torch.device('cuda'),
    device_priority: str = 'cpu'  # Default priority for speed requirements
) -> Dict[str, Any]:
    """Evaluate an optimized model and compare it to the baseline.
    
    Args:
        baseline_model: The baseline model for comparison
        optimized_model: The optimized model to compare
        technique_name: Name of the technique for display and for saving results
        data_loader: DataLoader for evaluation
        class_names: List of class names
        input_size: Input size for inference time measurement
        device: Device to use for accuracy evaluation
        device_priority: Which device to prioritize for speed requirements ('cpu' or 'cuda')
    
    Returns:
        Dictionary with comparison results
    """
    print(f"\nComparing performance of optimized model against baseline...")
    
    # Compare models
    comparison = compare_models(
        baseline_model=baseline_model,
        optimized_model=optimized_model,
        dataloader=data_loader,
        device=device,
        num_classes=len(class_names),
        class_names=class_names,
        input_size=input_size
    )
    
    # Save comparison results
    with open(f"../results/{technique_name}/comparison.json", 'w') as f:
        # Convert numpy values to native Python types for JSON serialization
        json_safe_comparison = json.loads(
            json.dumps(comparison, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x))
        json.dump(json_safe_comparison, f, indent=4)
    
    # Plot comparison
    _ = plot_model_comparison(
        comparison, output_path=f"../results/{technique_name}/comparison.png")
    
    # Check if requirements are met
    requirements_met = evaluate_requirements_met(
        comparison,
        size_requirement=TARGET_MODEL_COMPRESSION,
        speed_requirement=TARGET_INFERENCE_SPEEDUP,
        accuracy_requirement=MAX_ALLOWED_ACCURACY_DROP,
        device_priority=device_priority  # Use the specified device priority
    )
    
    # Save requirements check
    with open(f"../results/{technique_name}/requirements_met.json", 'w') as f:
        json.dump(requirements_met, f, indent=4)
    
    # Display comparison
    print(f"\n{technique_name} Results:")
    print(f"Model Size: {comparison['optimized']['size']['model_size_mb']:.2f} MB " 
          f"({comparison['improvements']['size_reduction']:.1%} reduction)")
    print(f"Parameters: {comparison['optimized']['size']['trainable_params']:,} "
          f"({comparison['improvements']['params_reduction']:.1%} reduction)")
    
    # Display CPU inference time improvements (if available)
    if comparison['optimized']['timing'].get('cpu') and comparison['improvements']['inference'].get('cpu'):
        print(f"CPU Inference Time: {comparison['optimized']['timing']['cpu']['avg_time_ms']:.2f} ms "
              f"({comparison['improvements']['inference']['cpu']['speedup']:.1f}x speedup)")
    
    # Display CUDA inference time improvements (if available)
    if comparison['optimized']['timing'].get('cuda') and comparison['improvements']['inference'].get('cuda'):
        print(f"CUDA Inference Time: {comparison['optimized']['timing']['cuda']['avg_time_ms']:.2f} ms "
              f"({comparison['improvements']['inference']['cuda']['speedup']:.1f}x speedup)")
    
    print(f"Accuracy: {comparison['optimized']['accuracy']['top1_acc']:.2f}% "
          f"({comparison['improvements']['accuracy_change']*100:+.2f}% change)")
    print(f"Requirements met: {requirements_met['all_requirements_met']}")
    
    # Create a dictionary with key results for later comparison
    result = {
        'name': technique_name,
        'model_size_mb': comparison['optimized']['size']['model_size_mb'],
        'total_params': comparison['optimized']['size']['trainable_params'],
        'accuracy': comparison['optimized']['accuracy']['top1_acc'],
        'size_reduction': comparison['improvements']['size_reduction'],
        'params_reduction': comparison['improvements']['params_reduction'],
        'accuracy_change': comparison['improvements']['accuracy_change'],
        'requirements_met': requirements_met['all_requirements_met']
    }
    
    # Add inference time metrics for both CPU and CUDA
    if comparison['optimized']['timing'].get('cpu') and comparison['improvements']['inference'].get('cpu'):
        result['cpu_inference_time_ms'] = comparison['optimized']['timing']['cpu']['avg_time_ms']
        result['cpu_inference_speedup'] = comparison['improvements']['inference']['cpu']['speedup']
    
    if comparison['optimized']['timing'].get('cuda') and comparison['improvements']['inference'].get('cuda'):
        result['cuda_inference_time_ms'] = comparison['optimized']['timing']['cuda']['avg_time_ms']
        result['cuda_inference_speedup'] = comparison['improvements']['inference']['cuda']['speedup']
    else:
        result['cuda_inference_time_ms'] = None
        result['cuda_inference_speedup'] = None
    
    return result

def compare_experiments(
    experiments=None,
    compression_results=None,
    baseline_metrics=None,
    baseline_name=None,
    results_dir="../results",
    accuracy_drop_threshold=MAX_ALLOWED_ACCURACY_DROP, 
    size_reduction_target=TARGET_MODEL_COMPRESSION,
    inference_reduction_target=TARGET_INFERENCE_SPEEDUP,
    device_priority="cpu"
):
    """
    Compare multiple model optimization experiments with specific targets for size, speed, and accuracy.

    This version replaces internal checks with `evaluate_requirements_met()`.

    Args:
        experiments: List of experiment names to load from disk (mutually exclusive with compression_results)
        compression_results: Dictionary of pre-loaded compression results (mutually exclusive with experiments)
        baseline_metrics: Dictionary of baseline model metrics (optional if baseline_name is provided)
        baseline_name: Name of the baseline experiment (to load from disk if baseline_metrics not provided)
        results_dir: Base directory containing experiment results
        accuracy_drop_threshold: Maximum allowed accuracy drop (relative, e.g., 0.05 for 5%)
        size_reduction_target: Target for size reduction (0.0 to 1.0), default 0.6 (60% reduction)
        inference_reduction_target: Target factor to reduce inference by (0.0 to 1.0), default 0.5 (50% reduction)
        device_priority: Which device to prioritize for inference time ('cuda' or 'cpu')

    Returns:
        DataFrame containing comparison results
    """
    def load_metrics(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metrics from {path}: {e}")
            return None

    # Create output directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Validate inputs
    if experiments is not None and compression_results is not None:
        raise ValueError("Provide either 'experiments' or 'compression_results', not both")

    if experiments is None and compression_results is None:
        raise ValueError("Either 'experiments' or 'compression_results' must be provided")

    # Load experiment results if needed
    if experiments is not None:
        compression_results = {
            exp_name: load_metrics(os.path.join(results_dir, exp_name, "metrics.json"))
            for exp_name in experiments
        }
        compression_results = {k: v for k, v in compression_results.items() if v is not None}

    # Load baseline metrics if needed
    if baseline_metrics is None and baseline_name is not None:
        baseline_path = os.path.join(results_dir, baseline_name, "metrics.json")
        baseline_metrics = load_metrics(baseline_path)

    if baseline_metrics is None:
        raise ValueError("Either `baseline_metrics` or a valid `baseline_name` must be provided")

    # Extract baseline values
    baseline_size = baseline_metrics['size']['model_size_mb']
    baseline_params = baseline_metrics['size']['total_params']
    baseline_acc = baseline_metrics['accuracy']['top1_acc']
    baseline_time = baseline_metrics['timing'].get(device_priority, {}).get('avg_time_ms', 0)

    # Compute required speedup target
    speed_target = 1 / (1 - inference_reduction_target)

    # Process results
    results_table = []

    for key, result in compression_results.items():
        if not isinstance(result, dict):
            print(f"Skipping {key}: Not a valid result dictionary")
            continue

        # Extract metrics
        model_size_mb = result.get('size', {}).get('model_size_mb', 0)
        total_params = result.get('size', {}).get('total_params', 0)
        accuracy = result.get('accuracy', {}).get('top1_acc', 0)
        inference_time_ms = result.get('timing', {}).get(device_priority, {}).get('avg_time_ms', 0)

        # Compute reductions and speedup
        size_reduction = 1.0 - (model_size_mb / baseline_size) if baseline_size > 0 else 0
        params_reduction = 1.0 - (total_params / baseline_params) if baseline_params > 0 else 0
        inference_speedup = baseline_time / inference_time_ms if baseline_time > 0 and inference_time_ms > 0 else 0
        accuracy_change = (accuracy - baseline_acc) / baseline_acc  # Relative accuracy change

        # Use evaluate_requirements_met() to check conditions
        comparison_data = {
            'improvements': {
                'size_reduction': size_reduction,
                'params_reduction': params_reduction,
                'inference': {
                    'cpu': {'reduction': 1 - (inference_time_ms / baseline_time)} if device_priority == 'cpu' else {},
                    'cuda': {'reduction': 1 - (inference_time_ms / baseline_time)} if device_priority == 'cuda' else {}
                },
                'accuracy_change': accuracy_change  # Relative accuracy drop
            }
        }

        eval_results = evaluate_requirements_met(
            comparison_results=comparison_data,
            size_requirement=size_reduction_target,
            speed_requirement=inference_reduction_target,
            accuracy_requirement=accuracy_drop_threshold,
            device_priority=device_priority
        )

        # Format results
        results_table.append({
            'Technique': result.get('name', key),
            'Model Size (MB)': f"{model_size_mb:.2f}",
            'Size Reduction': f"{size_reduction*100:.1f}%",
            'Inference Time (ms)': f"{inference_time_ms:.2f}",
            'Speedup': f"{inference_speedup:.1f}x",
            'Accuracy (%)': f"{accuracy:.2f}",
            'Acc. Change': f"{accuracy_change*100:+.1f}%",
            'All Reqs Met': "✓" if eval_results['all_requirements_met'] else "✗"
        })

    # Convert to DataFrame and display
    results_df = pd.DataFrame(results_table)
    print("\nComparison of All Compression Techniques:\n")
    display(results_df)

    return results_df

def list_experiments(results_dir="../results", metrics_filename="metrics.json", check_metrics=True):
    """
    List all experiment folders in the results directory, including subfolders.
    
    Args:
        results_dir: Path to the results directory containing experiment folders
        metrics_filename: Name of the metrics file to check for (default: "metrics.json")
        check_metrics: Whether to only return folders containing metrics files
        
    Returns:
        List of experiment paths relative to results_dir
    """
    experiments = []
    
    # Check if the results directory exists
    if not os.path.exists(results_dir):
        print(f"Warning: Results directory {results_dir} does not exist")
        return experiments
    
    # Use os.walk to iterate through all subdirectories
    for root, dirs, files in os.walk(results_dir):
        # Check if this directory contains the metrics file
        if not check_metrics or metrics_filename in files:
            # Get the path relative to results_dir
            if root != results_dir:
                rel_path = os.path.relpath(root, results_dir)
                experiments.append(rel_path)
    
    # Sort alphabetically
    experiments.sort()
    
    return experiments

# ------ QUANTIZATION UTILITIES ------ #

def is_quantized(model: nn.Module, silent: Optional[bool]=False) -> bool:
    """Check if a model is quantized.
    
    Args:
        model: PyTorch model to check
        silent: Whether printing should happen or not
        
    Returns:
        Boolean indicating whether the model is quantized
    """
    # Check for evidence of actual quantization
    for module in model.modules():
        module_name = module.__class__.__name__
        
        # Check for quantized module names (definitive sign)
        if any(q_type in module_name for q_type in ['Quantized', 'Int8', 'FP8', 'QFunctional']):
            if not silent:
                print("✅ Model is quantized")
            return True
        
        # Check for scale and zero_point in dynamic/static quantization
        if hasattr(module, 'scale') and hasattr(module, 'zero_point'):
            if not silent:
                print("✅ Model is quantized")
            return True
            
        # For TorchScript quantized models
        if hasattr(module, '_packed_params'):
            if not silent:
                print("✅ Model is quantized")
            return True
    
    # Check for fake quantization modules that are active
    for name, buffer in model.named_buffers():
        if 'fake_quant' in name or 'observer' in name:
            if not silent:
                print("✅ Model is quantized")
            return True
    
    if not silent:
        print("❌ Model is not quantized")
    return False

# ------ PRUNING UTILITIES ------ #

def find_prunable_modules(model: nn.Module, add_bias=False) -> List[Tuple[nn.Module, str]]:
    """
    Find all prunable modules in the model (Conv2d and Linear layers).
    
    Args:
        model: Model to analyze
        add_bias: Whether to add bias parameters, if they exist and you want to prune them
        
    Returns:
        List of (module, parameter_name) tuples
    """
    modules_to_prune = []
    
    # Iterate through all named modules
    for name, module in model.named_modules():
        # Check if the module is prunable (Conv2d or Linear)
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Add weight parameters
            modules_to_prune.append((module, 'weight'))
            
            # Optionally add bias parameters 
            if module.bias is not None and add_bias:
                modules_to_prune.append((module, 'bias'))
    
    return modules_to_prune

def calculate_sparsity(model: nn.Module) -> float:
    """
    Calculate the overall sparsity of the model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Sparsity ratio (percentage of zero weights)
    """
    total_params = 0
    zero_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Count weights
            weight = module.weight
            total_params += weight.nelement()
            zero_params += torch.sum(weight == 0).item()
            
            # Count bias if it exists
            if module.bias is not None:
                bias = module.bias
                total_params += bias.nelement()
                zero_params += torch.sum(bias == 0).item()
    
    # Calculate sparsity
    sparsity = 100.0 * zero_params / total_params
    
    return sparsity

def is_pruned(model: nn.Module, silent: Optional[bool]=False) -> bool:
    """Check if a model is pruned.
    
    Args:
        model: PyTorch model to check
        silent: Whether printing should happen or not
        
    Returns:
        Boolean indicating whether the model is pruned
    """
    is_pruned = nn.utils.prune.is_pruned(model)
    if is_pruned:
        if not silent:
            print("✅ Model is pruned")
        return True
    
    if not silent:
        print("❌ Model is not pruned")
    return False
