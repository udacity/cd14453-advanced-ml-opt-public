"""
UdaciSense Project: Visualization Utilities Module

This module provides functions for creating visualizations of model performance metrics,
training history, and model comparisons.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union
import torch
import torch.nn as nn

from utils import MAX_ALLOWED_ACCURACY_DROP, TARGET_INFERENCE_SPEEDUP, TARGET_MODEL_COMPRESSION 


# Set default style for consistent visualizations
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']


# ------ ACCURACY VISUALIZATION FUNCTIONS ------ #

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = True,
    cmap: str = 'Blues',
    dpi: int = 200
) -> plt.Figure:
    """Plot a confusion matrix.
    
    Args:
        confusion_matrix: NumPy array containing the confusion matrix
        class_names: List of class names
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        normalize: Whether to normalize the confusion matrix
        cmap: Colormap to use
        dpi: Resolution for saved figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize the confusion matrix if requested
    if normalize:
        row_sums = confusion_matrix.sum(axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        cm = np.around(
            confusion_matrix.astype('float') / row_sums[:, np.newaxis] * 100, 
            decimals=1
        )
        title = 'Normalized Confusion Matrix (%)'
        fmt = '.1f'
    else:
        cm = confusion_matrix
        title = 'Confusion Matrix (Count)'
        fmt = 'd'
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, ax=ax)
    
    # Set labels and title
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save the figure if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    
    # Plot figure
    plt.figure(fig)
    plt.show()
    
    return fig


def plot_per_class_accuracy(
    per_class_accuracy: Dict[str, float],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    sort_values: bool = True,
    color: str = '#3498db',
    dpi: int = 200
) -> plt.Figure:
    """Plot per-class accuracy.
    
    Args:
        per_class_accuracy: Dictionary mapping class names to accuracy values
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        sort_values: Whether to sort classes by accuracy
        color: Bar color
        dpi: Resolution for saved figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to pandas Series for easier manipulation
    accuracy_series = pd.Series(per_class_accuracy)
    
    # Sort if requested
    if sort_values:
        accuracy_series = accuracy_series.sort_values(ascending=False)
    
    # Create horizontal bar chart
    accuracy_series.plot(kind='barh', ax=ax, color=color)
    
    # Set labels and title
    ax.set_xlabel('Accuracy (%)')
    ax.set_ylabel('Class')
    ax.set_title('Per-Class Accuracy')
    
    # Add values at the end of each bar
    for i, value in enumerate(accuracy_series):
        ax.text(value + 1, i, f"{value:.1f}%", va='center')
    
    # Add a vertical line for mean accuracy
    mean_accuracy = accuracy_series.mean()
    ax.axvline(mean_accuracy, color='red', linestyle='--')
    ax.text(mean_accuracy + 1, len(accuracy_series) - 1, 
            f"Mean: {mean_accuracy:.1f}%", color='red', va='center')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save the figure if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Per-class accuracy plot saved to {output_path}")
    
    # Plot figure
    plt.figure(fig)
    plt.show()
    
    return fig

# ------ TRAINING HISTORY VISUALIZATION FUNCTIONS ------ #

def plot_training_history(
    training_stats: Dict[str, List],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 6),
    metrics: List[str] = ['loss', 'accuracy'],
    dpi: int = 200
) -> plt.Figure:
    """
    Plot training history metrics.
    
    Args:
        training_stats: Dictionary containing training statistics
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        metrics: List of metrics to plot ('loss', 'accuracy', 'lr')
        dpi: Resolution for saved figure
        
    Returns:
        Matplotlib figure
    """
    available_metrics = {
        'loss': ('train_loss', 'test_loss', 'Loss'),
        'accuracy': ('train_accuracy', 'test_accuracy', 'Accuracy (%)'),
        'lr': ('lr', None, 'Learning Rate')
    }
    
    # Calculate number of subplots needed
    num_plots = sum(1 for m in metrics if m in available_metrics)
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    
    # Handle single subplot case
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    for metric in metrics:
        if metric not in available_metrics:
            continue
            
        ax = axes[plot_idx]
        train_key, test_key, ylabel = available_metrics[metric]
        
        # Plot training metric if available
        if train_key in training_stats and len(training_stats[train_key]) > 0:
            ax.plot(training_stats['epoch'], training_stats[train_key], 
                   marker='o', markersize=4, linestyle='-', linewidth=2,
                   color=COLORS[0], label=f'Train {metric.title()}')
        
        # Plot test/validation metric if available
        if test_key and test_key in training_stats and len(training_stats[test_key]) > 0:
            ax.plot(training_stats['epoch'], training_stats[test_key], 
                   marker='s', markersize=4, linestyle='-', linewidth=2,
                   color=COLORS[1], label=f'Test {metric.title()}')
        
        # Set labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Training {metric.title()}')
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        if train_key in training_stats or (test_key and test_key in training_stats):
            ax.legend()
        
        plot_idx += 1
    
    # Overall title
    fig.suptitle('Training History', fontsize=16)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for the suptitle
    
    # Save the figure if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Training history plot saved to {output_path}")
        
    # Plot figure
    plt.figure(fig)
    plt.show()
    
    return fig

def plot_multiple_training_histories(
    histories: Dict[str, Dict[str, List]],
    metric: str,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    use_train: bool = False,
    dpi: int = 200
) -> plt.Figure:
    """
    Plot a comparison of training histories for multiple models.
    
    Args:
        histories: Dictionary mapping model names to training statistics
        metric: Metric to plot ('loss', 'accuracy', 'lr')
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        use_train: Whether to use training metrics (True) or test metrics (False)
        dpi: Resolution for saved figure
        
    Returns:
        Matplotlib figure
    """
    metric_keys = {
        'loss': ('train_loss', 'test_loss', 'Loss'),
        'accuracy': ('train_accuracy', 'test_accuracy', 'Accuracy (%)'),
        'lr': ('lr', None, 'Learning Rate')
    }
    
    # Get the appropriate metric key and label
    train_key, test_key, ylabel = metric_keys.get(metric, (metric, None, metric.title()))
    key_to_use = train_key if use_train else test_key
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each model's history
    for i, (model_name, history) in enumerate(histories.items()):
        if 'epoch' not in history or key_to_use not in history:
            continue
            
        ax.plot(history['epoch'], history[key_to_use], 
               marker='o' if i % 2 == 0 else 's', markersize=4, 
               linestyle='-', linewidth=2,
               color=COLORS[i % len(COLORS)], label=model_name)
    
    # Set labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    data_type = 'Training' if use_train else 'Validation'
    ax.set_title(f'{data_type} {metric.title()} Comparison')
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Adjust layout
    fig.tight_layout()
    
    # Save the figure if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Comparison plot saved to {output_path}")
        
    # Plot figure
    plt.figure(fig)
    plt.show()
    
    return fig

# ------ MODEL COMPARISON VISUALIZATION FUNCTIONS ------ #

def plot_model_comparison(
    comparison_results: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    device_priority: str = 'cpu',
    dpi: int = 200
) -> plt.Figure:
    """
    Create a visual comparison of baseline and optimized models with support for multi-device inference times.
    
    Args:
        comparison_results: Results from evaluation.compare_models function
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        device_priority: Which device to prioritize for visualization ('cpu' or 'cuda')
        dpi: Resolution for saved figure
        
    Returns:
        Matplotlib figure
    """
    baseline = comparison_results['baseline']
    optimized = comparison_results['optimized']
    improvements = comparison_results['improvements']
    
    # Check for device availability
    has_cuda_support = (baseline['timing'].get('cuda') and 
                optimized['timing'].get('cuda'))
    has_cpu_support = (baseline['timing'].get('cpu') and 
                optimized['timing'].get('cpu'))
    
    # Calculate layout dimensions
    num_plots = 5
    num_rows = (num_plots + 1) // 2  # Round up to nearest integer
    num_cols = 2
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Track current subplot index
    plot_idx = 0
    
    # 1. Model Size (MB)
    ax = axes[plot_idx]
    plot_idx += 1
    sizes = [baseline['size']['model_size_mb'], optimized['size']['model_size_mb']]
    ax.bar(['Baseline', 'Optimized'], sizes, color=[COLORS[0], COLORS[1]])
    ax.set_title('Model Size (MB)')
    ax.set_ylabel('Size (MB)')
    
    # Add value labels
    for i, v in enumerate(sizes):
        ax.text(i, v, f"{v:.2f} MB", ha='center', va='bottom')
    
    # Add improvement text
    size_reduction = improvements['size_reduction'] * 100
    ax.text(0.5, 0.9, f"{size_reduction:.1f}% reduction", 
            transform=ax.transAxes, ha='center', fontsize=12, color='black',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))
    
    # 2. Parameter Count
    ax = axes[plot_idx]
    plot_idx += 1
    params = [
        baseline['size']['trainable_params'] / 1e6, 
        optimized['size']['trainable_params'] / 1e6
    ]
    ax.bar(['Baseline', 'Optimized'], params, color=[COLORS[0], COLORS[1]])
    ax.set_title('Parameters (millions)')
    ax.set_ylabel('Parameters (M)')
    
    # Add value labels
    for i, v in enumerate(params):
        ax.text(i, v, f"{v:.2f}M", ha='center', va='bottom')
    
    # Add improvement text
    param_reduction = improvements['params_reduction'] * 100
    ax.text(0.5, 0.9, f"{param_reduction:.1f}% reduction", 
            transform=ax.transAxes, ha='center', fontsize=12, color='black',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))
    
    # Handle inference time plots based on available data
    
    # 3a. CUDA Inference Time (ms) - if available
    if has_cuda_support:
        ax = axes[plot_idx]
        plot_idx += 1
        
        cuda_times = [
            baseline['timing']['cuda']['avg_time_ms'],
            optimized['timing']['cuda']['avg_time_ms']
        ]
        ax.bar(['Baseline', 'Optimized'], cuda_times, color=[COLORS[0], COLORS[1]])
        ax.set_title('CUDA Inference Time (ms)')
        ax.set_ylabel('Time (ms)')
        
        # Add value labels
        for i, v in enumerate(cuda_times):
            ax.text(i, v, f"{v:.2f} ms", ha='center', va='bottom')
        
        # Add improvement text
        cuda_speedup = improvements['inference']['cuda']['speedup']
        ax.text(0.5, 0.9, f"{cuda_speedup:.1f}x speedup",
                transform=ax.transAxes, ha='center', fontsize=12, color='black',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))
    else:
        ax = axes[plot_idx]
        plot_idx += 1
        ax.text(0.5, 0.5, "CUDA inference not available",
                ha='center', va='center', fontsize=14, color='gray')
        ax.set_title('CUDA Inference Time (ms)')
        ax.axis('off')
    

    if has_cpu_support:
        # 3a. CPU Inference Time (ms)
        ax = axes[plot_idx]
        plot_idx += 1
        cpu_times = [
            baseline['timing']['cpu']['avg_time_ms'], 
            optimized['timing']['cpu']['avg_time_ms']
        ]
        ax.bar(['Baseline', 'Optimized'], cpu_times, color=[COLORS[0], COLORS[1]])
        ax.set_title('CPU Inference Time (ms)')
        ax.set_ylabel('Time (ms)')

        # Add value labels
        for i, v in enumerate(cpu_times):
            ax.text(i, v, f"{v:.2f} ms", ha='center', va='bottom')

        # Add improvement text
        if 'cpu' in improvements['inference']:
            cpu_speedup = improvements['inference']['cpu']['speedup']
            ax.text(0.5, 0.9, f"{cpu_speedup:.1f}x speedup", 
                    transform=ax.transAxes, ha='center', fontsize=12, color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))
    else:
        ax = axes[plot_idx]
        plot_idx += 1
        ax.text(0.5, 0.5, "CPU inference not available",
                ha='center', va='center', fontsize=14, color='gray')
        ax.set_title('CPU Inference Time (ms)')
        ax.axis('off')

    # 4. Accuracy
    ax = axes[plot_idx]
    plot_idx += 1
    accs = [
        baseline['accuracy']['top1_acc'], 
        optimized['accuracy']['top1_acc']
    ]
    ax.bar(['Baseline', 'Optimized'], accs, color=[COLORS[0], COLORS[1]])
    ax.set_title('Top-1 Accuracy (%)')
    ax.set_ylabel('Accuracy (%)')
    
    # Add value labels
    for i, v in enumerate(accs):
        ax.text(i, v, f"{v:.2f}%", ha='center', va='bottom')
    
    # Add improvement text
    acc_change = improvements['accuracy_change'] * 100
    color = 'green' if acc_change >= 0 else 'red'
    ax.text(0.5, 0.9, f"{acc_change:+.2f}% points", 
            transform=ax.transAxes, ha='center', fontsize=12, color=color,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))
    
    # Hide any unused subplots
    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])
    
    # Overall title
    fig.suptitle('Model Optimization Results', fontsize=16)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    
    # Save the figure if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Model comparison plot saved to {output_path}")
    
    # Plot figure
    plt.figure(fig)
    plt.show()
    
    return fig

def plot_multiple_models_comparison(
    models_data: Dict[str, Dict[str, Any]],
    metrics: List[str] = ['accuracy', 'size', 'cpu_inference_time', 'cuda_inference_time', 'parameters'],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    normalize: bool = True,
    dpi: int = 200
) -> plt.Figure:
    """
    Create a bar chart comparing multiple models across different metrics.
    Supports multi-device inference times.
    
    Args:
        models_data: Dictionary mapping model names to metric dictionaries
        metrics: List of metrics to compare
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        normalize: Whether to normalize metrics (1.0 = best model for each metric)
        dpi: Resolution for saved figure
        
    Returns:
        Matplotlib figure
    """
    # Extract metric values for each model
    metric_values = {}
    model_names = list(models_data.keys())
    
    for metric in metrics:
        metric_values[metric] = []
        
        # Extract value based on metric type
        for model_name in model_names:
            model = models_data[model_name]
            
            if metric == 'accuracy':
                value = model.get('accuracy', {}).get('top1_acc', 0)
            elif metric == 'size':
                value = model.get('size', {}).get('model_size_mb', 0)
            elif metric == 'parameters':
                value = model.get('size', {}).get('trainable_params', 0) / 1e6  # In millions
            elif metric == 'cpu_inference_time':
                # Get CPU inference time
                value = model.get('timing', {}).get('cpu', {}).get('avg_time_ms', 0)
            elif metric == 'cuda_inference_time':
                # Get CUDA inference time
                value = model.get('timing', {}).get('cuda', {}).get('avg_time_ms', 0)
            elif metric == 'inference_time':
                # Try to get from either multi-device or legacy format
                value = model.get('timing', {}).get('cpu', {}).get('avg_time_ms', 0)
            else:
                # Try to get the value directly
                value = model.get(metric, 0)
                
            metric_values[metric].append(value)
    
    # Normalize if requested
    if normalize:
        for metric in metrics:
            values = metric_values[metric]
            
            # Skip if all values are zero
            if all(v == 0 for v in values):
                continue
                
            # Determine if higher is better
            higher_is_better = metric in ['accuracy']
            
            if higher_is_better:
                best_value = max(values)
                metric_values[metric] = [v / best_value if best_value > 0 else 0 for v in values]
            else:
                best_value = min(v for v in values if v > 0) if any(v > 0 for v in values) else 1
                metric_values[metric] = [best_value / v if v > 0 else 0 for v in values]
    
    # Create figure
    num_metrics = len(metrics)
    num_cols = min(3, num_metrics)
    num_rows = (num_metrics + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Handle single axis case
    if num_metrics == 1:
        axes = np.array([axes])
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Create a bar plot for each metric
    for i, metric in enumerate(metrics):
        if i < len(axes):
            ax = axes[i]
            
            # Create bar chart
            x_pos = np.arange(len(model_names))
            ax.bar(x_pos, metric_values[metric], color=[COLORS[j % len(COLORS)] for j in range(len(model_names))])
            
            # Set labels and title
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            
            # Set title and y-label based on metric
            if metric == 'accuracy':
                title = 'Accuracy'
                ylabel = 'Top-1 Accuracy (%)' if not normalize else 'Normalized Accuracy'
            elif metric == 'size':
                title = 'Model Size'
                ylabel = 'Size (MB)' if not normalize else 'Normalized Size'
            elif metric == 'cpu_inference_time':
                title = 'CPU Inference Time'
                ylabel = 'Time (ms)' if not normalize else 'Normalized Speed'
            elif metric == 'cuda_inference_time':
                title = 'CUDA Inference Time'
                ylabel = 'Time (ms)' if not normalize else 'Normalized Speed'
            elif metric == 'inference_time':
                title = 'Inference Time'
                ylabel = 'Time (ms)' if not normalize else 'Normalized Speed'
            elif metric == 'parameters':
                title = 'Model Parameters'
                ylabel = 'Parameters (M)' if not normalize else 'Normalized Parameters'
            else:
                title = metric.replace('_', ' ').title()
                ylabel = metric.replace('_', ' ').title()
            
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            
            # Add value labels
            for j, v in enumerate(metric_values[metric]):
                if normalize:
                    ax.text(j, v, f"{v:.2f}", ha='center', va='bottom')
                else:
                    # Format based on metric
                    if metric == 'accuracy':
                        label = f"{v:.1f}%"
                    elif metric == 'size':
                        label = f"{v:.1f} MB"
                    elif 'inference_time' in metric:
                        label = f"{v:.1f} ms"
                    elif metric == 'parameters':
                        label = f"{v:.1f}M"
                    else:
                        label = f"{v:.2f}"
                    
                    ax.text(j, v, label, ha='center', va='bottom')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Hide unused subplots
    for i in range(num_metrics, len(axes)):
        fig.delaxes(axes[i])
    
    # Overall title
    title = 'Normalized Model Comparison' if normalize else 'Model Comparison'
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    
    # Save the figure if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Multiple models comparison plot saved to {output_path}")
        
    # Plot figure
    plt.figure(fig)
    plt.show()
    
    return fig

# ------ MODEL ANALYSIS VISUALIZATION FUNCTIONS ------ #

def plot_weight_distribution(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
    num_layers: int = 4,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
    dpi: int = 200
) -> plt.Figure:
    """
    Plot the distribution of weights in selected layers.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to plot (if None, selects automatically)
        num_layers: Number of layers to plot if layer_names is None
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        dpi: Resolution for saved figure
        
    Returns:
        Matplotlib figure
    """
    
    # If no layer names provided, select layers automatically
    if layer_names is None:
        layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight'):
                layer_names.append(name)
                if len(layer_names) >= num_layers:
                    break
    
    # Skip if no suitable layers found
    if not layer_names:
        print("No suitable layers found for weight distribution visualization.")
        return None
    
    # Create figure
    n_layers = len(layer_names)
    nrows = (n_layers + 1) // 2  # Arrange in rows of 2
    ncols = min(2, n_layers)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Handle single subplot case
    if n_layers == 1:
        axes = np.array([axes])
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Plot weight distribution for each layer
    for i, name in enumerate(layer_names):
        if i < len(axes):
            ax = axes[i]
            
            # Get the layer by name
            layer = None
            for n, m in model.named_modules():
                if n == name:
                    layer = m
                    break
            
            if layer is None or not hasattr(layer, 'weight'):
                continue
            
            # Get weights and flatten
            weights = layer.weight.data.cpu().numpy().flatten()
            
            # Plot histogram
            ax.hist(weights, bins=50, alpha=0.7, color=COLORS[i % len(COLORS)])
            
            # Add distribution statistics
            mean = np.mean(weights)
            std = np.std(weights)
            min_val = np.min(weights)
            max_val = np.max(weights)
            
            stats_text = f"Mean: {mean:.4f}\nStd: {std:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Set title and labels
            ax.set_title(f"{name} ({layer.__class__.__name__})")
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Count')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
    
    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        fig.delaxes(axes[i])
    
    # Overall title
    fig.suptitle('Weight Distributions', fontsize=16)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    
    # Save the figure if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Weight distribution plot saved to {output_path}")
        
    # Plot figure
    plt.figure(fig)
    plt.show()
    
    return fig

def plot_sparsity_heatmap(
    model: nn.Module,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    threshold: float = 1e-6,
    dpi: int = 200
) -> plt.Figure:
    """
    Create a heatmap showing sparsity levels in different layers.
    
    Args:
        model: PyTorch model (potentially pruned)
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        threshold: Threshold below which weights are considered zero
        dpi: Resolution for saved figure
        
    Returns:
        Matplotlib figure
    """
    # Collect layer sparsity data
    layer_data = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight'):
            # Calculate sparsity (percentage of zeros or near-zeros)
            weight = module.weight.data
            total = weight.numel()
            zeros = (torch.abs(weight) < threshold).sum().item()
            sparsity = zeros / total if total > 0 else 0
            
            layer_data.append({
                'name': name,
                'type': module.__class__.__name__,
                'shape': list(module.weight.shape),
                'params': total,
                'zeros': zeros,
                'sparsity': sparsity
            })
    
    # Skip if no suitable layers found
    if not layer_data:
        print("No suitable layers found for sparsity visualization.")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(layer_data)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                  gridspec_kw={'height_ratios': [1, 3]})
    
    # Plot overall sparsity
    if len(df) > 0:
        total_params = df['params'].sum()
        total_zeros = df['zeros'].sum()
        overall_sparsity = total_zeros / total_params if total_params > 0 else 0
        ax1.bar(['Model Sparsity'], [overall_sparsity], color=COLORS[0])
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Sparsity Ratio')
        ax1.set_title(f'Overall Model Sparsity: {overall_sparsity:.1%}')
        
        # Add exact value
        ax1.text(0, overall_sparsity, f"{overall_sparsity:.2%}", 
                ha='right', va='bottom')
        
        # Plot layer-wise sparsity
        if len(df) > 1:  # Only create heatmap if we have multiple layers
            # Sort by parameter count (largest first)
            df = df.sort_values('params', ascending=False)
            
            # Create horizontal bar chart
            sns.barh(y='name', x='sparsity', data=df, ax=ax2, color=COLORS[0])
            ax2.set_xlim(0, 1)
            ax2.set_xlabel('Sparsity Ratio')
            ax2.set_ylabel('Layer Name')
            ax2.set_title('Layer-wise Sparsity')
            
            # Add parameter count information
            for i, row in enumerate(df.itertuples()):
                # Add sparsity value
                ax2.text(row.sparsity + 0.02, i, f"{row.sparsity:.2%}", va='center')
                
                # Add parameter count at the beginning of the bar
                ax2.text(0.01, i, f"{row.params:,} params", va='center')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save the figure if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Sparsity heatmap saved to {output_path}")
        
    # Plot figure
    plt.figure(fig)
    plt.show()
    
    return fig

# ------ COMPREHENSIVE VISUALIZATION FUNCTIONS ------ #

def create_model_summary_dashboard(
    comparison_results: Dict[str, Dict[str, Any]],
    requirements_met: Dict[str, bool],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    device_priority: str = 'cpu',
    dpi: int = 200
) -> plt.Figure:
    """
    Create a comprehensive dashboard with model comparison and requirements.
    Supports multi-device inference times.
    
    Args:
        comparison_results: Results from evaluation.compare_models function
        requirements_met: Results from evaluation.evaluate_requirements_met function
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        device_priority: Which device to prioritize for requirements ('cpu' or 'cuda')
        dpi: Resolution for saved figure
        
    Returns:
        Matplotlib figure
    """
    baseline = comparison_results['baseline']
    optimized = comparison_results['optimized']
    improvements = comparison_results['improvements']
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=figsize, gridspec_kw={'height_ratios': [1, 2, 1]})
    
    # Plot requirements status
    ax = axes[0, 0]
    req_names = ['Size', 'Speed', 'Accuracy']
    
    # Get the appropriate speed requirement status based on device priority
    if  f'speed_requirement_met_{device_priority}' in requirements_met:
        speed_met = requirements_met[f'speed_requirement_met_{device_priority}']
    else:
        speed_met = requirements_met.get('speed_requirement_met', False)
    
    req_status = [
        requirements_met['size_requirement_met'],
        speed_met,
        requirements_met['accuracy_requirement_met']
    ]
    colors = ['green' if status else 'red' for status in req_status]
    ax.bar(req_names, [1, 1, 1], color=colors)
    ax.set_ylim(0, 1.5)
    ax.set_title('Requirements Status')
    
    # Add text labels for requirements
    for i, (name, status) in enumerate(zip(req_names, req_status)):
        text = 'MET' if status else 'NOT MET'
        ax.text(i, 0.5, text, ha='center', va='center', color='white', fontweight='bold')
    
    # Plot overall success status
    ax = axes[0, 1]
    overall_status = requirements_met['all_requirements_met']
    ax.bar(['Overall Status'], [1], color='green' if overall_status else 'red')
    ax.set_ylim(0, 1.5)
    ax.text(0, 0.5, 'SUCCESS' if overall_status else 'NEEDS IMPROVEMENT',
            ha='center', va='center', color='white', fontweight='bold')
    
    # Plot model comparison metrics
    ax = axes[1, 0]
    
    # Set up metrics
    metrics = ['Size (MB)', 'Parameters (M)']
    baseline_values = [
        baseline['size']['model_size_mb'],
        baseline['size']['trainable_params'] / 1e6
    ]
    optimized_values = [
        optimized['size']['model_size_mb'],
        optimized['size']['trainable_params'] / 1e6
    ]
    
    # Add inference times based on available data
    ## Add CPU inference time
    metrics.append('CPU Inference (ms)')
    baseline_values.append(baseline['timing']['cpu']['avg_time_ms'])
    optimized_values.append(optimized['timing']['cpu']['avg_time_ms'])
    
    ## Add CUDA inference time if available
    if baseline['timing'].get('cuda') and optimized['timing'].get('cuda'):
        metrics.append('CUDA Inference (ms)')
        baseline_values.append(baseline['timing']['cuda']['avg_time_ms'])
        optimized_values.append(optimized['timing']['cuda']['avg_time_ms'])
    
    # Add accuracy
    metrics.append('Accuracy (%)')
    baseline_values.append(baseline['accuracy']['top1_acc'])
    optimized_values.append(optimized['accuracy']['top1_acc'])
    
    # Set up bar positions
    x = np.arange(len(metrics))
    width = 0.35
    
    # Create grouped bar chart
    ax.bar(x - width/2, baseline_values, width, label='Baseline', color=COLORS[0])
    ax.bar(x + width/2, optimized_values, width, label='Optimized', color=COLORS[1])
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_title('Model Metrics Comparison')
    
    # Plot improvements
    ax = axes[1, 1]
    improvement_metrics = ['Size Reduction', 'Params Reduction']
    improvement_values = [
        improvements['size_reduction'] * 100,  # Convert to percentage
        improvements['params_reduction'] * 100,  # Convert to percentage
    ]
    
    # Add inference time improvements
    if 'cpu' in improvements.get('inference', {}):
        improvement_metrics.append('CPU Speedup')
        improvement_values.append(improvements['inference']['cpu']['speedup'])
    
    if improvements.get('inference', {}).get('cuda'):
        improvement_metrics.append('CUDA Speedup')
        improvement_values.append(improvements['inference']['cuda']['speedup'])
    
    # Add accuracy change
    improvement_metrics.append('Accuracy Change')
    improvement_values.append(improvements['accuracy_change'] * 100)
    
    # Define colors based on whether the change is good or bad
    colors = ['green' for _ in range(len(improvement_metrics) - 1)]
    colors.append('green' if improvements['accuracy_change'] >= 0 else 'red')
    
    # Create bar chart
    bars = ax.bar(improvement_metrics, improvement_values, color=colors)
    ax.set_title('Improvements')
    
    # Add value labels
    for i, (metric, value) in enumerate(zip(improvement_metrics, improvement_values)):
        if 'Speedup' in metric:
            label = f"{value:.1f}x"
        elif metric == 'Accuracy Change':
            label = f"{value:+.1f}%"
        else:
            label = f"{value:.1f}%"
        ax.text(i, value, label, ha='center', va='bottom')
    
    # Add summary text
    ax = axes[2, 0]
    ax.axis('off')  # Hide axes
    
    # Create summary text based on available data
    summary_text = [
        f"MODEL OPTIMIZATION SUMMARY\n\n",
        f"Size: {baseline['size']['model_size_mb']:.1f}MB → {optimized['size']['model_size_mb']:.1f}MB "
        f"({improvements['size_reduction']:.1%} reduction)\n",
        f"Parameters: {baseline['size']['trainable_params']:,} → {optimized['size']['trainable_params']:,} "
        f"({improvements['params_reduction']:.1%} reduction)\n"
    ]
    
    # Add inference time summaries
    ## CPU inference time
    summary_text.append(
        f"CPU Inference: {baseline['timing']['cpu']['avg_time_ms']:.1f}ms → "
        f"{optimized['timing']['cpu']['avg_time_ms']:.1f}ms "
        f"({improvements['inference']['cpu']['speedup']:.1f}x speedup)\n"
    )
    
    ## CUDA inference time (if available)
    if (baseline['timing'].get('cuda') and optimized['timing'].get('cuda') and
        improvements['inference'].get('cuda')):
        summary_text.append(
            f"CUDA Inference: {baseline['timing']['cuda']['avg_time_ms']:.1f}ms → "
            f"{optimized['timing']['cuda']['avg_time_ms']:.1f}ms "
            f"({improvements['inference']['cuda']['speedup']:.1f}x speedup)\n"
        )
    
    # Add accuracy summary
    summary_text.append(
        f"Accuracy: {baseline['accuracy']['top1_acc']:.1f}% → {optimized['accuracy']['top1_acc']:.1f}% "
        f"({improvements['accuracy_change']*100:+.1f}% points)"
    )
    
    # Join summary text parts
    summary_text_str = ''.join(summary_text)
    
    # Display summary
    ax.text(0.5, 0.5, summary_text_str, ha='center', va='center', transform=ax.transAxes,
           bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', boxstyle='round,pad=0.5'))
    
    # Add requirements details
    ax = axes[2, 1]
    ax.axis('off')  # Hide axes
    
    # Create requirements text based on available data
    requirements_text = [
        f"REQUIREMENTS STATUS\n\n",
        f"Size Reduction: {TARGET_MODEL_COMPRESSION*100}% required, {improvements['size_reduction']*100:.1f}% achieved - "
        f"{'✓ MET' if requirements_met['size_requirement_met'] else '✗ NOT MET'}\n"
    ]
    
    # Add the appropriate speed requirement status for the prioritized device
    if device_priority in improvements.get('inference', {}):
        speed_reduction = improvements['inference'][device_priority]['reduction'] * 100
        device_label = device_priority.upper()
        speed_req_met = requirements_met.get(f'speed_requirement_met_{device_priority}', False)
        
        requirements_text.append(
            f"{device_label} Speed Improvement: {TARGET_INFERENCE_SPEEDUP*100}% required, {speed_reduction:.1f}% achieved - "
            f"{'✓ MET' if speed_req_met else '✗ NOT MET'}\n"
        )
    
    # Add accuracy requirement
    requirements_text.append(
        f"Accuracy Drop: Max {MAX_ALLOWED_ACCURACY_DROP*100}% allowed, {abs(improvements['accuracy_change']*100):.1f}% actual - "
        f"{'✓ MET' if requirements_met['accuracy_requirement_met'] else '✗ NOT MET'}\n\n"
    )
    
    # Add overall status
    requirements_text.append(
        f"OVERALL: {'✓ ALL REQUIREMENTS MET' if requirements_met['all_requirements_met'] else '✗ NOT ALL REQUIREMENTS MET'}"
    )
    
    # Join requirements text parts
    requirements_text_str = ''.join(requirements_text)
    
    # Display requirements
    ax.text(0.5, 0.5, requirements_text_str, ha='center', va='center', transform=ax.transAxes,
           bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', boxstyle='round,pad=0.5'))
    
    # Add overall title
    fig.suptitle('Model Optimization Summary Dashboard', fontsize=16)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    
    # Save the figure if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Model summary dashboard saved to {output_path}")
        
    # Plot figure
    plt.figure(fig)
    plt.show()
    
    return fig

def create_evaluation_report(
    model_results: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    include_plots: bool = True,
    plot_dir: Optional[str] = None,
    dpi: int = 200
) -> Optional[pd.DataFrame]:
    """
    Create a comprehensive evaluation report for a model.
    
    Args:
        model_results: Results from evaluation functions
        output_path: Path to save the CSV report (optional)
        include_plots: Whether to generate plots
        plot_dir: Directory to save plots (required if include_plots is True)
        dpi: Resolution for saved figures
        
    Returns:
        DataFrame with evaluation results
    """
    # Create a flattened dictionary for the report
    report_data = {}
    
    # Extract accuracy metrics
    if 'accuracy' in model_results:
        for k, v in model_results['accuracy'].items():
            report_data[f'accuracy_{k}'] = v
    
    # Extract timing metrics
    if 'timing' in model_results:
        for k, v in model_results['timing'].items():
            report_data[f'timing_{k}'] = v
    
    # Extract size metrics
    if 'size' in model_results:
        for k, v in model_results['size'].items():
            report_data[f'size_{k}'] = v
    
    # Extract memory metrics if available
    if 'memory' in model_results:
        for k, v in model_results['memory'].items():
            if v is not None:  # Skip None values
                report_data[f'memory_{k}'] = v
    
    # Create DataFrame
    report_df = pd.DataFrame([report_data])
    
    # Generate plots if requested
    if include_plots and plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot per-class accuracy if available
        if 'per_class_accuracy' in model_results and model_results['per_class_accuracy']:
            plot_per_class_accuracy(
                model_results['per_class_accuracy'],
                output_path=os.path.join(plot_dir, 'per_class_accuracy.png'),
                dpi=dpi
            )
        
        # Add more plots as needed...
    
    # Save report if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report_df.to_csv(output_path, index=False)
        print(f"Evaluation report saved to {output_path}")
    
    return report_df

def create_comparison_report(
    comparison_results: Dict[str, Dict[str, Any]],
    requirements_met: Dict[str, bool],
    output_path: Optional[str] = None,
    include_plots: bool = True,
    plot_dir: Optional[str] = None,
    dpi: int = 200
) -> Optional[pd.DataFrame]:
    """
    Create a comprehensive comparison report between baseline and optimized models.
    
    Args:
        comparison_results: Results from evaluation.compare_models function
        requirements_met: Results from evaluation.evaluate_requirements_met function
        output_path: Path to save the CSV report (optional)
        include_plots: Whether to generate plots
        plot_dir: Directory to save plots (required if include_plots is True)
        dpi: Resolution for saved figures
        
    Returns:
        DataFrame with comparison results
    """
    baseline = comparison_results['baseline']
    optimized = comparison_results['optimized']
    improvements = comparison_results['improvements']
    
    # Create data for the report
    data = {
        'Metric': [
            'Model Size (MB)',
            'Parameters (M)',
            'Inference Time (ms)',
            'Top-1 Accuracy (%)',
            'Memory Usage (MB)',
            'Size Reduction (%)',
            'Parameter Reduction (%)',
            'Inference Speedup (x)',
            'Accuracy Change (pp)',
            'Size Requirement Met',
            'Speed Requirement Met',
            'Accuracy Requirement Met',
            'All Requirements Met'
        ],
        'Baseline': [
            baseline['size']['model_size_mb'],
            baseline['size']['trainable_params'] / 1e6,
            baseline['timing']['avg_time_ms'],
            baseline['accuracy']['top1_acc'],
            baseline['memory']['peak_memory_mb'] if 'memory' in baseline and 'peak_memory_mb' in baseline['memory'] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        ],
        'Optimized': [
            optimized['size']['model_size_mb'],
            optimized['size']['trainable_params'] / 1e6,
            optimized['timing']['avg_time_ms'],
            optimized['accuracy']['top1_acc'],
            optimized['memory']['peak_memory_mb'] if 'memory' in optimized and 'peak_memory_mb' in optimized['memory'] else None,
            improvements['size_reduction'] * 100,
            improvements['params_reduction'] * 100,
            improvements['inference_speedup'],
            improvements['accuracy_change'] * 100,
            'Yes' if requirements_met['size_requirement_met'] else 'No',
            'Yes' if requirements_met['speed_requirement_met'] else 'No',
            'Yes' if requirements_met['accuracy_requirement_met'] else 'No',
            'Yes' if requirements_met['all_requirements_met'] else 'No'
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate plots if requested
    if include_plots and plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        
        # Model comparison plot
        plot_model_comparison(
            comparison_results,
            output_path=os.path.join(plot_dir, 'model_comparison.png'),
            dpi=dpi
        )
        
        # Summary dashboard
        create_model_summary_dashboard(
            comparison_results,
            requirements_met,
            output_path=os.path.join(plot_dir, 'summary_dashboard.png'),
            dpi=dpi
        )
    
    # Save report if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Comparison report saved to {output_path}")
    
    return df

def plot_trade_off_scatter(
    techniques: List[Dict[str, Any]],
    x_metric: str = 'model_size_mb',
    y_metric: str = 'accuracy',
    size_metric: str = 'inference_time_ms',
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 200
) -> plt.Figure:
    """
    Create a scatter plot showing trade-offs between different metrics.
    
    Args:
        techniques: List of dictionaries with metrics for different techniques
        x_metric: Metric to plot on x-axis
        y_metric: Metric to plot on y-axis
        size_metric: Metric to use for point sizes
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        dpi: Resolution for saved figure
        
    Returns:
        Matplotlib figure with trade-off scatter plot
    """
    # Extract data points
    names = [t.get('name', f"Technique {i+1}") for i, t in enumerate(techniques)]
    
    # Extract metric values, handling nested dictionaries
    def extract_value(technique, metric):
        parts = metric.split('.')
        value = technique
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value
    
    x_values = [extract_value(t, x_metric) for t in techniques]
    y_values = [extract_value(t, y_metric) for t in techniques]
    sizes = [extract_value(t, size_metric) for t in techniques]
    
    # Filter out None values
    valid_indices = [i for i, (x, y, s) in enumerate(zip(x_values, y_values, sizes)) 
                    if x is not None and y is not None and s is not None]
    
    names = [names[i] for i in valid_indices]
    x_values = [x_values[i] for i in valid_indices]
    y_values = [y_values[i] for i in valid_indices]
    sizes = [sizes[i] for i in valid_indices]
    
    # Skip if no valid data points
    if not valid_indices:
        print("No valid data points for trade-off scatter plot.")
        return None
    
    # Normalize sizes for plotting
    min_size, max_size = min(sizes), max(sizes)
    if min_size == max_size:
        normalized_sizes = [100] * len(sizes)  # Same size for all points
    else:
        normalized_sizes = [50 + 250 * (s - min_size) / (max_size - min_size) for s in sizes]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    scatter = ax.scatter(x_values, y_values, s=normalized_sizes, alpha=0.7, 
                        c=range(len(names)), cmap='viridis')
    
    # Add labels for each point
    for i, name in enumerate(names):
        ax.annotate(name, (x_values[i], y_values[i]),
                   xytext=(10, 0), textcoords='offset points')
    
    # Add axis labels and title
    x_label = x_metric.replace('_', ' ').replace('.', ' ').title()
    y_label = y_metric.replace('_', ' ').replace('.', ' ').title()
    size_label = size_metric.replace('_', ' ').replace('.', ' ').title()
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'Trade-offs: {y_label} vs {x_label}\n(Point size represents {size_label})')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add colorbar legend
    plt.colorbar(scatter, label='Technique Index')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save the figure if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Trade-off scatter plot saved to {output_path}")
        
    # Plot figure
    plt.figure(fig)
    plt.show()
    
    return fig

# ------ UTILITY FUNCTIONS ------ #

def save_plot(fig: plt.Figure, output_path: str, dpi: int = 200) -> None:
    """
    Save a matplotlib figure to file, creating directories if needed.
    
    Args:
        fig: Matplotlib figure
        output_path: Path to save the figure
        dpi: Resolution for saved figure
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

def show_or_save_plot(fig: plt.Figure, output_path: Optional[str] = None, dpi: int = 200) -> None:
    """
    Either show or save a matplotlib figure based on output_path.
    
    Args:
        fig: Matplotlib figure
        output_path: Path to save the figure (if None, shows the plot)
        dpi: Resolution for saved figure
    """
    if output_path:
        save_plot(fig, output_path, dpi)
    else:
        plt.show()
    
    # Close figure to free memory
    plt.close(fig)