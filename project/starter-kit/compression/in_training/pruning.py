"""
UdaciSense Project: Training-Time Pruning Module

This module provides functions for implementing Gradual Magnitude Pruning (GMP)
during model training.
"""

import os
import copy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Tuple, Optional, List, Callable, Union, Literal

from utils.model import count_parameters, save_model, load_model, train_single_epoch, validate_single_epoch
from utils.compression import calculate_sparsity, find_prunable_modules, is_pruned

def compute_sparsity_schedule(
    initial_sparsity: float,
    final_sparsity: float,
    start_epoch: int,
    end_epoch: int,
    epochs: int,
    schedule_type: str = 'cubic'
) -> List[float]:
    """
    Compute sparsity schedule for gradual pruning.
    
    Args:
        initial_sparsity: Initial sparsity factor (0.0 to 1.0)
        final_sparsity: Target sparsity factor (0.0 to 1.0)
        start_epoch: Epoch to start pruning
        end_epoch: Epoch to end pruning
        epochs: Total number of epochs
        schedule_type: Type of schedule ('linear', 'exponential', 'cubic')
        
    Returns:
        List of sparsity factors for each epoch
    """
    # Validate inputs
    if initial_sparsity < 0.0 or initial_sparsity > 1.0:
        raise ValueError(f"Invalid initial sparsity: {initial_sparsity}. Must be between 0.0 and 1.0")
    if final_sparsity < initial_sparsity or final_sparsity > 1.0:
        raise ValueError(f"Invalid final sparsity: {final_sparsity}. Must be between {initial_sparsity} and 1.0")
    if start_epoch < 0 or start_epoch >= epochs:
        raise ValueError(f"Invalid start epoch: {start_epoch}. Must be between 0 and {epochs-1}")
    if end_epoch <= start_epoch or end_epoch >= epochs:
        raise ValueError(f"Invalid end epoch: {end_epoch}. Must be between {start_epoch+1} and {epochs-1}")
    
    # TODO: Create schedule
    # Feel free to implement one or all schedule types
    # Remember that sparsity should increase following the schedule type in [start_epoch, end_epoch]
    sparsity_schedule = None
    return sparsity_schedule

def prune_model_to_target(
    model: nn.Module,
    target_sparsity: float,
    pruning_method: str = 'global_unstructured',
    only_prune_conv: bool = False
) -> nn.Module:
    """
    Apply pruning to a model to reach a target sparsity level.
    
    Args:
        model: Model to prune
        target_sparsity: Target sparsity level (0.0 to 1.0)
        pruning_method: Pruning method ('l1_unstructured', 'random_unstructured', 'global_unstructured')
        only_prune_conv: Whether to only prune convolutional layers
        
    Returns:
        Pruned model
    """
    # TODO: Apply pruning based on method
    # Feel free to implement one or all pruning methods
    # Remember that you can find modules to prune with the find_prunable_modules() function
    # and that pruning reparameterization should only be applied once
    
    return model


def train_with_pruning(
    model: nn.Module, 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    config: Dict[str, Any],
    checkpoint_path: str = 'checkpoints/pruned_model.pth'
) -> Tuple[Dict[str, List], float, int]:
    """
    Train a model with gradual magnitude pruning.
    
    This function implements a training loop with pruning applied at specified intervals.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        config: Dictionary containing training configuration:
            - num_epochs: Maximum number of training epochs
            - criterion: Loss function
            - optimizer: PyTorch optimizer
            - scheduler: Learning rate scheduler
            - patience: Number of epochs to wait before early stopping
            - device: Device to train on
            - grad_clip_norm: Maximum norm for gradient clipping (optional)
            - initial_sparsity: Starting sparsity level (default: 0.0)
            - final_sparsity: Target final sparsity level (e.g., 0.5 for 50% sparsity)
            - start_epoch: Epoch to start pruning
            - end_epoch: Epoch to end pruning
            - pruning_frequency: Number of epochs between pruning steps
            - pruning_method: Method to use ("l1_unstructured", "global_unstructured", etc.)
            - schedule_type: Type of pruning schedule ('linear', 'exponential', 'cubic')
            - only_prune_conv: Whether to only prune Conv2d layers (default: False)
            
        checkpoint_path: Path to save the best model
        
    Returns:
        Tuple of (training_stats, best_accuracy, best_epoch)
    """
    # Extract training configuration
    num_epochs = config.get('num_epochs', 100)
    criterion = config.get('criterion')
    optimizer = config.get('optimizer')
    scheduler = config.get('scheduler')
    patience = config.get('patience', 10)
    device = config.get('device', 
                               torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    grad_clip_norm = config.get('grad_clip_norm', None)
    
    # Extract pruning configuration
    initial_sparsity = config.get('initial_sparsity', 0.0) 
    final_sparsity = config.get('final_sparsity', 0.5)
    start_epoch = config.get('start_epoch', 5)
    end_epoch = config.get('end_epoch', num_epochs - 5)
    pruning_frequency = config.get('pruning_frequency', 1)
    pruning_method = config.get('pruning_method', 'global_unstructured')
    schedule_type = config.get('schedule_type', 'cubic')
    only_prune_conv = config.get('only_prune_conv', False)
    make_permanent = config.get('make_permanent', True)
    
    # Generate sparsity schedule for all epochs
    sparsity_schedule = compute_sparsity_schedule(
        initial_sparsity=initial_sparsity,
        final_sparsity=final_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        epochs=num_epochs,
        schedule_type=schedule_type
    )
    
    # Display training configuration
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Training with gradual pruning for {num_epochs} epochs")
    print(f"Pruning schedule: {initial_sparsity:.2%} → {final_sparsity:.2%} sparsity from epoch {start_epoch} to {end_epoch}")
    
    # Training statistics
    best_accuracy = 0.0
    best_epoch = 0
    best_model = None
    training_stats = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "sparsity": [],
        "epoch_time": [],
        "lr": []
    }
    
    # Early stopping variable
    early_stop_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # TODO: Apply pruning if in pruning phase and it's a pruning frequency epoch
        # Remember to use the target sparsity for the current epoch
        # You can use the prune_model_to_target() function to update the model variable directly
        
        # Get current sparsity (for logging)
        current_sparsity = calculate_sparsity(model)
        
        # Train for one epoch
        train_loss, train_accuracy = train_single_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip_norm=grad_clip_norm, epoch=epoch, num_epochs=num_epochs,
        )
        
        # Evaluate on test set
        test_loss, test_accuracy = validate_single_epoch(
            model, test_loader, criterion, device, epoch, num_epochs
        )
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print statistics
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, "
              f"Sparsity: {current_sparsity:.2f}%, "
              f"LR: {lr:.6f}, Time: {epoch_time:.2f}s")
        
        # Save best pruned model
        if test_accuracy > best_accuracy and current_sparsity > 0:
            print(f"New best pruned model! Saving... ({test_accuracy:.2f}%)")
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            
            # Create a copy for saving to avoid modifying the training model
            best_model = copy.deepcopy(model)
                        
            # Save the model with pruning masks (PyTorch will automatically handle this)
            save_model(best_model, checkpoint_path)
                
            early_stop_counter = 0  # Reset early stopping counter
        else:
            early_stop_counter += 1
        
        # Early stopping condition
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
            break
        
        # Record statistics
        training_stats["epoch"].append(epoch + 1)
        training_stats["train_loss"].append(train_loss)
        training_stats["train_accuracy"].append(train_accuracy)
        training_stats["test_loss"].append(test_loss)
        training_stats["test_accuracy"].append(test_accuracy)
        training_stats["sparsity"].append(current_sparsity)
        training_stats["epoch_time"].append(epoch_time)
        training_stats["lr"].append(lr)
    
    # After training, post-process and save the best model
    print("Making pruning permanent on best model...")
    
    ## First, check that model is indeed pruned
    is_pruned(best_model)
                                 
    ## Make pruning permanent before saving, if model is pruned
    prunable_modules = find_prunable_modules(best_model)
    for module, name in prunable_modules:
        if hasattr(module, f'{name}_mask'):
            prune.remove(module, name)
            
    ## Save the permanently pruned model
    save_model(best_model, checkpoint_path)
    
    # Get final sparsity
    final_sparsity = calculate_sparsity(best_model)
    
    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    print(f"Best model saved as '{checkpoint_path}' at epoch {best_epoch}")
    print(f"Final model sparsity: {final_sparsity:.2f}%")
    
    return best_model, training_stats, best_accuracy, best_epoch