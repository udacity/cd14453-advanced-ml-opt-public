"""
UdaciSense Project: Model Utilities Module

This module provides classes and functions for creating, loading, and training models
for the household objects dataset.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Tuple, Optional, List, Callable, Type


# ------ MODEL CLASSES ------ #

class MobileNetV3_Household(nn.Module):
    """MobileNetV3-Small model for household objects dataset.
    
    This model adapts the pretrained MobileNetV3-Small architecture for
    the household objects classification task.
    
    Attributes:
        model: The underlying MobileNetV3 model with a modified classifier
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.2):
        """Initialize a MobileNetV3-Small model for household objects.
        
        Args:
            num_classes: Number of output classes (10 for household objects)
            dropout_rate: Dropout probability in the classifier
        """
        super().__init__()
        
        # Load the MobileNetV3-Small model with pretrained weights
        self.model = models.mobilenet_v3_small(weights="DEFAULT")
        
        # Modify the classifier for the household objects dataset
        last_channel = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(last_channel, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Output tensor of shape [B, num_classes]
        """
        # Resize the image to the format expected by MobileNetV3
        x = torch.nn.functional.interpolate(
            x, size=(224, 224), mode='bilinear', align_corners=False
        )
        return self.model(x)


# ------ UTILITY METHODS ------ #

def count_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(
    path: str, 
    device: str = 'cpu',
    model_class: Type[nn.Module] = MobileNetV3_Household,
    **model_kwargs,
) -> nn.Module:
    """Load a model from a checkpoint file with correct initialization parameters.
    
    Args:
        path: Path to saved model file.
        device: Device to load model on ('cpu' or 'cuda').
        model_class: The model class to instantiate and load weights into.
        model_kwargs: Additional keyword arguments for model initialization.

    Returns:
        Loaded PyTorch model.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    # Load checkpoint
    loaded = torch.load(path, map_location=device)

    # If it's a GraphModule, return it directly
    if hasattr(loaded, 'graph') and hasattr(loaded, '_modules'):
        return loaded.to(device) if device else loaded

    # If it's a state_dict, load it into a new model instance
    if isinstance(loaded, dict) and 'state_dict' not in loaded:

        # Instantiate the model with correct parameters
        model = model_class(**model_kwargs)

        # Load weights
        model.load_state_dict(loaded)
        return model.to(device)
    
    raise ValueError("Unsupported checkpoint format.")
    

def save_model(model: nn.Module, path: str) -> None:
    """Save a model to the given path.
    
    Args:
        model: PyTorch model
        path: Path to save model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Special handling for FX-optimized models
    if hasattr(model, '_modules') and hasattr(model, 'graph'):
        torch.save(model, path)
    else:
        torch.save(model.state_dict(), path)

    print(f"Model saved to {path}")


def get_model_size(
    model: nn.Module, 
    filename: Optional[str] = None
) -> float:
    """Get the size of a model in megabytes.
    
    Args:
        model: PyTorch model
        filename: If provided, save the model to this file before measuring
        
    Returns:
        Size of the model in megabytes
    """
    if filename:
        torch.save(model.state_dict(), filename)
        size_bytes = os.path.getsize(filename)
    else:
        # Create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile() as f:
            torch.save(model.state_dict(), f.name)
            size_bytes = os.path.getsize(f.name)
    
    return size_bytes / (1024 * 1024)  # Convert to MB


def print_model_summary(model: nn.Module) -> None:
    """Print a summary of model architecture and parameters.
    
    Args:
        model: PyTorch model
    """
    print(f"Model Architecture:")
    print(f"{model}\n")
    
    print(f"Total Parameters: {count_parameters(model):,}")


# ------ TRAINING METHODS ------ #

def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    training_config: Dict[str, Any], 
    checkpoint_path: str = 'checkpoints/baseline_model.pth'
) -> Tuple[Dict[str, List], float, int]:
    """Standard training method for the model.
    
    This function implements a standard training loop with early stopping
    and model checkpointing.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        training_config: Dictionary containing training configuration:
            - num_epochs: Maximum number of training epochs
            - criterion: Loss function
            - optimizer: PyTorch optimizer
            - scheduler: Learning rate scheduler
            - patience: Number of epochs to wait before early stopping
            - device: Device to train on
            - grad_clip_norm: Maximum norm for gradient clipping (optional)
        checkpoint_path: Path to save the best model
        
    Returns:
        Tuple of (training_stats, best_accuracy, best_epoch)
    """
    # Extract training configuration
    num_epochs = training_config.get('num_epochs', 100)
    criterion = training_config.get('criterion')
    optimizer = training_config.get('optimizer')
    scheduler = training_config.get('scheduler')
    patience = training_config.get('patience', 10)
    device = training_config.get('device', 
                               torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    grad_clip_norm = training_config.get('grad_clip_norm', None)
    
    # Display training configuration
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Training with standard method for {num_epochs} epochs")
    
    # Training statistics
    best_accuracy = 0.0
    best_epoch = 0
    training_stats = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "epoch_time": [],
        "lr": []
    }
    
    # Early stopping variable
    early_stop_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
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
              f"LR: {lr:.6f}, Time: {epoch_time:.2f}s")
        
        # Save best model
        if test_accuracy > best_accuracy:
            print(f"New best model! Saving... ({test_accuracy:.2f}%)")
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            
            save_model(model, checkpoint_path)
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
        training_stats["epoch_time"].append(epoch_time)
        training_stats["lr"].append(lr)
    
    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    print(f"Best model saved as '{checkpoint_path}' at epoch {best_epoch}")
    
    return training_stats, best_accuracy, best_epoch


def train_single_epoch(
    model: nn.Module, 
    train_loader: DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device, 
    grad_clip_norm: Optional[float] = None, 
    epoch: Optional[int] = None, 
    num_epochs: Optional[int] = None,
    augmentation_fn: Optional[Callable] = None
) -> Tuple[float, float]:
    """Train a model for a single epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: PyTorch optimizer
        device: Device to train on
        grad_clip_norm: Maximum norm for gradient clipping (optional)
        epoch: Current epoch (optional, for progress display)
        num_epochs: Total number of epochs (optional, for progress display)
        augmentation_fn: Optional function for applying data augmentation (e.g., mixup, cutmix)
        
    Returns:
        Tuple of (train_loss, train_accuracy)
    """
    # Set model to training mode
    model.train()
    
    # Initialize tracking variables
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    # Progress bar description
    if epoch is not None and num_epochs is not None:
        desc = f"Epoch {epoch+1}/{num_epochs} [Train]"
    else:
        desc = "Training"
        
    pbar = tqdm(train_loader, desc=desc)
    
    # Iterate over batches
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply data augmentation if provided
        if augmentation_fn is not None:
            inputs, targets_a, targets_b, lam = augmentation_fn(inputs, targets)
            mixed_targets = True
        else:
            mixed_targets = False
            
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Compute loss
        if mixed_targets:
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping if configured
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            
        # Update weights
        optimizer.step()
        
        # Update statistics for the entire epoch
        train_loss += loss.item() * inputs.size(0)  # Weight by batch size
        
        # Update accuracy metrics
        _, predicted = outputs.max(1)
        batch_total = targets.size(0)
        
        if mixed_targets:
            batch_correct = (lam * predicted.eq(targets_a).sum().float() + 
                          (1 - lam) * predicted.eq(targets_b).sum().float()).item()
        else:
            batch_correct = predicted.eq(targets).sum().item()
            
        train_correct += batch_correct
        train_total += batch_total
        
        # Update progress bar with current batch stats
        pbar.set_postfix({
            "loss": loss.item(),
            "batch_acc": 100. * batch_correct / batch_total,
            "running_acc": 100. * train_correct / train_total,
            "lr": optimizer.param_groups[0]['lr']
        })
    
    # Calculate final metrics for training - average over all samples
    train_loss = train_loss / train_total
    train_accuracy = 100. * train_correct / train_total
    
    return train_loss, train_accuracy


def validate_single_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device, 
    epoch: Optional[int] = None, 
    num_epochs: Optional[int] = None
) -> Tuple[float, float]:
    """Evaluate a model on a validation/test set.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to run evaluation on
        epoch: Current epoch (optional, for progress display)
        num_epochs: Total number of epochs (optional, for progress display)
        
    Returns:
        Tuple of (loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Prepare progress bar with epoch info if provided
    if epoch is not None and num_epochs is not None:
        desc = f"Epoch {epoch+1}/{num_epochs} [Test]"
    else:
        desc = "Evaluating"
        
    # No gradients needed for evaluation
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Print statistics
            pbar.set_postfix({
                "loss": total_loss / (pbar.n + 1),
                "acc": 100. * correct / total
            })
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy