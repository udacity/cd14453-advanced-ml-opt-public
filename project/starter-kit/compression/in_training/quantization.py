"""
UdaciSense Project: Quantization-Aware Training Module

This module provides a quantizable MobileNetV3 model implementation for the household objects 
dataset, along with functions for quantization-aware training and model conversion.
"""

import copy
import time
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.ao.quantization
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from torchvision.models.quantization.mobilenetv3 import _mobilenet_v3_conf, _mobilenet_v3_model
from tqdm import tqdm

from utils.model import get_model_size, save_model, train_single_epoch, validate_single_epoch


class QuantizableMobileNetV3_Household(nn.Module):
    """Quantizable MobileNetV3 model for household objects dataset.
    
    This model is designed to be compatible with PyTorch's quantization features,
    including quantization-aware training (QAT).
    
    Attributes:
        model: The underlying MobileNetV3 model with a modified classifier
    """
    
    def __init__(
        self, 
        num_classes: int = 10, 
        dropout_rate: float = 0.2, 
        quantize: bool = False, 
        pretrained: bool = True
    ):
        """Initialize a quantizable MobileNetV3 model.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout probability in the classifier
            quantize: Whether to create a quantization-ready model
            pretrained: Whether to load ImageNet pretrained weights
        """
        super().__init__()
        
        # Create a quantizable MobileNetV3 Small
        inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small")
        self.model = _mobilenet_v3_model(
            inverted_residual_setting=inverted_residual_setting,
            last_channel=last_channel,
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None,
            progress=True,
            quantize=quantize,
        )
        
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
    
    def fuse_model(self, is_qat: bool = False) -> 'QuantizableMobileNetV3_Household':
        """Fuse operations like Conv+BN+ReLU for improved performance.
        
        Args:
            is_qat: Whether the fusion is for quantization-aware training
            
        Returns:
            Self with fused operations
        """
        # TODO: Fuse the model 
        return self

# TODO: Define all the steps necessary to prepare the model for QAT
# Look at built-in pytorch functionalities, wherever possible
def _prepare_qat_model(model: nn.Module, backend: str = "fbgemm") -> nn.Module:
    """Prepare model for quantization-aware training.
    
    This function performs the necessary steps to convert a regular model
    to be ready for quantization-aware training.
    
    Args:
        model: Model to prepare for QAT
        backend: Quantization backend to use ("fbgemm" or "qnnpack")
    
    Returns:
        Model prepared for QAT
    """
    pass


# TODO: Define all the steps necessary to convert the model to fully quantized
# Look at built-in pytorch functionalities, wherever possible
def _convert_qat_model_to_quantized(model: nn.Module) -> nn.Module:
    """Convert a QAT model to a fully quantized model for inference.
    
    Args:
        model: QAT-trained model
        
    Returns:
        Fully quantized model
    """
    pass


def train_model_qat(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    training_config: Dict[str, Any],
    checkpoint_path: str,
    backend: str = "fbgemm",
) -> Tuple[nn.Module, Dict[str, Any], float, int]:
    """Train a model using quantization-aware training.
    
    This function implements the complete QAT workflow, including:
    1. Initial training before QAT
    2. QAT activation and fine-tuning
    3. Observer disabling and batch norm freezing
    4. Final conversion to a fully quantized model
    
    Args:
        model: PyTorch model (should support fuse_model method)
        train_loader: Training data loader
        test_loader: Test data loader
        training_config: Dictionary containing training configuration
        checkpoint_path: Path to save the best QAT model
        backend: Quantization backend ("fbgemm" for x86, "qnnpack" for ARM)
        
    Returns:
        Tuple of (quantized_model, training_stats, best_accuracy, best_epoch)
    """
    # Step 1: Define training variables
    
    # Extract training configuration
    num_epochs = training_config.get('num_epochs', 100)
    criterion = training_config.get('criterion')
    optimizer = training_config.get('optimizer')
    scheduler = training_config.get('scheduler')
    patience = training_config.get('patience', 10)
    device = training_config.get('device', 
                                torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    grad_clip_norm = training_config.get('grad_clip_norm', None)
    freeze_bn_epochs = training_config.get('freeze_bn_epochs', 0)  # Default: don't freeze BN
    qat_start_epoch = training_config.get('qat_start_epoch', 0)  # When to start QAT
    
    print(f"Training with quantization-aware training for {num_epochs} epochs")
    print(f"QAT start epoch: {qat_start_epoch}, Finetune BN stats epochs: {freeze_bn_epochs}")
    print(f"QAT will be activated after epoch {qat_start_epoch}")
        
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
    
    # Step 2: Train the model with QAT
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Make sure model is in train mode
        model.train()
        
        # Prepare model for QAT at the start of QAT epoch
        # Think about how to update the optimizer too!
        # Save the prepared model in the model variable directly
        
        # Train for one epoch
        train_loss, train_accuracy = train_single_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip_norm=grad_clip_norm, epoch=epoch, num_epochs=num_epochs,
        )
        
        # TODO: Disable observers after sufficient QAT training to stabilize quantization parameters at your chosen epoch
        # Update the model variable in place
        
        # TODO: Freeze batch norm mean and variance estimates if the epoch matches freeze_bn_epochs
        # Update the model variable in place

        # Evaluate on test set
        if epoch >= qat_start_epoch:
            # IMPORTANT! Move model to CPU for inference
            eval_model = copy.deepcopy(model).cpu()
            eval_model.eval()
            
            # TODO: Convert to quantized model for evaluation
            # Save output in a new quantized_model variable
            
            # Evaluate quantized model
            test_loss, test_accuracy = validate_single_epoch(
                quantized_model, test_loader, criterion, torch.device("cpu"), epoch, num_epochs
            )
        else:
            # Evaluate fp32 model
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
        if test_accuracy > best_accuracy and epoch >= qat_start_epoch:
            print(f"New best quantized model! Saving... ({test_accuracy:.2f}%)")
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
    print(f"Best QAT model saved as '{checkpoint_path}' at epoch {best_epoch}")
    
    # Step 3: Convert the best QAT model to final quantized model for inference
    print("Converting best QAT model to fully quantized model...")
    model.load_state_dict(torch.load(checkpoint_path))
    quantized_model = _convert_qat_model_to_quantized(model)
    
    return quantized_model, training_stats, best_accuracy, best_epoch