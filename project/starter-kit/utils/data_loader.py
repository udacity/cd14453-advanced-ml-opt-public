"""
UdaciSense Project: Data Utilities Module

This module provides functions for loading and preprocessing the household objects dataset
(a subset of CIFAR-100) for model training and evaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL.Image import BICUBIC
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Optional, Union, Literal

# ------ CONSTANTS ------ #

# UdaciSense classes (household objects from CIFAR-100)
HOUSEHOLD_CLASSES = ('clock', 'keyboard', 'lamp', 'telephone', 'television', 
                     'bed', 'chair', 'couch', 'table', 'wardrobe')

# Data directory
DATA_DIR = "../data"

# Map from class names to CIFAR-100 indices
CLASS_TO_IDX = {
    'clock': 22,
    'keyboard': 39,
    'lamp': 40,
    'telephone': 86,
    'television': 87,
    'bed': 5,
    'chair': 20,
    'couch': 25,
    'table': 84,
    'wardrobe': 94
}

# Map from our indices to CIFAR-100 indices
IDX_TO_CIFAR100 = [CLASS_TO_IDX[cls] for cls in HOUSEHOLD_CLASSES]

# Normalization constants (ImageNet mean and std)
DATA_MEAN = (0.485, 0.456, 0.406)
DATA_STD = (0.229, 0.224, 0.225)


# ------ DATASET CLASS ------ #

class HouseholdDataset(Dataset):
    """Custom dataset class for household objects from CIFAR-100.
    
    This dataset wraps the subset of CIFAR-100 that contains household objects.
    
    Attributes:
        data: Image data as numpy array of shape [N, H, W, C]
        targets: Class labels (0-9)
        transform: Transform to apply to the images
        classes: List of class names
        class_to_idx: Mapping from class names to indices
    """
    
    def __init__(
        self, 
        data: np.ndarray, 
        targets: np.ndarray, 
        transform: Optional[transforms.Compose] = None
    ):
        """Initialize the dataset.
        
        Args:
            data: Image data (numpy array of shape [N, H, W, C])
            targets: Class labels (0-9)
            transform: Transform to apply to the images
        """
        self.data = data
        self.targets = targets
        self.transform = transform
        self.classes = HOUSEHOLD_CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(HOUSEHOLD_CLASSES)}
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Tuple of (image, label)
        """
        img = self.data[idx]
        target = self.targets[idx]
        
        # Convert to PIL image
        img = Image.fromarray(img)
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
            
        return img, target
    
# ------ TRANSFORM FUNCTIONS ------ #

def get_transforms(
    image_size: Literal["CIFAR", "IMAGENET"] = "CIFAR", 
    train: bool = True, 
    augment: bool = True
) -> transforms.Compose:
    """Get image transforms for the dataset.
    
    Args:
        image_size: Either "CIFAR" (32x32) or "IMAGENET" (224x224)
        train: Whether the transforms are for the training set
        augment: Whether to apply data augmentation
        
    Returns:
        Composition of transforms
        
    Raises:
        ValueError: If image_size is not "CIFAR" or "IMAGENET"
    """
    # Validate image size
    if image_size not in ["CIFAR", "IMAGENET"]:
        raise ValueError(f"Invalid image_size: '{image_size}'. Use 'CIFAR' or 'IMAGENET'")
    
    # Set input size based on configuration
    input_size = get_input_size(image_size, "dimension-only")
    
    transform_list = []
    
    # Apply appropriate transformations based on configuration
    if image_size == "IMAGENET":
        if train and augment:
            transform_list.extend([
                transforms.Resize(int(input_size * 1.14), BICUBIC),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip()
            ])
        else:
            transform_list.append(transforms.Resize(input_size, BICUBIC))
    elif train and augment:  # CIFAR size with augmentation
        transform_list.extend([
            transforms.RandomCrop(input_size, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip()
        ])
    
    # Add tensor conversion and normalization (common for all configurations)
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEAN, DATA_STD)
    ])
    
    return transforms.Compose(transform_list)

# ------ DATA EXTRACTION AND LOADING ------ #

def extract_household_from_cifar100(
    root: str = f"{DATA_DIR}/complete", 
    train: bool = True, 
    download: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract household objects data from CIFAR-100 dataset.
    
    Args:
        root: Root directory for the CIFAR-100 dataset
        train: Whether to extract from training set or test set
        download: Whether to download the dataset if not found
        
    Returns:
        Tuple of (data, targets) as numpy arrays
    """
    print(f"Extracting household classes from CIFAR100 for {'train' if train else 'test'} set...")
    # Load CIFAR-100 dataset without transforms
    dataset = torchvision.datasets.CIFAR100(
        root=root, train=train, download=download, transform=None)
    
    # Extract data and targets for household object classes
    selected_data = []
    selected_targets = []
    
    for img, label in zip(dataset.data, dataset.targets):
        if label in IDX_TO_CIFAR100:
            selected_data.append(img)
            # Remap class index to our consecutive indices (0-9)
            selected_targets.append(IDX_TO_CIFAR100.index(label))
    
    return np.array(selected_data), np.array(selected_targets)

def save_household_subset(
    data: np.ndarray, 
    targets: np.ndarray, 
    train: bool = True, 
    subset_dir: str = f"{DATA_DIR}"
) -> None:
    """Save the household subset to disk as individual image files.
    
    Args:
        data: Image data array of shape [N, H, W, C]
        targets: Class targets array
        train: Whether this is the training set or test set
        subset_dir: Directory to save the images
    """
    # Create directory if it doesn't exist
    os.makedirs(subset_dir, exist_ok=True)
    
    # Create a directory for each class
    for class_idx, class_name in enumerate(HOUSEHOLD_CLASSES):
        class_dir = os.path.join(subset_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # Save each image in its class directory
    for idx, (img_data, target) in enumerate(zip(data, targets)):
        class_name = HOUSEHOLD_CLASSES[target]
        img = Image.fromarray(img_data)
        
        # Save image with a unique filename
        img_filename = f"{class_name}_{idx}.png"
        img_path = os.path.join(subset_dir, class_name, img_filename)
        img.save(img_path)
    
    print(f"Saved {len(data)} images to {subset_dir}")

def get_household_dataset(
    image_size: Literal["CIFAR", "IMAGENET"] = "CIFAR", 
    train: bool = True, 
    augment: bool = True, 
    create_if_missing: bool = True
) -> HouseholdDataset:
    """Get household objects dataset.
    
    Args:
        image_size: Either "CIFAR" (32x32) or "IMAGENET" (224x224)
        train: Whether to load the training set
        augment: Whether to apply data augmentation
        create_if_missing: Whether to create the image dataset if missing
        
    Returns:
        HouseholdDataset object
    """
    transform = get_transforms(image_size, train, augment)
    
    # Determine the directory path
    split_name = 'train' if train else 'test'
    dataset_dir = os.path.join(f"{DATA_DIR}/household_images", split_name)
    
    # Extract data from CIFAR-100
    data, targets = extract_household_from_cifar100(
        root=f"{DATA_DIR}/complete", train=train, download=True)
    
    # Save as individual images if requested
    if (not os.path.exists(dataset_dir) or not os.listdir(dataset_dir)) and create_if_missing:
        print(f"Saving the household subset as raw images at {dataset_dir}.")
        save_household_subset(data, targets, train, subset_dir=dataset_dir)
       
    return HouseholdDataset(data, targets, transform)

def get_household_loaders(
    image_size: Literal["CIFAR", "IMAGENET"] = "CIFAR", 
    batch_size: int = 128, 
    num_workers: int = 2, 
    create_if_missing: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for household objects dataset.
    
    Args:
        image_size: Either "CIFAR" (32x32) or "IMAGENET" (224x224)
        batch_size: Batch size for the data loaders
        num_workers: Number of worker processes for data loading
        create_if_missing: Whether to create the image dataset if missing
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Get datasets
    train_dataset = get_household_dataset(
        image_size=image_size, 
        train=True, 
        augment=True,
        create_if_missing=create_if_missing,
    )
    
    test_dataset = get_household_dataset(
        image_size=image_size, 
        train=False, 
        augment=False,
        create_if_missing=create_if_missing,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

# ------ VISUALIZATION FUNCTIONS ------ #

def visualize_batch(dataloader: DataLoader, num_images: int = 8) -> None:
    """Visualize a batch of images from a dataloader.
    
    Args:
        dataloader: PyTorch DataLoader
        num_images: Number of images to visualize
    """
    # Get a batch of images and labels
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Unnormalize images
    images_np = images.numpy().transpose((0, 2, 3, 1))
    mean = np.array(DATA_MEAN)
    std = np.array(DATA_STD)
    images_np = std * images_np + mean
    images_np = np.clip(images_np, 0, 1)
    
    # Plot images
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(images_np[i])
        ax.set_title(HOUSEHOLD_CLASSES[labels[i]])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def print_dataloader_stats(dataloader: DataLoader, dataset_type: str = "Dataset") -> None:
    """Print statistics about the dataset.
    
    Args:
        dataloader: DataLoader object
        dataset_type: String describing the dataset type
    """
    print(f"Statistics for {dataset_type}")
    print(f" Size: {len(dataloader.dataset)}")
    
    # Count samples per class
    if hasattr(dataloader.dataset, 'targets'):
        class_counts = {}
        for target in dataloader.dataset.targets:
            class_counts[target] = class_counts.get(target, 0) + 1
        
        print(" Samples per class:")
        class_names = dataloader.dataset.classes
        for i in range(len(class_counts)):
            count = class_counts.get(i, 0)
            print(f"  {class_names[i]}: {count}")

def get_input_size(
    image_size: Literal["CIFAR", "IMAGENET"], 
    size_format: Literal["complete", "dimension-only"] = "complete"
) -> Union[Tuple[int, int, int, int], int]:
    """Get the input size for the model based on the image size configuration.
    
    Args:
        image_size: String representing the image size configuration
        size_format: Format of the output:
                     - 'complete' for full tensor shape [B, C, H, W]
                     - 'dimension-only' for single dimension (H or W)
        
    Returns:
        Input size as either a tuple (batch, channels, height, width) or a single integer
        
    Raises:
        ValueError: If size_format or image_size is invalid
    """
    # Validate inputs
    if size_format not in ["complete", "dimension-only"]:
        raise ValueError(f"Invalid size_format: '{size_format}'. Use 'complete' or 'dimension-only'")
        
    if image_size not in ["CIFAR", "IMAGENET"]:
        raise ValueError(f"Invalid image_size: '{image_size}'. Use 'CIFAR' or 'IMAGENET'")
    
    input_dim = 32 if image_size == "CIFAR" else 224
    
    if size_format == "complete":
        return (1, 3, input_dim, input_dim)
    else:
        return input_dim