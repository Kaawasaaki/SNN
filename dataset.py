# dataset.py - MNIST Dataset Loader optimized for SNN training

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def get_mnist_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    root: str = "./data",
    download: bool = True,
    val_split: float = 0.0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    augment_train: bool = True,
    normalize_method: str = 'standard'
) -> Tuple[DataLoader, DataLoader]:
    """
    Create MNIST data loaders optimized for SNN training
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        root: Root directory for dataset
        download: Whether to download dataset if not found
        val_split: Fraction of training data to use for validation (0.0 = use test set)
        pin_memory: Pin memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs
        augment_train: Apply data augmentation to training set
        normalize_method: 'standard', 'zero_one', or 'neg_one_one'
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    # Create data directory
    data_path = Path(root)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Define transforms based on normalization method
    if normalize_method == 'standard':
        # Standard MNIST normalization (mean=0.1307, std=0.3081)
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        logger.info("Using standard MNIST normalization")
    elif normalize_method == 'zero_one':
        # Normalize to [0, 1] - good for Poisson encoding
        normalize = transforms.Lambda(lambda x: x)  # ToTensor already gives [0,1]
        logger.info("Using [0,1] normalization for Poisson encoding")
    else:  # neg_one_one
        # Normalize to [-1, 1]
        normalize = transforms.Normalize((0.5,), (0.5,))
        logger.info("Using [-1,1] normalization")
    
    # Training transforms with optional augmentation
    train_transforms = [
        transforms.ToTensor(),
    ]
    
    if augment_train:
        # Light augmentation suitable for MNIST and SNN
        train_transforms.extend([
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        ])
        logger.info("Training augmentation enabled")
    
    train_transforms.append(normalize)
    
    # Test/validation transforms (no augmentation)
    test_transforms = [
        transforms.ToTensor(),
        normalize
    ]
    
    # Create transform compositions
    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose(test_transforms)
    
    # Load datasets
    logger.info(f"Loading MNIST dataset from {root}")
    
    train_dataset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=download,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=root,
        train=False,
        download=download,
        transform=test_transform
    )
    
    logger.info(f" Loaded MNIST: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    # Handle validation split
    if val_split > 0.0:
        # Split training set into train/val
        train_size = int((1 - val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_subset, val_dataset = random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f" Split training: {train_size} train, {val_size} validation")
        final_train_dataset = train_subset
        final_val_dataset = val_dataset
    else:
        # Use full training set for training, test set for validation
        final_train_dataset = train_dataset
        final_val_dataset = test_dataset
        logger.info(" Using full train set + test set as validation")
    
    # Adjust num_workers for memory constraints
    if num_workers > 2 and torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory <= 4:
            num_workers = min(num_workers, 2)
            logger.info(f" Reduced workers to {num_workers} for memory optimization")
    
    # Create data loaders
    train_loader = DataLoader(
        final_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=True  # Ensure consistent batch sizes for SNN
    )
    
    val_loader = DataLoader(
        final_val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for validation
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=False
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"   Train: {len(train_loader)} batches of size {batch_size}")
    logger.info(f"   Val:   {len(val_loader)} batches")
    logger.info(f"   Workers: {num_workers}, Pin memory: {pin_memory}")
    
    return train_loader, val_loader


def get_sample_data(num_samples: int = 8, normalize_method: str = 'standard') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a small sample of MNIST data for testing/debugging
    
    Args:
        num_samples: Number of samples to return
        normalize_method: Same as get_mnist_loaders
        
    Returns:
        Tuple of (data, targets)
    """
    # Quick transform based on normalization
    if normalize_method == 'standard':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif normalize_method == 'zero_one':
        transform = transforms.ToTensor()
    else:  # neg_one_one
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    samples = [dataset[i] for i in indices]
    
    data = torch.stack([s[0] for s in samples])
    targets = torch.tensor([s[1] for s in samples])
    
    return data, targets


def analyze_dataset_statistics(root: str = "./data"):
    """Analyze MNIST dataset statistics for optimal preprocessing"""
    
    logger.info("Analyzing MNIST dataset statistics...")
    
    # Load raw dataset (just ToTensor, no normalization)
    raw_transform = transforms.ToTensor()
    
    train_dataset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=raw_transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=root,
        train=False,
        download=False,
        transform=raw_transform
    )
    
    # Sample data for statistics
    sample_size = min(1000, len(train_dataset))
    indices = np.random.choice(len(train_dataset), sample_size, replace=False)
    
    data_samples = []
    for i in indices:
        data_samples.append(train_dataset[i][0])
    
    data_tensor = torch.stack(data_samples)
    
    # Calculate statistics
    mean = data_tensor.mean()
    std = data_tensor.std()
    min_val = data_tensor.min()
    max_val = data_tensor.max()
    
    logger.info(f" Dataset Statistics (sample of {sample_size}):")
    logger.info(f"   Mean: {mean:.4f}")
    logger.info(f"   Std:  {std:.4f}")
    logger.info(f"   Min:  {min_val:.4f}")
    logger.info(f"   Max:  {max_val:.4f}")
    
    # Class distribution
    train_targets = []
    for i in range(min(10000, len(train_dataset))):
        train_targets.append(train_dataset[i][1])
    
    unique, counts = np.unique(train_targets, return_counts=True)
    logger.info(f" Class distribution (sample):")
    for cls, count in zip(unique, counts):
        logger.info(f"   Class {cls}: {count} samples")
    
    # Recommendations
    logger.info(f"\n Normalization Recommendations:")
    logger.info(f"   Standard MNIST: Normalize((0.1307,), (0.3081,))")
    logger.info(f"   For Poisson encoding: Use [0,1] range (no normalization after ToTensor)")
    logger.info(f"   For symmetric range: Normalize((0.5,), (0.5,)) → [-1,1]")


def create_subset_loaders(
    train_samples: int = 5000,
    test_samples: int = 1000,
    **loader_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create smaller MNIST subset loaders for quick experimentation
    
    Args:
        train_samples: Number of training samples to use
        test_samples: Number of test samples to use
        **loader_kwargs: Arguments passed to get_mnist_loaders
        
    Returns:
        Tuple of (train_loader, test_loader) with reduced data
    """
    
    logger.info(f" Creating MNIST subset: {train_samples} train, {test_samples} test")
    
    # Get full loaders first
    full_train_loader, full_test_loader = get_mnist_loaders(**loader_kwargs)
    
    # Create subsets
    train_dataset = full_train_loader.dataset
    test_dataset = full_test_loader.dataset
    
    # Random indices for subsets
    train_indices = np.random.choice(len(train_dataset), train_samples, replace=False)
    test_indices = np.random.choice(len(test_dataset), test_samples, replace=False)
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    # Create new loaders
    batch_size = loader_kwargs.get('batch_size', 128)
    num_workers = loader_kwargs.get('num_workers', 4)
    pin_memory = loader_kwargs.get('pin_memory', True)
    
    subset_train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True
    )
    
    subset_test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False
    )
    
    logger.info(f" Subset loaders created: {len(subset_train_loader)} train batches, {len(subset_test_loader)} test batches")
    
    return subset_train_loader, subset_test_loader


if __name__ == "__main__":
    print(" MNIST Dataset Module Test")
    
    # Test dataset loading
    print("\n--- Testing Standard Loader ---")
    train_loader, test_loader = get_mnist_loaders(
        batch_size=64,
        num_workers=2,
        augment_train=True,
        normalize_method='standard'
    )
    
    # Test batch
    data_batch, target_batch = next(iter(train_loader))
    print(f" Batch shape: {data_batch.shape}, targets: {target_batch.shape}")
    print(f"   Data range: [{data_batch.min():.3f}, {data_batch.max():.3f}]")
    print(f"   Target classes: {sorted(target_batch.unique().tolist())}")
    
    # Test different normalization methods
    print("\n--- Testing Normalization Methods ---")
    methods = ['standard', 'zero_one', 'neg_one_one']
    
    for method in methods:
        train_loader_norm, _ = get_mnist_loaders(
            batch_size=32,
            num_workers=0,
            normalize_method=method,
            augment_train=False
        )
        
        data_batch, _ = next(iter(train_loader_norm))
        print(f"{method:12}: range [{data_batch.min():.3f}, {data_batch.max():.3f}], "
              f"mean {data_batch.mean():.3f}, std {data_batch.std():.3f}")
    
    # Test subset loader
    print("\n--- Testing Subset Loader ---")
    subset_train, subset_test = create_subset_loaders(
        train_samples=1000,
        test_samples=200,
        batch_size=32,
        num_workers=0
    )
    
    print(f" Subset: {len(subset_train)} train batches, {len(subset_test)} test batches")
    
    # Dataset statistics
    print("\n--- Dataset Statistics ---")
    analyze_dataset_statistics()
    
    print("\n All tests passed! Dataset module ready for SNN training.")
    
    # SNN-specific recommendations
    print("\n SNN Training Recommendations:")
    print("   • Use normalize_method='zero_one' for Poisson encoding")
    print("   • Use normalize_method='neg_one_one' if your model expects [-1,1]")
    print("   • Enable augment_train=True for better generalization")
    print("   • Use drop_last=True for consistent SNN timestep processing")
    print("   • Consider val_split=0.1 if you need a proper validation set")
    
