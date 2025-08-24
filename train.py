import os
import sys
import numpy as np
import logging
from pathlib import Path
import json
import time
from typing import Dict, Tuple, List
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm

class MemoryOptimizedConfig:
    """Memory-optimized configuration for 4GB GPU with automatic model selection"""
    
    # Model selection - choose based on memory constraints
    # Options: 'auto', 'memory_optimized', 'compact', 'original'
    MODEL_TYPE = 'auto'  # Will auto-select based on available memory
    
    # Reduced parameters for memory efficiency
    NUM_EPOCHS = 20
    BATCH_SIZE = 32           
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP_VALUE = 0.5
    
    # SNN settings optimized for memory
    NUM_STEPS = 15            
    BETA = 0.9
    
    # Memory management
    ACCUMULATION_STEPS = 4    
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * ACCUMULATION_STEPS  
    
    # Training optimizations
    LABEL_SMOOTHING = 0.05
    USE_AMP = True           
    
    # Hardware settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEEDS = [42]             # Single seed to save memory
    
    # Memory cleanup settings
    CLEANUP_FREQUENCY = 10   
    
    # Output
    OUTPUT_DIR = Path("./models_mnist_opt")
    SAVE_CHECKPOINTS_ONLY = True  
    
    def select_model_type(self):
        """Automatically select model type based on available GPU memory"""
        if not torch.cuda.is_available():
            logger.info("  Using CPU - selecting CompactCSNN for efficiency")
            return 'compact'
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        logger.info(f" GPU Memory: {gpu_memory:.1f}GB")
        
        if gpu_memory <= 4:
            logger.info(" Low memory GPU detected - using CompactCSNN")
            return 'compact'
        elif gpu_memory <= 8:
            logger.info(" Medium memory GPU - using MemoryOptimizedCSNN")  
            return 'memory_optimized'
        else:
            logger.info(" High memory GPU - using MemoryOptimizedCSNN with higher settings")
            return 'memory_optimized'
    
    def adjust_for_model_type(self, model_type):
        """Adjust hyperparameters based on selected model"""
        if model_type == 'compact':
            # More aggressive settings for compact model
            self.BATCH_SIZE = min(self.BATCH_SIZE, 16)
            self.NUM_STEPS = min(self.NUM_STEPS, 12)
            self.ACCUMULATION_STEPS = max(self.ACCUMULATION_STEPS, 8)
            self.LEARNING_RATE = 2e-3  
            logger.info(" Adjusted settings for CompactCSNN")
        elif model_type == 'memory_optimized':
            
            logger.info(" Using standard memory-optimized settings")
        
        # Recalculate effective batch size
        self.EFFECTIVE_BATCH_SIZE = self.BATCH_SIZE * self.ACCUMULATION_STEPS
        logger.info(f"Final settings: batch={self.BATCH_SIZE}, steps={self.NUM_STEPS}, effective_batch={self.EFFECTIVE_BATCH_SIZE}")


def create_model(model_type: str, config: MemoryOptimizedConfig):
    """Create model instance based on type selection"""
    try:
        if model_type == 'compact':
            model = CompactCSNN(beta=config.BETA)
            logger.info(" Created CompactCSNN model")
        elif model_type == 'memory_optimized':
            model = MemoryOptimizedCSNN(beta=config.BETA)
            logger.info(" Created MemoryOptimizedCSNN model")
        elif model_type == 'original':
            # Try to use original CSNN if available
            model = CSNN(beta=config.BETA)
            logger.info(" Created original CSNN model")
        else:
            # Fallback to memory optimized
            model = MemoryOptimizedCSNN(beta=config.BETA)
            logger.info(" Fallback to MemoryOptimizedCSNN model")
        
        # Log model info
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f" Model parameters: {param_count:,}")
        
        # Calculate model size if method available
        if hasattr(model, 'get_model_size'):
            model_size = model.get_model_size()
            logger.info(f" Model size: {model_size:.2f} MB")
        
        return model
        
    except Exception as e:
        logger.error(f" Failed to create {model_type} model: {e}")
        logger.info("Falling back to CompactCSNN...")
        return CompactCSNN(beta=config.BETA)



# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Import Custom Modules ---
try:
    from model import CSNN, MemoryOptimizedCSNN, CompactCSNN  # Import all model variants
    from dataset import get_mnist_loaders
    logger.info("Successfully imported model variants and dataset loader")
except ImportError as e:
    logger.error(f" Error importing custom modules: {e}")
    logger.error("Make sure model.py and dataset.py exist in the same directory")
    logger.error("Required exports: CSNN (or MemoryOptimizedCSNN/CompactCSNN) and get_mnist_loaders")
    sys.exit(1)



def setup_memory_optimization():
    """Setup CUDA memory optimizations"""
    if torch.cuda.is_available():
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass
        
        
        torch.cuda.empty_cache()
        
        # Set deterministic operations for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        logger.info("Memory optimizations enabled")


def memory_cleanup():
    """Aggressive memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def calculate_snn_accuracy(output_spikes: torch.Tensor, targets: torch.Tensor) -> int:
    """Memory-efficient accuracy calculation"""
    with torch.no_grad():
        spike_counts = output_spikes.sum(dim=0)  
        _, predicted_idx = spike_counts.max(dim=1)
        correct = (predicted_idx == targets).sum().item()
    return correct


class MemoryEfficientTrainer:
    """Memory-efficient training class with gradient accumulation"""
    
    def __init__(self, model, optimizer, scheduler, loss_fn, scaler, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.scaler = scaler
        self.config = config
        
    def train_epoch(self, train_loader, epoch: int) -> Tuple[float, float]:
        """Memory-optimized training epoch"""
        self.model.train()
        train_loss_total = 0.0
        train_correct_total = 0
        train_samples_total = 0
        
        # Initialize gradient accumulation
        self.optimizer.zero_grad(set_to_none=True)
        
        train_iterator = tqdm(train_loader, desc=f"Training (Epoch {epoch})", leave=False)
        
        for batch_idx, (data, targets) in enumerate(train_iterator):
            data = data.to(self.config.DEVICE, non_blocking=True)
            targets = targets.to(self.config.DEVICE, non_blocking=True)
            
            # Forward pass with memory optimization
            with autocast(enabled=self.config.USE_AMP):
                # Process in smaller chunks if batch is large
                if data.size(0) > 16:
                    # Split batch into smaller chunks
                    chunk_size = 8
                    total_loss = 0
                    total_correct = 0
                    
                    for i in range(0, data.size(0), chunk_size):
                        chunk_data = data[i:i+chunk_size]
                        chunk_targets = targets[i:i+chunk_size]
                        
                        spk_rec = self.model(chunk_data, self.config.NUM_STEPS)
                        loss_val = self.loss_fn(spk_rec.sum(dim=0), chunk_targets)
                        loss_val = loss_val / self.config.ACCUMULATION_STEPS  # Scale for accumulation
                        
                        # Backward pass
                        self.scaler.scale(loss_val).backward()
                        
                        total_loss += loss_val.item() * self.config.ACCUMULATION_STEPS
                        total_correct += calculate_snn_accuracy(spk_rec, chunk_targets)
                        
                        # Clean up intermediate tensors
                        del spk_rec, loss_val, chunk_data, chunk_targets
                    
                else:
                    spk_rec = self.model(data, self.config.NUM_STEPS)
                    loss_val = self.loss_fn(spk_rec.sum(dim=0), targets)
                    loss_val = loss_val / self.config.ACCUMULATION_STEPS
                    
                    self.scaler.scale(loss_val).backward()
                    
                    total_loss = loss_val.item() * self.config.ACCUMULATION_STEPS
                    total_correct = calculate_snn_accuracy(spk_rec, targets)
                    
                    del spk_rec
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config.ACCUMULATION_STEPS == 0:
                
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP_VALUE)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                if isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()
                
                self.optimizer.zero_grad(set_to_none=True)
            
            #
            train_loss_total += total_loss
            train_correct_total += total_correct
            train_samples_total += len(targets)
            
            # Memory cleanup
            if batch_idx % self.config.CLEANUP_FREQUENCY == 0:
                memory_cleanup()
            
            # Update progress
            current_acc = (train_correct_total / train_samples_total) * 100
            train_iterator.set_postfix({
                'loss': f"{total_loss:.4f}", 
                'acc': f"{current_acc:.2f}%",
                'mem': f"{torch.cuda.memory_allocated()/1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
            })
            
            # Clean up batch tensors
            del data, targets
        
        # Handle remaining gradients if not divisible by accumulation steps
        if len(train_loader) % self.config.ACCUMULATION_STEPS != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP_VALUE)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        
        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_acc = (train_correct_total / train_samples_total) * 100
        
        return avg_train_loss, avg_train_acc
    
    def validate_epoch(self, val_loader) -> float:
        """Memory-optimized validation"""
        self.model.eval()
        val_correct_total = 0
        val_samples_total = 0
        
        with torch.no_grad():
            val_iterator = tqdm(val_loader, desc="Validating", leave=False)
            for batch_idx, (data, targets) in enumerate(val_iterator):
                data = data.to(self.config.DEVICE, non_blocking=True)
                targets = targets.to(self.config.DEVICE, non_blocking=True)
                
                with autocast(enabled=self.config.USE_AMP):
                    # Process validation in small chunks
                    if data.size(0) > 16:
                        total_correct = 0
                        chunk_size = 8
                        
                        for i in range(0, data.size(0), chunk_size):
                            chunk_data = data[i:i+chunk_size]
                            chunk_targets = targets[i:i+chunk_size]
                            
                            spk_rec = self.model(chunk_data, self.config.NUM_STEPS)
                            total_correct += calculate_snn_accuracy(spk_rec, chunk_targets)
                            
                            del spk_rec, chunk_data, chunk_targets
                    else:
                        spk_rec = self.model(data, self.config.NUM_STEPS)
                        total_correct = calculate_snn_accuracy(spk_rec, targets)
                        del spk_rec
                
                val_correct_total += total_correct
                val_samples_total += len(targets)
                
                # Memory cleanup
                if batch_idx % self.config.CLEANUP_FREQUENCY == 0:
                    memory_cleanup()
                
                current_acc = (val_correct_total / val_samples_total) * 100
                val_iterator.set_postfix({
                    'acc': f"{current_acc:.2f}%",
                    'mem': f"{torch.cuda.memory_allocated()/1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
                })
                
                del data, targets
        
        avg_val_acc = (val_correct_total / val_samples_total) * 100
        return avg_val_acc


def main():
    """Main function with automatic model selection and full compatibility"""
    # Initialize config
    config = MemoryOptimizedConfig()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup memory optimizations
    setup_memory_optimization()
    
    # Auto-select model type if needed
    if config.MODEL_TYPE == 'auto':
        selected_model_type = config.select_model_type()
    else:
        selected_model_type = config.MODEL_TYPE
    
    # Adjust config based on model selection
    config.adjust_for_model_type(selected_model_type)
    
    logger.info("Memory-Optimized SNN Training for MNIST")
    logger.info(f" Device: {config.DEVICE}")
    logger.info(f"Model: {selected_model_type}")
    logger.info(f"Batch size: {config.BATCH_SIZE} (effective: {config.EFFECTIVE_BATCH_SIZE})")
    logger.info(f" Timesteps: {config.NUM_STEPS}, β: {config.BETA}")
    logger.info(f" Mixed precision: {config.USE_AMP}")
    
    # Verify imports and compatibility
    logger.info("\n Checking module compatibility...")
    try:
        # Test model creation
        test_model_type = selected_model_type
        test_model = create_model(test_model_type, config)
        del test_model  # Clean up
        logger.info(" Model creation successful")
        
        # Test dataset loading  
        test_train, test_val = get_mnist_loaders(batch_size=4, num_workers=0, root="./data")
        test_batch = next(iter(test_train))
        del test_train, test_val, test_batch
        logger.info(" Dataset loading successful")
        
    except Exception as e:
        logger.error(f" Compatibility check failed: {e}")
        logger.error("\n Setup Instructions:")
        logger.error("1. Ensure model.py exports: CSNN, MemoryOptimizedCSNN, CompactCSNN")
        logger.error("2. Ensure dataset.py exports: get_mnist_loaders")
        logger.error("3. Install required packages: torch, torchvision, snntorch")
        return
    
    logger.info(" All compatibility checks passed!")
    
    # Training
    results = {}
    start_time = time.time()
    
    for seed in config.SEEDS:
        try:
            best_acc = train_model_for_seed(seed, config)
            results[seed] = best_acc
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f" OOM Error for seed {seed}")
                logger.error(f"Current settings: batch={config.BATCH_SIZE}, steps={config.NUM_STEPS}")
                
                # Emergency memory cleanup
                memory_cleanup()
                
                # Suggest even more aggressive optimizations
                logger.info("\n Emergency Memory Optimizations:")
                logger.info(f"  1. Set BATCH_SIZE = {max(8, config.BATCH_SIZE // 2)}")
                logger.info(f"  2. Set NUM_STEPS = {max(8, config.NUM_STEPS - 3)}")
                logger.info(f"  3. Set MODEL_TYPE = 'compact'")
                logger.info(f"  4. Add: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
                logger.info(f"  5. Consider training on CPU if GPU memory < 2GB free")
                break
            else:
                logger.error(f"Training failed for seed {seed}: {e}")
                logger.error("Check model and dataset compatibility")
        except Exception as e:
            logger.error(f" Unexpected error for seed {seed}: {e}", exc_info=True)
    
    total_time = time.time() - start_time
    
    # Results summary
    logger.info(f"\n{'='*70}")
    logger.info(" MEMORY-OPTIMIZED TRAINING COMPLETE")
    logger.info(f"{'='*70}")
    
    if results:
        for seed, acc in results.items():
            logger.info(f" Seed {seed}: {acc:.2f}%")
        
        best_result = max(results.values())
        avg_result = np.mean(list(results.values()))
        
        # Performance assessment with model-specific expectations
        if selected_model_type == 'compact':
            if best_result > 92.0:
                logger.info("CompactCSNN achieved great accuracy!")
            elif best_result > 85.0:
                logger.info("Strong performance for compact model")
            elif best_result > 75.0:
                logger.info(" Reasonable for memory-constrained training")
            else:
                logger.info(" Consider tuning hyperparameters")
        else:  # memory_optimized
            if best_result > 96.0:
                logger.info(" Ready for adversarial research!")
            elif best_result > 92.0:
                logger.info("Strong baseline achieved")
            elif best_result > 85.0:
                logger.info("Solid performance")
            else:
                logger.info("Consider hyperparameter tuning")
        
        logger.info(f"Best accuracy: {best_result:.2f}%")
        if len(results) > 1:
            logger.info(f"Average accuracy: {avg_result:.2f}%")
        
        # Save final summary with compatibility info
        summary = {
            'model_type': selected_model_type,
            'results': results,
            'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
            'training_time_hours': total_time / 3600,
            'best_accuracy': best_result,
            'average_accuracy': avg_result,
            'compatibility_check': 'passed'
        }
        
        try:
            with open(config.OUTPUT_DIR / 'results_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f" Results saved to: {config.OUTPUT_DIR}/results_summary.json")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    else:
        logger.error(" No successful training runs completed")
        logger.info("\n Troubleshooting Guide:")
        logger.info("1. Check GPU memory: nvidia-smi")
        logger.info("2. Try MODEL_TYPE = 'compact'")  
        logger.info("3. Reduce BATCH_SIZE to 8 or 4")
        logger.info("4. Reduce NUM_STEPS to 8-10")
        logger.info("5. Enable: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    
    logger.info(f"  Total experiment time: {total_time/60:.1f} minutes")
    logger.info(f" Next step: Use trained model for adversarial attack research!")


def train_model_for_seed(seed: int, config: MemoryOptimizedConfig) -> float:
    """Memory-optimized training for a single seed with full model compatibility"""
    # Get model type from config
    model_type = config.MODEL_TYPE
    if model_type == 'auto':
        model_type = config.select_model_type()
    
    logger.info(f"{'='*60}\nSTARTING TRAINING - SEED: {seed} | MODEL: {model_type}\n{'='*60}")
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Memory cleanup before starting
    memory_cleanup()
    
    # Load data with model-specific normalization
    try:
        normalize_method = 'zero_one' if model_type == 'compact' else 'neg_one_one'
        train_loader, test_loader = get_mnist_loaders(
            batch_size=config.BATCH_SIZE,
            num_workers=2,
            root="./data",
            pin_memory=True,
            normalize_method=normalize_method,
            augment_train=True,
            download=True
        )
        logger.info(f" Dataset loaded with {normalize_method} normalization")
    except Exception as e:
        logger.error(f" Dataset loading failed: {e}")
        raise
    
    # Create and initialize model
    net = create_model(model_type, config).to(config.DEVICE)
    
    # Count parameters and log model info
    param_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info(f"Model: {model_type} with {param_count:,} parameters")
    logger.info(f"  Training: {config.NUM_STEPS} timesteps, β={config.BETA}")
    logger.info(f" Effective batch size: {config.EFFECTIVE_BATCH_SIZE} (via accumulation)")
    
    # Initialize training components
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Scheduler selection based on model type
    if model_type == 'compact':
        # Use OneCycle for compact model
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.LEARNING_RATE,
            epochs=config.NUM_EPOCHS,
            steps_per_epoch=len(train_loader) // config.ACCUMULATION_STEPS,
            pct_start=0.2
        )
    else:
        # Use Cosine for larger models
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=config.LEARNING_RATE * 0.01
        )
    
    scaler = GradScaler(enabled=config.USE_AMP)
    
    # Initialize trainer
    trainer = MemoryEfficientTrainer(net, optimizer, scheduler, loss_fn, scaler, config)
    
    # Training loop
    best_val_accuracy = 0.0
    patience_counter = 0
    patience_limit = 7 if model_type == 'compact' else 5
    
    logger.info(f" Starting training with patience={patience_limit}")
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"\nEpoch {epoch}/{config.NUM_EPOCHS} | LR: {current_lr:.6f}")
        
        # Training
        try:
            avg_train_loss, avg_train_acc = trainer.train_epoch(train_loader, epoch)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f" OOM during training epoch {epoch}")
                memory_cleanup()
                raise e
            else:
                logger.error(f"Training error: {e}")
                raise e
        
        # Validation
        try:
            avg_val_acc = trainer.validate_epoch(test_loader)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f" OOM during validation epoch {epoch}")
                memory_cleanup()
                raise e
            else:
                logger.error(f"Validation error: {e}")
                raise e
        
        # Update scheduler (for non-OneCycle)
        if not isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        
        # Logging with memory info
        logger.info(f"   Train: Loss {avg_train_loss:.4f}, Acc {avg_train_acc:.2f}%")
        logger.info(f"   Val: Acc {avg_val_acc:.2f}%")
        logger.info(f"    Time: {epoch_time:.1f}s")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_cached = torch.cuda.memory_reserved() / 1e9
            logger.info(f"   GPU Memory: {memory_used:.1f}GB used, {memory_cached:.1f}GB cached")
        
        # Save best model
        if avg_val_acc > best_val_accuracy:
            best_val_accuracy = avg_val_acc
            patience_counter = 0
            
            # Save comprehensive checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_type': model_type,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_accuracy,
                'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
                'seed': seed,
                'normalize_method': normalize_method
            }
            
            model_filename = config.OUTPUT_DIR / f"best_model_for_{seed}_{model_type}.pth"
            torch.save(checkpoint, model_filename)
            logger.info(f"   New best saved: {best_val_accuracy:.2f}%")
        else:
            patience_counter += 1
        
        # Progress assessment with model-specific expectations
        if epoch >= 5:
            if model_type == 'compact':
                if avg_val_acc < 40.0:
                    logger.warning(f"    CompactCSNN: Low accuracy ({avg_val_acc:.1f}%) - check encoding")
                elif avg_val_acc > 85.0:
                    logger.info(f"  CompactCSNN: Excellent progress ({avg_val_acc:.1f}%)!")
            else:  # memory_optimized
                if avg_val_acc < 50.0:
                    logger.warning(f"  Low accuracy ({avg_val_acc:.1f}%) - check hyperparameters")
                elif avg_val_acc > 90.0:
                    logger.info(f"  Great progress: {avg_val_acc:.1f}%!")
        
        # Early stopping
        if patience_counter >= patience_limit:
            logger.info(f"  Early stopping triggered after {patience_limit} epochs without improvement")
            break
        
        # Aggressive memory cleanup
        memory_cleanup()
    
    # Final cleanup
    memory_cleanup()
    
    logger.info(f"\nTraining complete for seed {seed}")
    logger.info(f"   Model: {model_type}")
    logger.info(f"   Best accuracy: {best_val_accuracy:.2f}%")
    logger.info(f"   Final epoch: {epoch}")
    return best_val_accuracy

if __name__ == "__main__":
    main()