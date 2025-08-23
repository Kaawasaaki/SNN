# model.py - Memory-Optimized CSNN for 4GB GPU

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import math
import torch.nn.functional as F

class MemoryOptimizedCSNN(nn.Module):
    """Memory-optimized Convolutional SNN for MNIST with 4GB GPU constraint"""
    
    def __init__(self, beta=0.9, slope=25, num_classes=10, dropout_rate=0.1):
        super().__init__()
        self.spike_grad = surrogate.fast_sigmoid(slope=slope)
        self.beta = beta
        self.dropout_rate = dropout_rate
        
        # --- Memory-Optimized Architecture for MNIST ---
        # Reduced channel sizes to save memory
        
        # First conv block - lighter start
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Second conv block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)  # Stride for downsampling
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Third conv block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)  # Stride for downsampling
        self.bn3 = nn.BatchNorm2d(128)
        self.lif3 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # Fourth conv block - final feature extraction
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.lif4 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
        
        # Global average pooling to reduce parameters
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier with reduced hidden size
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 128, bias=False)  # Smaller hidden layer
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.lif5 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
        self.dropout_fc = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, num_classes, bias=False)
        self.lif6 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
        
        # Initialize weights
        self._initialize_weights()
        
        # Calculate expected feature map sizes for verification
        self._calculate_sizes()
    
    def _calculate_sizes(self):
        """Calculate and log expected tensor sizes"""
        # MNIST: 28x28
        # After conv1: 28x28 (32 channels)
        # After conv2 (stride=2): 14x14 (64 channels)  
        # After conv3 (stride=2): 7x7 (128 channels)
        # After conv4: 7x7 (256 channels)
        # After global_avg_pool: 1x1 (256 channels)
        
        print("Expected tensor sizes:")
        print(f"  Input: [B, 1, 28, 28]")
        print(f"  After conv1: [B, 32, 28, 28]")  
        print(f"  After conv2: [B, 64, 14, 14]")
        print(f"  After conv3: [B, 128, 7, 7]")
        print(f"  After conv4: [B, 256, 7, 7]")
        print(f"  After global_avg_pool: [B, 256, 1, 1]")
        print(f"  After flatten: [B, 256]")
        print(f"  After fc1: [B, 128]")
        print(f"  Final output: [B, 10]")
    
    def _initialize_weights(self):
        """Initialize weights for better SNN performance"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, num_steps):
        """Memory-efficient forward pass"""
        batch_size = x.size(0)
        device = x.device
        
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()  
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()
        
        # Input preprocessing - normalize to [0,1] for Poisson encoding
        if x.min() < 0:  # If data is normalized (e.g., [-1,1] or zero-mean)
            x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        else:
            x_norm = torch.clamp(x, 0, 1)
        
        # Pre-generate random tensors to save memory during timestep loop
        base_random = torch.rand_like(x_norm)
        
        # Collect output spikes - use list for memory efficiency
        spk_rec = []
        
        for step in range(num_steps):
            # Generate Poisson spikes with temporal variation
            step_factor = 0.8 + 0.4 * (step / num_steps)  # Increase probability over time
            step_random = step_factor * base_random + (1 - step_factor) * torch.rand_like(base_random)
            step_random = torch.clamp(step_random, 0, 1)
            x_spikes = (step_random < x_norm).float()
            
            # Forward pass through network
            # Conv Block 1
            cur1 = self.conv1(x_spikes)
            cur1 = self.bn1(cur1)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.dropout1(spk1)
            
            # Conv Block 2 (with stride=2 downsampling)
            cur2 = self.conv2(spk1)
            cur2 = self.bn2(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.dropout2(spk2)
            
            # Conv Block 3 (with stride=2 downsampling)
            cur3 = self.conv3(spk2)
            cur3 = self.bn3(cur3)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3 = self.dropout3(spk3)
            
            # Conv Block 4
            cur4 = self.conv4(spk3)
            cur4 = self.bn4(cur4)
            spk4, mem4 = self.lif4(cur4, mem4)
            
            # Global average pooling
            pooled = self.global_avg_pool(spk4)  # [B, 256, 1, 1]
            flat = self.flatten(pooled)          # [B, 256]
            
            # Fully connected layers
            cur5 = self.fc1(flat)
            cur5 = self.bn_fc1(cur5)
            spk5, mem5 = self.lif5(cur5, mem5)
            spk5 = self.dropout_fc(spk5)
            
            cur6 = self.fc2(spk5)
            spk6, mem6 = self.lif6(cur6, mem6)
            
            spk_rec.append(spk6)
            
            # Clean up intermediate tensors to save memory
            del cur1, cur2, cur3, cur4, cur5, cur6
            del spk1, spk2, spk3, spk4, spk5
            del x_spikes, step_random
        
        # Stack output spikes
        output = torch.stack(spk_rec, dim=0)  # [num_steps, batch_size, num_classes]
        
        return output
    
    def get_model_size(self):
        """Calculate model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Alternative even more memory-efficient version
class CompactCSNN(nn.Module):
    """Ultra-compact SNN for extremely memory-constrained environments"""
    
    def __init__(self, beta=0.9, slope=25, num_classes=10):
        super().__init__()
        self.spike_grad = surrogate.fast_sigmoid(slope=slope)
        self.beta = beta
        
        # Minimal architecture
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2, bias=False)  # 28->14
        self.bn1 = nn.BatchNorm2d(16)
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2, bias=False)  # 14->7
        self.bn2 = nn.BatchNorm2d(32)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
        
        # Global average pooling eliminates most parameters
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Minimal classifier
        self.fc = nn.Linear(32, num_classes, bias=False)
        self.lif3 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, num_steps):
        batch_size = x.size(0)
        
        # Initialize membranes
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        # Normalize input
        x_norm = torch.clamp((x + 1) / 2, 0, 1)
        base_random = torch.rand_like(x_norm)
        
        spk_rec = []
        
        for step in range(num_steps):
            # Simple Poisson encoding
            step_random = 0.5 * base_random + 0.5 * torch.rand_like(base_random)
            x_spikes = (step_random < x_norm).float()
            
            # Forward pass
            cur1 = self.bn1(self.conv1(x_spikes))
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.bn2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Global pooling and classification
            pooled = self.global_pool(spk2)
            flat = self.flatten(pooled)
            
            cur3 = self.fc(flat)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            spk_rec.append(spk3)
        
        return torch.stack(spk_rec, dim=0)


# Alias for backward compatibility
CSNN = MemoryOptimizedCSNN


def test_memory_usage():
    """Test memory usage of different model sizes"""
    print("\n" + "="*60)
    print("MEMORY USAGE COMPARISON")
    print("="*60)
    
    models = {
        'MemoryOptimizedCSNN': MemoryOptimizedCSNN(),
        'CompactCSNN': CompactCSNN()
    }
    
    for name, model in models.items():
        param_count = model.count_parameters() if hasattr(model, 'count_parameters') else sum(p.numel() for p in model.parameters())
        
        if hasattr(model, 'get_model_size'):
            model_size = model.get_model_size()
        else:
            model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        
        print(f"\n{name}:")
        print(f"  Parameters: {param_count:,}")
        print(f"  Model size: {model_size:.2f} MB")
        
        # Test forward pass memory
        dummy_input = torch.randn(4, 1, 28, 28)
        
        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
            
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                output = model(dummy_input, 10)
            
            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"  Peak GPU memory: {peak_memory:.2f} GB")
            print(f"  Output shape: {output.shape}")
            
            # Cleanup
            del model, output
            torch.cuda.empty_cache()
        else:
            with torch.no_grad():
                output = model(dummy_input, 10)
            print(f"  Output shape: {output.shape}")


if __name__ == "__main__":
    print(" Memory-Optimized SNN Models for MNIST")
    
    # Test model creation
    print("\n--- Testing MemoryOptimizedCSNN ---")
    model1 = MemoryOptimizedCSNN(beta=0.9)
    print(f"Parameters: {model1.count_parameters():,}")
    print(f"Model size: {model1.get_model_size():.2f} MB")
    
    print("\n--- Testing CompactCSNN ---")
    model2 = CompactCSNN(beta=0.9)
    param_count2 = sum(p.numel() for p in model2.parameters())
    print(f"Parameters: {param_count2:,}")
    
    # Test forward pass
    print("\n--- Testing Forward Pass ---")
    dummy_input = torch.randn(2, 1, 28, 28)
    
    with torch.no_grad():
        output1 = model1(dummy_input, 15)
        output2 = model2(dummy_input, 15)
    
    print(f"MemoryOptimizedCSNN output: {output1.shape}")
    print(f"CompactCSNN output: {output2.shape}")
    
    # Memory usage comparison
    test_memory_usage()
    
    print("\n Model verification complete!")
    print("\nRecommendations for 4GB GPU:")
    print("  • Use CompactCSNN for extreme memory constraints")
    print("  • Use MemoryOptimizedCSNN for better accuracy")
    print("  • Start with batch_size=16 or smaller")
    print("  • Use num_steps=10-15 for training")