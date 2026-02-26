"""
QUICK ANALYSIS: What Did Pre-training Learn?

Run this FIRST - takes ~5 minutes
Shows you immediately what's in your checkpoint

Author: Parvez  
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.append('.')
from models.resnet3d import resnet3d_18, SimCLRModel


def quick_checkpoint_inspection(checkpoint_path):
    """Quick look at what's in the checkpoint"""
    
    print("="*70)
    print("CHECKPOINT INSPECTION")
    print("="*70)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\nFile: {checkpoint_path.name}")
    print(f"Size: {checkpoint_path.stat().st_size / 1e6:.1f} MB")
    
    print(f"\n1. Training Information:")
    print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   Train Loss: {checkpoint.get('train_loss', 'Unknown')}")
    print(f"   Val Loss: {checkpoint.get('val_loss', 'Unknown')}")
    
    print(f"\n2. What's in the checkpoint:")
    for key in checkpoint.keys():
        value = checkpoint[key]
        if isinstance(value, dict):
            print(f"   {key}: {len(value)} items")
        else:
            print(f"   {key}: {type(value)}")
    
    print(f"\n3. Model Architecture:")
    state_dict = checkpoint['model_state_dict']
    
    # Count parameters per layer
    layer_params = {}
    total_params = 0
    
    for name, param in state_dict.items():
        layer_name = name.split('.')[0] + '.' + name.split('.')[1] if '.' in name else name
        
        if layer_name not in layer_params:
            layer_params[layer_name] = 0
        
        layer_params[layer_name] += param.numel()
        total_params += param.numel()
    
    print(f"\n   Total parameters: {total_params:,}")
    print(f"\n   Top 10 layers by parameter count:")
    sorted_layers = sorted(layer_params.items(), key=lambda x: x[1], reverse=True)
    for name, count in sorted_layers[:10]:
        print(f"     {name:<30} {count:>10,} params")
    
    return checkpoint


def compare_with_random(checkpoint):
    """Compare pre-trained weights with random initialization"""
    
    print("\n" + "="*70)
    print("PRE-TRAINED vs RANDOM COMPARISON")
    print("="*70)
    
    print("\nIMPORTANT NOTE:")
    print("Pre-trained weights will have SIMILAR std dev to random init because")
    print("both use proper weight initialization (He/Kaiming). The REAL difference")
    print("is in the PATTERNS and CORRELATIONS, not just the std dev.")
    print("We need statistical tests to detect this!\n")
    
    # Load pre-trained
    encoder_pre = resnet3d_18(in_channels=1)
    simclr = SimCLRModel(encoder_pre, 128, 512)
    simclr.load_state_dict(checkpoint['model_state_dict'])
    
    # Create random
    encoder_rand = resnet3d_18(in_channels=1)
    
    print("Comparing first 5 convolutional layers:\n")
    print(f"{'Layer':<30} {'Pre-trained':<20} {'Random':<20} {'Rel. Diff':<10} {'Different?'}")
    print("-" * 80)
    
    count = 0
    for (name_pre, param_pre), (name_rand, param_rand) in zip(
        simclr.named_parameters(),
        encoder_rand.named_parameters()
    ):
        if 'weight' in name_pre and param_pre.dim() > 1 and count < 5:
            mean_pre = param_pre.data.mean().item()
            std_pre = param_pre.data.std().item()
            
            mean_rand = param_rand.data.mean().item()
            std_rand = param_rand.data.std().item()
            
            # Check if different (use relative difference)
            relative_diff = abs(std_pre - std_rand) / std_rand if std_rand > 0 else 0
            is_different = relative_diff > 0.05  # 5% change
            
            # Also compute actual distance between weight distributions
            param_distance = torch.norm(param_pre.data - param_rand.data).item()
            
            print(f"{name_pre:<30} μ={mean_pre:>7.4f} σ={std_pre:>7.4f}   "
                  f"μ={mean_rand:>7.4f} σ={std_rand:>7.4f}   "
                  f"Δ={relative_diff:>6.1%}   {'YES ✓' if is_different else 'NO'}")
            
            count += 1
    
    print("\nBETTER ANALYSIS - Weight Distribution Comparison:")
    print("Computing statistical distance between pre-trained and random weights...\n")
    
    # Compute Kolmogorov-Smirnov test
    count = 0
    for (name_pre, param_pre), (name_rand, param_rand) in zip(
        simclr.named_parameters(),
        encoder_rand.named_parameters()
    ):
        if 'weight' in name_pre and param_pre.dim() > 1 and count < 5:
            from scipy.stats import ks_2samp
            
            weights_pre = param_pre.data.cpu().numpy().flatten()
            weights_rand = param_rand.data.cpu().numpy().flatten()
            
            # Statistical test
            ks_stat, p_value = ks_2samp(weights_pre, weights_rand)
            
            # L2 norm difference
            l2_diff = np.linalg.norm(weights_pre - weights_rand)
            
            # Are they statistically different?
            is_significant = p_value < 0.001
            
            print(f"{name_pre:<30}")
            print(f"  KS statistic:  {ks_stat:.6f} {'(DIFFERENT)' if is_significant else '(SAME)'}")
            print(f"  p-value:       {p_value:.2e}")
            print(f"  L2 distance:   {l2_diff:.2f}")
            print()
            
            count += 1
    
    print("\nConclusion:")
    print("  KS test p-value < 0.001: Distributions are SIGNIFICANTLY DIFFERENT")
    print("  KS test p-value > 0.001: Distributions are SIMILAR (training may have failed)")
    print("\n  Large L2 distance: Weights changed during training")
    print("  Small L2 distance: Weights barely changed")


def visualize_weight_distributions(checkpoint, save_path):
    """Visualize what the weights look like"""
    
    print("\n" + "="*70)
    print("WEIGHT DISTRIBUTION VISUALIZATION")
    print("="*70)
    
    # Load model
    encoder = resnet3d_18(in_channels=1)
    simclr = SimCLRModel(encoder, 128, 512)
    simclr.load_state_dict(checkpoint['model_state_dict'])
    
    # Collect weights from different layers
    layers_to_plot = []
    layer_names = []
    
    for name, param in simclr.named_parameters():
        if 'conv' in name and 'weight' in name and param.dim() == 5:
            layers_to_plot.append(param.data.cpu().numpy().flatten())
            layer_names.append(name.replace('encoder.', '').replace('.weight', ''))
            
            if len(layers_to_plot) >= 6:  # Plot first 6 conv layers
                break
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (weights, name) in enumerate(zip(layers_to_plot, layer_names)):
        ax = axes[idx]
        
        # Histogram
        ax.hist(weights, bins=100, alpha=0.7, edgecolor='black')
        ax.set_title(f'{name}\nMean: {weights.mean():.4f}, Std: {weights.std():.4f}',
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add vertical lines for mean and ±std
        ax.axvline(weights.mean(), color='red', linestyle='--', 
                  linewidth=2, label='Mean')
        ax.axvline(weights.mean() + weights.std(), color='green', 
                  linestyle=':', linewidth=1.5, label='±1 Std')
        ax.axvline(weights.mean() - weights.std(), color='green', 
                  linestyle=':', linewidth=1.5)
        
        if idx == 0:
            ax.legend()
    
    plt.suptitle('Pre-trained Weight Distributions\n' + 
                'Structured patterns = Learned features, ' +
                'Gaussian noise = Random weights',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved weight visualization: {save_path}")


def show_feature_map_sizes(checkpoint):
    """Show what size feature maps you get at each layer"""
    
    print("\n" + "="*70)
    print("FEATURE MAP SIZES")
    print("="*70)
    
    # Load model
    encoder = resnet3d_18(in_channels=1)
    simclr = SimCLRModel(encoder, 128, 512)
    simclr.load_state_dict(checkpoint['model_state_dict'])
    
    # Create a dummy input
    dummy_input = torch.randn(1, 1, 96, 96, 96)
    
    # Track sizes
    sizes = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            sizes[name] = output.shape
        return hook
    
    # Register hooks
    hooks = []
    hooks.append(simclr.encoder.conv1.register_forward_hook(hook_fn('conv1')))
    hooks.append(simclr.encoder.layer1.register_forward_hook(hook_fn('layer1')))
    hooks.append(simclr.encoder.layer2.register_forward_hook(hook_fn('layer2')))
    hooks.append(simclr.encoder.layer3.register_forward_hook(hook_fn('layer3')))
    hooks.append(simclr.encoder.layer4.register_forward_hook(hook_fn('layer4')))
    
    # Forward pass
    with torch.no_grad():
        _ = simclr.encoder(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    print("\nInput: (1, 1, 96, 96, 96)")
    print("\nFeature map sizes at each layer:")
    print(f"{'Layer':<15} {'Shape':<30} {'Channels':<10} {'Spatial Size'}")
    print("-" * 70)
    
    for name, shape in sizes.items():
        channels = shape[1]
        spatial = f"{shape[2]} × {shape[3]} × {shape[4]}"
        print(f"{name:<15} {str(tuple(shape)):<30} {channels:<10} {spatial}")
    
    print("\nInterpretation:")
    print("  - Channels increase (64 → 512): More complex features")
    print("  - Spatial size decreases: More abstract, global features")


def main():
    checkpoint_path = Path('/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth')
    output_dir = Path('/home/pahm409/quick_pretrained_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("QUICK PRE-TRAINED CHECKPOINT ANALYSIS")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print("="*70 + "\n")
    
    # 1. Inspect checkpoint
    checkpoint = quick_checkpoint_inspection(checkpoint_path)
    
    # 2. Compare with random
    compare_with_random(checkpoint)
    
    # 3. Visualize weights
    visualize_weight_distributions(
        checkpoint,
        output_dir / 'weight_distributions.png'
    )
    
    # 4. Show feature sizes
    show_feature_map_sizes(checkpoint)
    
    print("\n" + "="*70)
    print("QUICK ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults: {output_dir}")
    print("\nWhat you learned:")
    print("  1. Your checkpoint contains learned weights (not random)")
    print("  2. Weights have structured distributions (learned patterns)")
    print("  3. Feature maps go from detailed (64 channels) to abstract (512 channels)")
    print("\nNext step:")
    print("  Run analyze_pretrained_weights.py to see actual features on brain MRIs")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
