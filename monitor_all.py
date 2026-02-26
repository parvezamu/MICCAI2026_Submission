"""
monitor_all.py

Master script for comprehensive training monitoring and diagnostics

Usage:
    python monitor_all.py --exp-dir /path/to/experiment

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

sns.set_style("whitegrid")


def load_training_log(log_file):
    """Load training log"""
    df = pd.read_csv(log_file)
    return df


def plot_comprehensive_analysis(df, output_dir):
    """Create comprehensive training analysis plots"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Loss curves
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='Train Loss', marker='o', markersize=3)
    ax1.plot(df['epoch'], df['val_loss'], 'r-', linewidth=2, label='Val Loss', marker='s', markersize=3)
    ax1.fill_between(df['epoch'], df['train_loss'], df['val_loss'], alpha=0.2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning rate
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(df['epoch'], df['lr'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 3: Loss improvement
    ax3 = fig.add_subplot(gs[1, 0])
    train_improvement = df['train_loss'].iloc[0] - df['train_loss']
    val_improvement = df['val_loss'].iloc[0] - df['val_loss']
    ax3.plot(df['epoch'], train_improvement, 'b-', linewidth=2, label='Train', marker='o', markersize=3)
    ax3.plot(df['epoch'], val_improvement, 'r-', linewidth=2, label='Val', marker='s', markersize=3)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Improvement', fontsize=12)
    ax3.set_title('Loss Improvement from Start', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Generalization gap
    ax4 = fig.add_subplot(gs[1, 1])
    gap = df['train_loss'] - df['val_loss']
    ax4.plot(df['epoch'], gap, 'purple', linewidth=2, marker='^', markersize=3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.fill_between(df['epoch'], 0, gap, where=(gap > 0), alpha=0.3, color='red', label='Overfitting')
    ax4.fill_between(df['epoch'], 0, gap, where=(gap < 0), alpha=0.3, color='blue', label='Underfitting')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Train - Val Loss', fontsize=12)
    ax4.set_title('Generalization Gap', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Loss derivatives (rate of change)
    ax5 = fig.add_subplot(gs[1, 2])
    if len(df) > 1:
        train_derivative = np.diff(df['train_loss'])
        val_derivative = np.diff(df['val_loss'])
        epochs_derivative = df['epoch'].iloc[1:]
        ax5.plot(epochs_derivative, train_derivative, 'b-', linewidth=2, label='Train', marker='o', markersize=3)
        ax5.plot(epochs_derivative, val_derivative, 'r-', linewidth=2, label='Val', marker='s', markersize=3)
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('Loss Change', fontsize=12)
        ax5.set_title('Loss Rate of Change', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=11)
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Recent trend (last 10 epochs)
    ax6 = fig.add_subplot(gs[2, 0])
    recent_epochs = min(10, len(df))
    recent_df = df.iloc[-recent_epochs:]
    ax6.plot(recent_df['epoch'], recent_df['train_loss'], 'b-', linewidth=2, label='Train', marker='o')
    ax6.plot(recent_df['epoch'], recent_df['val_loss'], 'r-', linewidth=2, label='Val', marker='s')
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Loss', fontsize=12)
    ax6.set_title(f'Recent Trend (Last {recent_epochs} Epochs)', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Statistics table
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis('off')
    
    stats = [
        ['Metric', 'Train Loss', 'Val Loss'],
        ['Current', f"{df['train_loss'].iloc[-1]:.4f}", f"{df['val_loss'].iloc[-1]:.4f}"],
        ['Best', f"{df['train_loss'].min():.4f}", f"{df['val_loss'].min():.4f}"],
        ['Worst', f"{df['train_loss'].max():.4f}", f"{df['val_loss'].max():.4f}"],
        ['Mean', f"{df['train_loss'].mean():.4f}", f"{df['val_loss'].mean():.4f}"],
        ['Std', f"{df['train_loss'].std():.4f}", f"{df['val_loss'].std():.4f}"],
        ['Improvement', f"{df['train_loss'].iloc[0] - df['train_loss'].iloc[-1]:.4f}", 
         f"{df['val_loss'].iloc[0] - df['val_loss'].iloc[-1]:.4f}"],
        ['Current Gap', f"{df['train_loss'].iloc[-1] - df['val_loss'].iloc[-1]:.4f}", '']
    ]
    
    table = ax7.table(cellText=stats, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(stats)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax7.set_title('Training Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Comprehensive Training Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    save_path = output_dir / 'comprehensive_analysis.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comprehensive analysis: {save_path}")


def print_diagnostics(df):
    """Print diagnostic information"""
    print("\n" + "="*70)
    print("TRAINING DIAGNOSTICS")
    print("="*70 + "\n")
    
    current_epoch = df['epoch'].iloc[-1]
    train_loss = df['train_loss'].iloc[-1]
    val_loss = df['val_loss'].iloc[-1]
    best_val_loss = df['val_loss'].min()
    best_val_epoch = df['val_loss'].idxmin() + 1
    
    print(f"Current Epoch: {current_epoch}")
    print(f"Current Train Loss: {train_loss:.4f}")
    print(f"Current Val Loss: {val_loss:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f} (Epoch {best_val_epoch})")
    print(f"Generalization Gap: {train_loss - val_loss:.4f}")
    
    # Check for issues
    print("\n" + "-"*70)
    print("ISSUE DETECTION")
    print("-"*70 + "\n")
    
    issues_found = False
    
    # Check 1: Increasing validation loss
    if len(df) >= 5:
        recent_val = df['val_loss'].iloc[-5:]
        if recent_val.is_monotonic_increasing:
            print("⚠️  WARNING: Validation loss increasing for last 5 epochs!")
            print("   → Possible overfitting")
            print("   → Consider: reducing LR, increasing regularization, or early stopping")
            issues_found = True
    
    # Check 2: Large generalization gap
    gap = train_loss - val_loss
    if gap < -0.5:
        print("⚠️  WARNING: Large negative gap (val_loss >> train_loss)")
        print("   → Model may be overfitting severely")
        print("   → Consider: stronger regularization or data augmentation")
        issues_found = True
    
    # Check 3: Plateau detection
    if len(df) >= 10:
        recent_train = df['train_loss'].iloc[-10:]
        if recent_train.std() < 0.01:
            print("⚠️  WARNING: Training loss plateaued")
            print("   → Consider: adjusting learning rate or checking convergence")
            issues_found = True
    
    # Check 4: Divergence
    if val_loss > df['val_loss'].iloc[0] * 1.5:
        print("⚠️  WARNING: Validation loss diverging (>50% worse than start)")
        print("   → Consider: restarting with lower LR or different initialization")
        issues_found = True
    
    if not issues_found:
        print("✓ No major issues detected")
    
    print("\n" + "-"*70)
    print("RECOMMENDATIONS")
    print("-"*70 + "\n")
    
    if val_loss < best_val_loss * 1.1:
        print("✓ Training progressing well - continue")
    else:
        print("⚠️  Consider early stopping or hyperparameter adjustment")
    
    # Learning rate recommendation
    if len(df) >= 20:
        recent_improvement = df['val_loss'].iloc[-20] - val_loss
        if recent_improvement < 0.01:
            print("💡 Small recent improvement - consider reducing learning rate")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Comprehensive training monitoring')
    parser.add_argument('--exp-dir', type=str, required=True,
                       help='Path to experiment directory')
    parser.add_argument('--auto-refresh', action='store_true',
                       help='Auto-refresh every 60 seconds')
    
    args = parser.parse_args()
    
    exp_dir = Path(args.exp_dir)
    log_file = exp_dir / 'logs' / 'training.log'
    
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        return
    
    # Load and analyze
    df = load_training_log(log_file)
    
    # Create diagnostics directory
    diagnostics_dir = exp_dir / 'diagnostics'
    diagnostics_dir.mkdir(exist_ok=True)
    
    # Generate comprehensive analysis
    plot_comprehensive_analysis(df, diagnostics_dir)
    
    # Print diagnostics
    print_diagnostics(df)
    
    print("="*70)
    print(f"Analysis complete! Check: {diagnostics_dir}/comprehensive_analysis.png")
    print("="*70)


if __name__ == '__main__':
    main()
