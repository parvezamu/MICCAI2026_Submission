"""
train_simclr.py

Main training script for SimCLR foundation model
Fixed: Proper GPU handling with CUDA_VISIBLE_DEVICES

Author: Parvez
Date: January 2026
"""

import os
# Set GPU BEFORE importing torch
gpu_id = os.environ.get('SIMCLR_GPU_ID', '7')
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import shutil

# Import custom modules
sys.path.append('.')
from models.resnet3d import resnet3d_18, resnet3d_34, SimCLRModel
from losses.simclr_loss import NTXentLoss
from dataset.simclr_dataset import SimCLRStrokeDataset
from augmentation.simclr_augmentations import SimCLRAugmentation


class SimCLRTrainer:
    """
    Trainer for SimCLR foundation model
    """
    
    def __init__(self, config_path):
        """
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup experiment directory
        self.setup_experiment_dir()
        
        # Set random seeds
        self.set_seed(self.config['experiment']['seed'])
        
        # Setup device
        self.setup_device()
        
        # Build model
        self.build_model()
        
        # Setup data
        self.setup_data()
        
        # Setup optimization
        self.setup_optimization()
        
        # Setup logging
        self.setup_logging()
        
        print("\n" + "="*70)
        print("SimCLR Training Configuration")
        print("="*70)
        print(f"Experiment: {self.config['experiment']['name']}")
        print(f"Output dir: {self.exp_dir}")
        print(f"Device: {self.device}")
        print(f"Datasets: {self.config['data']['datasets']}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Epochs: {self.config['training']['epochs']}")
        print(f"Learning rate: {self.config['training']['learning_rate']}")
        print("="*70 + "\n")
    
    def setup_experiment_dir(self):
        """Create experiment directory"""
        base_dir = Path(self.config['experiment']['output_dir'])
        exp_name = self.config['experiment']['name']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.exp_dir = base_dir / f"{exp_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.exp_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.exp_dir / 'logs').mkdir(exist_ok=True)
        (self.exp_dir / 'images').mkdir(exist_ok=True)
        
        # Save configuration
        config_save_path = self.exp_dir / 'config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print(f"✓ Experiment directory created: {self.exp_dir}")
    
    def set_seed(self, seed):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        # Make CUDA deterministic (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def setup_device(self):
        """Setup GPU device"""
        if self.config['training']['use_gpu'] and torch.cuda.is_available():
            # GPU 7 is already set as device 0 via CUDA_VISIBLE_DEVICES
            self.device = torch.device('cuda:0')
            
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = torch.device('cpu')
            print("✓ Using CPU")
    
    def build_model(self):
        """Build SimCLR model"""
        # Create encoder
        arch = self.config['model']['architecture']
        depth = self.config['model']['depth']
        
        if arch == 'resnet3d':
            if depth == 18:
                encoder = resnet3d_18(in_channels=1)
            elif depth == 34:
                encoder = resnet3d_34(in_channels=1)
            else:
                raise ValueError(f"Unsupported ResNet depth: {depth}")
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        # Create SimCLR model with projection head
        self.model = SimCLRModel(
            encoder=encoder,
            projection_dim=self.config['model']['projection_dim'],
            hidden_dim=self.config['model']['hidden_dim']
        )
        
        self.model = self.model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✓ Model built: {arch}-{depth}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def setup_data(self):
        """Setup datasets and dataloaders"""
        # Create augmentation
        aug_config = self.config['data']['augmentation']
        patch_size = tuple(self.config['data']['patch_size'])
        min_depth = self.config['data']['min_depth']
        
        self.augmentation = SimCLRAugmentation(
            patch_size=patch_size,
            min_depth=min_depth,
            **aug_config
        )
        
        # Create datasets
        preprocessed_dir = self.config['data']['preprocessed_dir']
        datasets = self.config['data']['datasets']
        splits_file = self.config['data']['splits_file']
        
        self.train_dataset = SimCLRStrokeDataset(
            preprocessed_dir=preprocessed_dir,
            datasets=datasets,
            split='train',
            splits_file=splits_file,
            augmentation=self.augmentation
        )
        
        self.val_dataset = SimCLRStrokeDataset(
            preprocessed_dir=preprocessed_dir,
            datasets=datasets,
            split='val',
            splits_file=splits_file,
            augmentation=self.augmentation
        )
        
        # Create dataloaders
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['data']['num_workers']
        pin_memory = self.config['data']['pin_memory']
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        print(f"✓ Datasets created")
        print(f"  Train: {len(self.train_dataset)} samples, {len(self.train_loader)} batches")
        print(f"  Val: {len(self.val_dataset)} samples, {len(self.val_loader)} batches")
    
    def setup_optimization(self):
        """Setup optimizer, scheduler, and loss"""
        # Loss function
        self.criterion = NTXentLoss(
            temperature=self.config['training']['temperature']
        ).to(self.device)
        
        # Optimizer
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if self.config['training']['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif self.config['training']['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['training']['optimizer']}")
        
        # Learning rate scheduler
        epochs = self.config['training']['epochs']
        warmup_epochs = self.config['training']['warmup_epochs']
        
        if self.config['training']['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs - warmup_epochs,
                eta_min=lr * 0.01
            )
        else:
            self.scheduler = None
        
        # Mixed precision training
        self.use_amp = self.config['training']['mixed_precision']
        if self.use_amp:
            self.scaler = GradScaler()
            print("✓ Using mixed precision training (AMP)")
        
        print(f"✓ Optimization setup complete")
    
    def setup_logging(self):
        """Setup logging"""
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Create log file
        self.log_file = self.exp_dir / 'logs' / 'training.log'
        with open(self.log_file, 'w') as f:
            f.write("epoch,train_loss,val_loss,lr\n")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["training"]["epochs"]}')
        
        for batch_idx, batch in enumerate(pbar):
            view1 = batch['view1'].to(self.device)
            view2 = batch['view2'].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    _, z1 = self.model(view1)
                    _, z2 = self.model(view2)
                    loss = self.criterion(z1, z2)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                _, z1 = self.model(view1)
                _, z2 = self.model(view2)
                loss = self.criterion(z1, z2)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if (batch_idx + 1) % self.config['logging']['log_every'] == 0:
                avg_loss = total_loss / num_batches
                current_lr = self.optimizer.param_groups[0]['lr']
                
                with open(self.log_file, 'a') as f:
                    f.write(f"{epoch},{avg_loss:.6f},{current_lr:.6f}\n")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                view1 = batch['view1'].to(self.device)
                view2 = batch['view2'].to(self.device)
                
                if self.use_amp:
                    with autocast():
                        _, z1 = self.model(view1)
                        _, z2 = self.model(view2)
                        loss = self.criterion(z1, z2)
                else:
                    _, z1 = self.model(view1)
                    _, z2 = self.model(view2)
                    loss = self.criterion(z1, z2)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'val_loss': self.val_losses[-1] if self.val_losses else None,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = self.exp_dir / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.exp_dir / 'checkpoints' / 'best_model.pth'
            shutil.copy(checkpoint_path, best_path)
            print(f"  ✓ Saved best model (val_loss: {self.val_losses[-1]:.4f})")
        
        self.keep_top_k_checkpoints()
    
    def keep_top_k_checkpoints(self):
        """Keep only top-k best checkpoints"""
        keep_top_k = self.config['training']['keep_top_k']
        
        checkpoint_dir = self.exp_dir / 'checkpoints'
        checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        if len(checkpoints) > keep_top_k:
            checkpoints_with_loss = []
            for ckpt in checkpoints:
                epoch_num = int(ckpt.stem.split('_')[-1])
                if epoch_num <= len(self.val_losses):
                    val_loss = self.val_losses[epoch_num - 1]
                    checkpoints_with_loss.append((ckpt, val_loss))
            
            checkpoints_with_loss.sort(key=lambda x: x[1])
            
            for ckpt, _ in checkpoints_with_loss[keep_top_k:]:
                ckpt.unlink()
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("Starting Training")
        print("="*70 + "\n")
        
        epochs = self.config['training']['epochs']
        warmup_epochs = self.config['training']['warmup_epochs']
        
        for epoch in range(epochs):
            if epoch < warmup_epochs:
                lr_scale = (epoch + 1) / warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config['training']['learning_rate'] * lr_scale
            
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            val_loss = self.validate(epoch)
            self.val_losses.append(val_loss)
            
            if self.scheduler and epoch >= warmup_epochs:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
            
            if (epoch + 1) % self.config['training']['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
            
            print()
        
        print("="*70)
        print("Training Complete!")
        print(f"Best Val Loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch})")
        print(f"Experiment saved to: {self.exp_dir}")
        print("="*70 + "\n")
        
        self.save_training_curves()
    
    def save_training_curves(self):
        """Save training curves"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        axes[0].plot(epochs, self.train_losses, label='Train', linewidth=2)
        axes[0].plot(epochs, self.val_losses, label='Val', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, self.learning_rates, linewidth=2, color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'training_curves.png', dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    trainer = SimCLRTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()
