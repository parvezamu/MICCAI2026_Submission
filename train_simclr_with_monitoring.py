"""
train_simclr_with_monitoring.py

Enhanced SimCLR training with comprehensive monitoring and DSC validation

Author: Parvez
Date: January 2026
"""

import os
gpu_id = os.environ.get('SIMCLR_GPU_ID', '3')
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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

sys.path.append('.')
from models.resnet3d import resnet3d_18, resnet3d_34, SimCLRModel
from losses.simclr_loss import NTXentLoss
from dataset.simclr_dataset import SimCLRStrokeDataset
from augmentation.simclr_augmentations import SimCLRAugmentation


def dice_coefficient(pred, target, smooth=1e-6):
    """Compute Dice coefficient"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()


class SimCLRTrainer:
    """Enhanced trainer with monitoring and diagnostics"""
    
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_experiment_dir()
        self.set_seed(self.config['experiment']['seed'])
        self.setup_device()
        self.build_model()
        self.setup_data()
        self.setup_optimization()
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
        base_dir = Path(self.config['experiment']['output_dir'])
        exp_name = self.config['experiment']['name']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.exp_dir = base_dir / f"{exp_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        (self.exp_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.exp_dir / 'logs').mkdir(exist_ok=True)
        (self.exp_dir / 'images').mkdir(exist_ok=True)
        (self.exp_dir / 'diagnostics').mkdir(exist_ok=True)
        
        config_save_path = self.exp_dir / 'config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print(f"✓ Experiment directory created: {self.exp_dir}")
    
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def setup_device(self):
        if self.config['training']['use_gpu'] and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = torch.device('cpu')
            print("✓ Using CPU")
    
    def build_model(self):
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
        
        self.model = SimCLRModel(
            encoder=encoder,
            projection_dim=self.config['model']['projection_dim'],
            hidden_dim=self.config['model']['hidden_dim']
        )
        
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✓ Model built: {arch}-{depth}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def setup_data(self):
        aug_config = self.config['data']['augmentation']
        patch_size = tuple(self.config['data']['patch_size'])
        min_depth = self.config['data']['min_depth']
        
        self.augmentation = SimCLRAugmentation(
            patch_size=patch_size,
            min_depth=min_depth,
            **aug_config
        )
        
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
        self.criterion = NTXentLoss(
            temperature=self.config['training']['temperature']
        ).to(self.device)
        
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
            raise ValueError(f"Unsupported optimizer")
        
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
        
        self.use_amp = self.config['training']['mixed_precision']
        if self.use_amp:
            self.scaler = GradScaler()
            print("✓ Using mixed precision training (AMP)")
        
        print(f"✓ Optimization setup complete")
    
    def setup_logging(self):
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.representations_train = []
        self.representations_val = []
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        self.log_file = self.exp_dir / 'logs' / 'training.log'
        with open(self.log_file, 'w') as f:
            f.write("epoch,train_loss,val_loss,lr\n")
    
    def extract_representations(self, dataloader, max_samples=200):
        """Extract representations for visualization"""
        self.model.eval()
        
        representations = []
        labels = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i * dataloader.batch_size >= max_samples:
                    break
                
                view1 = batch['view1'].to(self.device)
                h, _ = self.model(view1)
                
                representations.append(h.cpu().numpy())
                labels.extend(batch['dataset'])
        
        if representations:
            representations = np.concatenate(representations, axis=0)
        else:
            representations = np.array([])
        
        return representations, labels
    
    def visualize_representations(self, epoch):
        """Visualize learned representations using t-SNE"""
        print("  Extracting representations for visualization...")
        
        train_reps, train_labels = self.extract_representations(self.train_loader, max_samples=200)
        val_reps, val_labels = self.extract_representations(self.val_loader, max_samples=100)
        
        if len(train_reps) == 0 or len(val_reps) == 0:
            return
        
        # Combine train and val
        all_reps = np.concatenate([train_reps, val_reps], axis=0)
        all_labels = train_labels + val_labels
        split_labels = ['train'] * len(train_labels) + ['val'] * len(val_labels)
        
        # Apply t-SNE
        print("  Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(all_reps)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Color by dataset
        ax = axes[0]
        unique_datasets = list(set(all_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_datasets)))
        
        for i, dataset in enumerate(unique_datasets):
            mask = [label == dataset for label in all_labels]
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=dataset,
                alpha=0.6,
                s=50
            )
        
        ax.set_title(f'Learned Representations - Epoch {epoch+1} (by Dataset)', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Color by split
        ax = axes[1]
        for split in ['train', 'val']:
            mask = [label == split for label in split_labels]
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=split,
                alpha=0.6,
                s=50
            )
        
        ax.set_title(f'Learned Representations - Epoch {epoch+1} (by Split)', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.exp_dir / 'diagnostics' / f'representations_epoch_{epoch+1}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved representation visualization: {save_path}")
    
    def visualize_augmentations(self, epoch):
        """Visualize augmentation quality"""
        sample = self.train_dataset[0]
        view1 = sample['view1'].numpy()[0]  # (D, H, W)
        view2 = sample['view2'].numpy()[0]
        
        # Select middle slice
        mid_slice = view1.shape[0] // 2
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(view1[mid_slice], cmap='gray')
        axes[0].set_title('Augmented View 1', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(view2[mid_slice], cmap='gray')
        axes[1].set_title('Augmented View 2', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.suptitle(f'Augmentation Quality - Epoch {epoch+1}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.exp_dir / 'diagnostics' / f'augmentations_epoch_{epoch+1}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved augmentation visualization: {save_path}")
    
    def plot_training_curves(self, epoch):
        """Plot training curves"""
        if len(self.train_losses) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.train_losses) + 1)
        
        # Plot 1: Losses
        ax = axes[0, 0]
        ax.plot(epochs, self.train_losses, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
        ax.plot(epochs, self.val_losses, 'r-', linewidth=2, label='Val Loss', marker='s', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Learning Rate
        ax = axes[0, 1]
        ax.plot(epochs, self.learning_rates, 'g-', linewidth=2, marker='d', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Loss difference (train - val)
        ax = axes[1, 0]
        loss_diff = [t - v for t, v in zip(self.train_losses, self.val_losses)]
        ax.plot(epochs, loss_diff, 'purple', linewidth=2, marker='^', markersize=4)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Train Loss - Val Loss', fontsize=12)
        ax.set_title('Generalization Gap', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Loss improvement
        ax = axes[1, 1]
        train_improvement = [self.train_losses[0] - loss for loss in self.train_losses]
        val_improvement = [self.val_losses[0] - loss for loss in self.val_losses]
        ax.plot(epochs, train_improvement, 'b-', linewidth=2, label='Train', marker='o', markersize=4)
        ax.plot(epochs, val_improvement, 'r-', linewidth=2, label='Val', marker='s', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Improvement from Epoch 1', fontsize=12)
        ax.set_title('Cumulative Loss Improvement', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.exp_dir / 'diagnostics' / f'training_curves_epoch_{epoch+1}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved training curves: {save_path}")
    
    def train_epoch(self, epoch):
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["training"]["epochs"]}')
        
        for batch_idx, batch in enumerate(pbar):
            view1 = batch['view1'].to(self.device)
            view2 = batch['view2'].to(self.device)
            
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
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, epoch):
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
        print("\n" + "="*70)
        print("Starting Training with Enhanced Monitoring")
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
            
            # Log to file
            with open(self.log_file, 'a') as f:
                f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{current_lr:.6f}\n")
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Gap (Train-Val): {train_loss - val_loss:.4f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
            
            if (epoch + 1) % self.config['training']['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Diagnostics
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print("\n  Running diagnostics...")
                self.plot_training_curves(epoch)
                self.visualize_representations(epoch)
                if epoch == 0:
                    self.visualize_augmentations(epoch)
            
            print()
        
        print("="*70)
        print("Training Complete!")
        print(f"Best Val Loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch})")
        print(f"Experiment saved to: {self.exp_dir}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    trainer = SimCLRTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()
