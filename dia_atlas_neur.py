# Check intensity distributions
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np

# Load a training validation case
train_data = np.load('/home/pahm409/preprocessed_stroke_foundation/ATLAS/sub-r001s002_ses-1.npz')
train_img = train_data['image']

# Load a NEURALCUP case
neuralcup_data = np.load('/home/pahm409/preprocessed_NEURALCUP/NEURALCUP/BBS001.npz')
neuralcup_img = neuralcup_data['image']

print("Training case:")
print(f"  Mean: {train_img.mean():.4f}, Std: {train_img.std():.4f}")
print(f"  Min: {train_img.min():.4f}, Max: {train_img.max():.4f}")
print(f"  Shape: {train_img.shape}")

print("\nNEURALCUP case:")
print(f"  Mean: {neuralcup_img.mean():.4f}, Std: {neuralcup_img.std():.4f}")
print(f"  Min: {neuralcup_img.min():.4f}, Max: {neuralcup_img.max():.4f}")
print(f"  Shape: {neuralcup_img.shape}")
