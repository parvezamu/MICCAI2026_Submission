"""
check_flair_vs_t1.py

Compare preprocessed intensities
"""

import numpy as np
from pathlib import Path

# Load same case from both
t1_data = np.load('/home/pahm409/preprocessed_NEURALCUP_1mm/NEURALCUP/BBS001.npz')
flair_data = np.load('/home/pahm409/preprocessed_NEURALCUP_FLAIR/NEURALCUP/BBS001.npz')

print("BBS001 Comparison:")
print("\nT1:")
print(f"  Image mean: {t1_data['image'].mean():.4f}")
print(f"  Image std: {t1_data['image'].std():.4f}")
print(f"  Lesion mask sum: {t1_data['lesion_mask'].sum()}")

print("\nFLAIR:")
print(f"  Image mean: {flair_data['image'].mean():.4f}")
print(f"  Image std: {flair_data['image'].std():.4f}")
print(f"  Lesion mask sum: {flair_data['lesion_mask'].sum()}")

# Check lesion intensities
t1_lesion = t1_data['image'][t1_data['lesion_mask'] > 0]
flair_lesion = flair_data['image'][flair_data['lesion_mask'] > 0]

print("\nLesion intensities:")
print(f"  T1:    mean={t1_lesion.mean():.4f}, std={t1_lesion.std():.4f}")
print(f"  FLAIR: mean={flair_lesion.mean():.4f}, std={flair_lesion.std():.4f}")
