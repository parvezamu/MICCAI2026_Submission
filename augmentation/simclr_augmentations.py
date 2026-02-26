"""
augmentation/simclr_augmentations.py

3D data augmentation for SimCLR
Handles full volumes and adaptive patching

Author: Parvez
Date: January 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import rotate, zoom
import random


class SimCLRAugmentation:
    """
    SimCLR data augmentation for 3D medical images
    Handles full volumes and variable sizes
    """
    
    def __init__(
        self,
        patch_size=(96, 128, 128),
        min_depth=64,
        random_crop=True,
        random_flip=True,
        random_rotation=15,
        random_scale=(0.9, 1.1),
        random_intensity=0.1,
        random_gamma=(0.8, 1.2),
        gaussian_noise=0.01,
        gaussian_blur=0.5
    ):
        self.patch_size = patch_size
        self.min_depth = min_depth
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.random_rotation = random_rotation
        self.random_scale = random_scale
        self.random_intensity = random_intensity
        self.random_gamma = random_gamma
        self.gaussian_noise = gaussian_noise
        self.gaussian_blur = gaussian_blur
    
    def __call__(self, volume):
        """
        Apply augmentations to create two views
        
        Args:
            volume: numpy array (D, H, W)
        
        Returns:
            view1, view2: Two augmented views as torch tensors (1, D, H, W)
        """
        # Create two different augmented views
        view1 = self._augment_volume(volume.copy())
        view2 = self._augment_volume(volume.copy())
        
        # Convert to torch tensors
        view1 = torch.from_numpy(view1).unsqueeze(0).float()  # (1, D, H, W)
        view2 = torch.from_numpy(view2).unsqueeze(0).float()
        
        return view1, view2
    
    def _augment_volume(self, volume):
        """Apply random augmentations to a single volume"""
        
        # 1. Handle thin volumes (e.g., ISLES2018 with few slices)
        volume = self._handle_thin_volume(volume)
        
        # 2. Random crop to patch size
        if self.random_crop:
            volume = self._random_crop_3d(volume, self.patch_size)
        else:
            volume = self._center_crop_3d(volume, self.patch_size)
        
        # 3. Random flip
        if self.random_flip and random.random() > 0.5:
            volume = self._random_flip_3d(volume)
        
        # 4. Random rotation
        if self.random_rotation > 0 and random.random() > 0.5:
            volume = self._random_rotation_3d(volume, self.random_rotation)
        
        # 5. Random scaling
        if self.random_scale and random.random() > 0.5:
            volume = self._random_scale_3d(volume, self.random_scale)
        
        # 6. Random intensity shift
        if self.random_intensity > 0 and random.random() > 0.5:
            volume = self._random_intensity_shift(volume, self.random_intensity)
        
        # 7. Random gamma correction
        if self.random_gamma and random.random() > 0.5:
            volume = self._random_gamma_correction(volume, self.random_gamma)
        
        # 8. Gaussian noise
        if self.gaussian_noise > 0 and random.random() > 0.5:
            volume = self._add_gaussian_noise(volume, self.gaussian_noise)
        
        # 9. Gaussian blur
        if self.gaussian_blur > 0 and random.random() > 0.5:
            volume = self._gaussian_blur_3d(volume, self.gaussian_blur)
        
        return volume
    
    def _handle_thin_volume(self, volume):
        """
        Handle volumes with few slices (e.g., ISLES2018)
        Pad to minimum depth if needed
        """
        d, h, w = volume.shape
        
        if d < self.min_depth:
            # Pad symmetrically
            pad_total = self.min_depth - d
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            
            volume = np.pad(
                volume,
                ((pad_before, pad_after), (0, 0), (0, 0)),
                mode='constant',
                constant_values=volume.min()
            )
        
        return volume
    
    def _random_crop_3d(self, volume, crop_size):
        """Random 3D crop"""
        d, h, w = volume.shape
        cd, ch, cw = crop_size
        
        # If volume is smaller than crop size, pad it
        if d < cd or h < ch or w < cw:
            pad_d = max(0, cd - d)
            pad_h = max(0, ch - h)
            pad_w = max(0, cw - w)
            
            volume = np.pad(
                volume,
                (
                    (pad_d//2, pad_d - pad_d//2),
                    (pad_h//2, pad_h - pad_h//2),
                    (pad_w//2, pad_w - pad_w//2)
                ),
                mode='constant',
                constant_values=volume.min()
            )
            d, h, w = volume.shape
        
        # Random crop
        d_start = random.randint(0, d - cd)
        h_start = random.randint(0, h - ch)
        w_start = random.randint(0, w - cw)
        
        return volume[
            d_start:d_start+cd,
            h_start:h_start+ch,
            w_start:w_start+cw
        ]
    
    def _center_crop_3d(self, volume, crop_size):
        """Center 3D crop"""
        d, h, w = volume.shape
        cd, ch, cw = crop_size
        
        d_start = (d - cd) // 2
        h_start = (h - ch) // 2
        w_start = (w - cw) // 2
        
        return volume[
            d_start:d_start+cd,
            h_start:h_start+ch,
            w_start:w_start+cw
        ]
    
    def _random_flip_3d(self, volume):
        """Random flip along axes"""
        # Flip along each axis with 50% probability
        if random.random() > 0.5:
            volume = np.flip(volume, axis=0)  # Depth
        if random.random() > 0.5:
            volume = np.flip(volume, axis=1)  # Height
        if random.random() > 0.5:
            volume = np.flip(volume, axis=2)  # Width
        
        return volume.copy()
    
    def _random_rotation_3d(self, volume, max_angle):
        """Random rotation in 3D"""
        # Rotate around random axis
        axes = [(0, 1), (0, 2), (1, 2)]  # Possible rotation planes
        axis = random.choice(axes)
        angle = random.uniform(-max_angle, max_angle)
        
        volume = rotate(volume, angle, axes=axis, reshape=False, order=1)
        
        return volume
    
    def _random_scale_3d(self, volume, scale_range):
        """Random scaling"""
        scale = random.uniform(scale_range[0], scale_range[1])
        
        original_shape = volume.shape
        scaled_shape = tuple(int(dim * scale) for dim in original_shape)
        
        # Zoom
        volume = zoom(volume, scale, order=1)
        
        # Crop or pad back to original shape
        volume = self._crop_or_pad_to_shape(volume, original_shape)
        
        return volume
    
    def _crop_or_pad_to_shape(self, volume, target_shape):
        """Crop or pad volume to target shape"""
        current_shape = volume.shape
        
        result = np.zeros(target_shape, dtype=volume.dtype)
        
        slices_in = []
        slices_out = []
        
        for i in range(3):
            if current_shape[i] > target_shape[i]:
                # Crop
                start = (current_shape[i] - target_shape[i]) // 2
                slices_in.append(slice(start, start + target_shape[i]))
                slices_out.append(slice(0, target_shape[i]))
            else:
                # Pad
                start = (target_shape[i] - current_shape[i]) // 2
                slices_in.append(slice(0, current_shape[i]))
                slices_out.append(slice(start, start + current_shape[i]))
        
        result[tuple(slices_out)] = volume[tuple(slices_in)]
        
        return result
    
    def _random_intensity_shift(self, volume, max_shift):
        """Random intensity shift"""
        shift = random.uniform(-max_shift, max_shift)
        volume = volume + shift
        return volume
    
    def _random_gamma_correction(self, volume, gamma_range):
        """Random gamma correction"""
        # Normalize to [0, 1]
        min_val = volume.min()
        max_val = volume.max()
        
        if max_val > min_val:
            volume_norm = (volume - min_val) / (max_val - min_val)
            
            # Apply gamma
            gamma = random.uniform(gamma_range[0], gamma_range[1])
            volume_norm = np.power(volume_norm, gamma)
            
            # Denormalize
            volume = volume_norm * (max_val - min_val) + min_val
        
        return volume
    
    def _add_gaussian_noise(self, volume, std):
        """Add Gaussian noise"""
        noise = np.random.normal(0, std, volume.shape)
        volume = volume + noise
        return volume
    
    def _gaussian_blur_3d(self, volume, sigma):
        """Apply Gaussian blur"""
        from scipy.ndimage import gaussian_filter
        sigma_val = random.uniform(0, sigma)
        volume = gaussian_filter(volume, sigma=sigma_val)
        return volume


if __name__ == '__main__':
    # Test augmentation
    augmenter = SimCLRAugmentation(
        patch_size=(96, 128, 128),
        min_depth=64
    )
    
    # Test with normal volume
    volume = np.random.randn(120, 144, 144)
    view1, view2 = augmenter(volume)
    print(f"Normal volume: {volume.shape} -> {view1.shape}, {view2.shape}")
    
    # Test with thin volume (like ISLES2018)
    thin_volume = np.random.randn(10, 144, 144)
    view1, view2 = augmenter(thin_volume)
    print(f"Thin volume: {thin_volume.shape} -> {view1.shape}, {view2.shape}")
