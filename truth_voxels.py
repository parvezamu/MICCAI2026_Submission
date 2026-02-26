import nibabel as nib
import numpy as np

# Pick any ISLES case
msk_file = '/home/pahm409/ISLES2022_reg/ISLES2022/sub-strokecase0237/sub-strokecase0237_ses-0001_msk.nii.gz'
#msk_file = "/home/pahm409/isles_test_results_FINAL/fold_0/sub-strokecase0237/ground_truth.nii.gz"
msk_img = nib.load(msk_file)
data = msk_img.get_fdata()

print(f"Unique values: {np.unique(data)}")
print(f"Min: {data.min()}, Max: {data.max()}")
print(f"Non-zero count: {(data > 0).sum()}")
print(f"Data dtype: {data.dtype}")
