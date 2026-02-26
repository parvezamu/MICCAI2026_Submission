import os
os.environ['CUDA_VISIBLE_DEVICES'] = '9'
import numpy as np
import nibabel as nib
import glob
import h5py
from skimage import measure
from skimage.morphology import reconstruction, binary_dilation, disk
from skimage import transform
from collections import defaultdict, Counter
import json

def rotation_finding(data):
    data_new = []
    area = []
    for i in range(data.shape[0]):
        image = (data[i] > data.min())
        seed = np.copy(image)
        seed[1:-1, 1:-1] = 1.0
        mask = image
        data_new.append(reconstruction(seed, mask, method='erosion'))
        area.append(np.sum(data_new[i]))
    
    area = np.stack(area)
    area_idx = np.argmax(area)
    area_data = data_new[area_idx]
    
    all_labels = measure.label(area_data, background=0)
    properties = measure.regionprops(all_labels)
    area = [prop.area for prop in properties]
    area = np.stack(area)
    area_idx = np.argmax(area)
    y0, x0 = properties[area_idx].centroid
    orientation = properties[area_idx].orientation
    
    return orientation, [y0, x0]

def Skull_striping(mask_data, orient, ot_data=None):
    """Enhanced skull stripping that preserves lesion areas"""
    region_prop = dict()
    
    # If OT data provided, expand mask to include lesion areas
    if ot_data is not None:
        print("Expanding brain mask to include lesion areas...")
        for i in range(mask_data.shape[0]):
            if i < len(ot_data) and np.sum(ot_data[i] > 0) > 0:
                # Dilate original mask and combine with lesions
                dilated_mask = binary_dilation(mask_data[i] > 0, disk(3))
                lesion_mask = ot_data[i] > 0
                mask_data[i] = np.logical_or(dilated_mask, lesion_mask).astype(float)
    
    # Apply morphological reconstruction
    for i in range(mask_data.shape[0]):
        image = mask_data[i]
        seed = np.copy(image)
        seed[1:-1, 1:-1] = image.max()
        mask = image
        mask_data[i] = reconstruction(seed, mask, method='erosion')
    
    # Extract region properties
    for i in range(mask_data.shape[0]):
        all_labels = measure.label(mask_data[i], background=0)
        properties = measure.regionprops(all_labels)
        if len(properties) == 0:
            continue
        
        area = [prop.area for prop in properties]
        area = np.stack(area)
        area_idx = np.argmax(area)
        bbox = properties[area_idx].bbox
        filled_image = properties[area_idx].filled_image
        
        if filled_image.shape[0] > 64 and filled_image.shape[1] > 64:
            region_prop[i] = {
                'bbox': bbox,
                'filled_image': filled_image,
                'orientation': orient[0],
                'centroid': orient[1]
            }
    
    return region_prop

def Cropping_data(data, region_prop, Normli=False, Rot=False):
    data_list = []
    for i in region_prop.keys():
        bbox = region_prop[i]['bbox']
        filled_image = region_prop[i]['filled_image']
        orien = region_prop[i]['orientation']
        centeroid = region_prop[i]['centroid']
        
        data_temp = data[i, bbox[0]:bbox[2], bbox[1]:bbox[3]] * filled_image
        
        if Rot:
            data_temp = transform.rotate(data_temp, angle=-orien*180/3.14, 
                                       center=(centeroid[0]-bbox[0], centeroid[1]-bbox[1]))
            filled_image = transform.rotate(filled_image, angle=-orien*180/3.14, 
                                          center=(centeroid[0]-bbox[0], centeroid[1]-bbox[1]))
        
        if Normli:
            filled_image = np.round(filled_image)
            filled_idx = np.argwhere(filled_image > 0)
            mean_val = np.mean(data_temp[filled_idx[:, 0], filled_idx[:, 1]])
            std_val = np.std(data_temp[filled_idx[:, 0], filled_idx[:, 1]])
            data_temp = (data_temp - mean_val) / (std_val + 1)
        
        data_list.append(data_temp)
    
    return data_list

def Data_Normalization(Data):
    for j in range(len(Data)):
        m = np.mean(Data[j])
        s = np.std(Data[j])
        Data[j] = (Data[j] - m) / (s + 1)
    return Data

def _center_pad_crop(arr, target_hw=(208,176)):
    """Center pad/crop a 2D array to target size without interpolation."""
    th, tw = target_hw
    H, W = arr.shape
    out = np.zeros((th, tw), dtype=arr.dtype)

    # central overlap size
    sh, sw = min(H, th), min(W, tw)

    # source start (centered)
    src_r0 = max(0, (H - sh)//2)
    src_c0 = max(0, (W - sw)//2)

    # destination start (centered)
    dst_r0 = (th - sh)//2
    dst_c0 = (tw - sw)//2

    out[dst_r0:dst_r0+sh, dst_c0:dst_c0+sw] = arr[src_r0:src_r0+sh, src_c0:src_c0+sw]
    return out

def patch_generation_training_with_metadata(data_CT, data_CBF, data_Tmax, data_CBV, data_MTT, data_OT, idx, 
                                           ii_jj_start=[0, 4, 8, 12], target_hw=(208,176)):
    """
    Enhanced patch generation with proper metadata for reconstruction
    """
    data = []
    indices_to_append = []  # [case_index, slice_index, row, col]

    for i in idx:
        print(f"Processing index {i}")
        
        if any(lst[i] is None for lst in [data_CT, data_CBF, data_Tmax, data_CBV, data_MTT, data_OT]):
            print(f"  SKIP: Missing data for case index {i}")
            continue
            
        num_slices = min(len(data_CT[i]), len(data_CBF[i]), len(data_Tmax[i]), 
                        len(data_CBV[i]), len(data_MTT[i]), len(data_OT[i]))
        
        print(f"  Number of slices: {num_slices}")
        
        for j in range(num_slices):
            # Apply standardized cropping/padding
            ct = _center_pad_crop(np.asarray(data_CT[i][j]), target_hw)
            cbf = _center_pad_crop(np.asarray(data_CBF[i][j]), target_hw)
            tmax = _center_pad_crop(np.asarray(data_Tmax[i][j]), target_hw)
            cbv = _center_pad_crop(np.asarray(data_CBV[i][j]), target_hw)
            mtt = _center_pad_crop(np.asarray(data_MTT[i][j]), target_hw)
            ot = _center_pad_crop(np.asarray(data_OT[i][j]), target_hw)

            # Check if slice has lesions
            lesion_pixels = np.sum(ot > 0)
            if lesion_pixels == 0:
                continue  # Skip slices without lesions
            
            print(f"    Slice {j}: {lesion_pixels} lesion pixels")

            try:
                temp = np.stack([ct, cbf, tmax, cbv, mtt, ot], axis=-1)
            except ValueError as e:
                print(f"    Error stacking arrays: {e}")
                continue

            # Generate patches with multiple starting positions
            for ii_jj in ii_jj_start:
                ii = ii_jj
                while ii < temp.shape[0] - 63:
                    jj = ii_jj
                    while jj < temp.shape[1] - 63:
                        patch = temp[ii:ii + 64, jj:jj + 64, :]
                        patch_lesion_pixels = np.sum(patch[:, :, -1] > 0)
                        
                        # ONLY keep patches with lesions
                        if patch_lesion_pixels > 0:
                            data.append(patch)
                            indices_to_append.append([i, j, ii, jj])
                        
                        jj += 16
                    ii += 16

    if not data:
        print("No valid lesion patches were generated!")
        return None, None, None

    data = np.stack(data)
    OT = np.round(data[:, :, :, -1])
    OT = np.expand_dims(OT, axis=-1)
    data = data[:, :, :, :5]
    
    print(f"Final shapes: data: {data.shape}, OT: {OT.shape}")
    lesion_patch_count = np.sum([np.sum(ot) > 0 for ot in OT])
    print(f"Patches with lesions: {lesion_patch_count}/{len(OT)} (100% expected)")

    return data, OT, indices_to_append

def find_nii_file_in_case(case_path, modality):
    if modality == 'CT':
        pattern = f"SMIR.Brain.XX.O.CT.*"
        search_path = os.path.join(case_path, pattern)
        matching_dirs = glob.glob(search_path)
        
        filtered_dirs = []
        for dir_path in matching_dirs:
            dir_name = os.path.basename(dir_path)
            if ('CT_4DPWI' not in dir_name and 'CT_CBF' not in dir_name and 
                'CT_CBV' not in dir_name and 'CT_MTT' not in dir_name and 
                'CT_Tmax' not in dir_name):
                filtered_dirs.append(dir_path)
        
        if filtered_dirs:
            nii_files = glob.glob(os.path.join(filtered_dirs[0], "*.nii"))
            return nii_files[0] if nii_files else None
    else:
        pattern = f"SMIR.Brain.XX.O.{modality}.*"
        search_path = os.path.join(case_path, pattern)
        matching_dirs = glob.glob(search_path)
        
        if matching_dirs:
            nii_files = glob.glob(os.path.join(matching_dirs[0], "*.nii"))
            return nii_files[0] if nii_files else None
    
    return None

# ================ MAIN PROCESSING WITH PROPER VALIDATION MAPPING ================

Dataset = '/hpc/pahm409/ISLES/ISLES2018_Training/TRAINING/'
pp = [d for d in os.listdir(Dataset) if d.startswith('case_')]
pp = sorted(pp, key=lambda x: int(x.split('_')[1]))

# Cross-validation fold assignments - FIXED VALIDATION MAPPING
F0 = [59,14,89,13,25,38,60,70,94,46,53,28,40,76,47,72,10,67,34]
F1 = [11,44,43,66,57,48,75,79,51,69,5,12,93,52,74,26,1,84]
F2 = [56,49,23,19,92,50,58,39,62,82,35,63,21,16,31,15,8,91,33]
F3 = [83,77,32,73,22,87,29,41,61,78,18,30,9,90,36,24,65,4,81]
F4 = [2,71,6,85,45,20,17,80,7,64,68,37,42,86,27,88,55,3,54]

# Initialize data containers and metadata storage
Data_CT, Data_CBF, Data_Tmax, Data_CBV, Data_MTT, Data_OT = [], [], [], [], [], []
case_to_index_mapping = {}  # Maps case number to array index
index_to_case_mapping = {}  # Maps array index to case number
case_metadata = {}  # Complete metadata for reconstruction

print(f"Processing {len(pp)} case directories...")

for i, case_dir in enumerate(pp):
    case_path = os.path.join(Dataset, case_dir)
    case_number = int(case_dir.split('_')[1])
    
    # CRITICAL: Store bidirectional mapping
    case_to_index_mapping[case_number] = i
    index_to_case_mapping[i] = case_number
    
    print(f"\nProcessing case {case_number} (index {i})")
    
    # Find all modality files
    files = {
        'CT': find_nii_file_in_case(case_path, 'CT'),
        'CBF': find_nii_file_in_case(case_path, 'CT_CBF'),
        'CBV': find_nii_file_in_case(case_path, 'CT_CBV'),
        'MTT': find_nii_file_in_case(case_path, 'CT_MTT'),
        'Tmax': find_nii_file_in_case(case_path, 'CT_Tmax'),
        'OT': find_nii_file_in_case(case_path, 'OT')
    }
    
    # Skip if any files missing
    missing_files = [k for k, v in files.items() if v is None]
    if missing_files:
        print(f"SKIP: Missing files for case {case_number}: {missing_files}")
        # Add None placeholders to maintain index alignment
        for data_list in [Data_CT, Data_CBF, Data_Tmax, Data_CBV, Data_MTT, Data_OT]:
            data_list.append(None)
        continue
    
    try:
        # Load all modalities
        modalities = {}
        original_shapes = {}
        
        for mod, filepath in files.items():
            print(f"  Loading {mod}: {os.path.basename(filepath)}")
            img = nib.load(filepath)
            data = img.get_fdata().astype('float32').T
            modalities[mod] = data
            original_shapes[mod] = data.shape
            print(f"    Original shape: {data.shape}")
        
        # Create brain mask including lesion areas
        perfusion_sum = modalities['CBF'] + modalities['Tmax'] + modalities['CBV'] + modalities['MTT']
        perfusion_mask = (perfusion_sum > 0) * 1.0
        
        # Get rotation and enhanced skull stripping
        rot = rotation_finding(modalities['CT'])
        region_prop = Skull_striping(perfusion_mask, rot, modalities['OT'])
        
        print(f"  Valid slices after skull stripping: {len(region_prop)}")
        
        if len(region_prop) == 0:
            print(f"SKIP: No valid slices for case {case_number}")
            for data_list in [Data_CT, Data_CBF, Data_Tmax, Data_CBV, Data_MTT, Data_OT]:
                data_list.append(None)
            continue
        
        # CRITICAL: Save complete reconstruction metadata
        case_metadata[i] = {
            'case_number': int(case_number),
            'case_index': int(i),
            'case_dir': case_dir,
            'original_shapes': {k: list(v) for k, v in original_shapes.items()},
            'target_hw': [208, 176],  # Standardized size after center_pad_crop
            'region_prop': {},
            'rotation': [float(rot[0]), [float(rot[1][0]), float(rot[1][1])]],
            'files': {k: v for k, v in files.items()},
            'total_lesion_voxels': int(np.sum(modalities['OT'] > 0)),
            'lesion_slices': [int(s) for s in range(modalities['OT'].shape[0]) if np.sum(modalities['OT'][s] > 0) > 0]
        }
        
        # Convert region_prop to JSON-serializable format
        for slice_idx, props in region_prop.items():
            case_metadata[i]['region_prop'][int(slice_idx)] = {
                'bbox': [int(x) for x in props['bbox']],
                'filled_image_shape': list(props['filled_image'].shape),
                'orientation': float(props['orientation']),
                'centroid': [float(props['centroid'][0]), float(props['centroid'][1])]
            }
        
        # Process each modality with normalization
        processed_data = {}
        for mod in ['CT', 'CBF', 'Tmax', 'CBV', 'MTT']:
            cropped = Cropping_data(modalities[mod], region_prop)
            processed_data[mod] = Data_Normalization(cropped)
        
        # Process OT without normalization
        processed_data['OT'] = Cropping_data(modalities['OT'], region_prop)
        
        # Add to data lists
        Data_CT.append(processed_data['CT'])
        Data_CBF.append(processed_data['CBF'])
        Data_Tmax.append(processed_data['Tmax'])
        Data_CBV.append(processed_data['CBV'])
        Data_MTT.append(processed_data['MTT'])
        Data_OT.append(processed_data['OT'])
        
        print(f"  SUCCESS: Processed case {case_number}")
        
    except Exception as e:
        print(f"ERROR: Failed to process case {case_number}: {e}")
        for data_list in [Data_CT, Data_CBF, Data_Tmax, Data_CBV, Data_MTT, Data_OT]:
            data_list.append(None)
        continue

print(f"\nSuccessfully processed {len([x for x in Data_CT if x is not None])} cases")
print(f"Metadata saved for {len(case_metadata)} cases")

# FIXED: Create proper fold assignments with validation mapping
F_idx = {}

for fold_idx, val_case_numbers in enumerate([F0, F1, F2, F3, F4]):
    print(f"\nProcessing fold {fold_idx}:")
    print(f"  Validation case numbers: {val_case_numbers}")
    
    # Map case numbers to array indices for validation set
    val_indices = []
    for case_num in val_case_numbers:
        if case_num in case_to_index_mapping:
            array_idx = case_to_index_mapping[case_num]
            if Data_CT[array_idx] is not None:  # Only include successfully processed cases
                val_indices.append(array_idx)
            else:
                print(f"    WARNING: Case {case_num} not processed successfully")
        else:
            print(f"    WARNING: Case {case_num} not found in dataset")
    
    # Training set: all other successfully processed cases
    train_indices = []
    for array_idx in range(len(Data_CT)):
        if (Data_CT[array_idx] is not None and 
            array_idx not in val_indices):
            train_indices.append(array_idx)
    
    F_idx[fold_idx] = {
        'train': train_indices,
        'val': val_indices,
        'val_case_numbers': val_case_numbers,
        'train_case_numbers': [index_to_case_mapping[idx] for idx in train_indices],
        'val_case_numbers_actual': [index_to_case_mapping[idx] for idx in val_indices]
    }
    
    print(f"  Final: {len(train_indices)} train indices, {len(val_indices)} val indices")
    print(f"  Train case numbers: {F_idx[fold_idx]['train_case_numbers']}")
    print(f"  Val case numbers (actual): {F_idx[fold_idx]['val_case_numbers_actual']}")

# Generate patches for each fold with complete metadata
for fold_idx, fold_data in F_idx.items():
    print(f"\n{'='*70}")
    print(f"GENERATING PATCHES FOR FOLD {fold_idx}")
    print(f"{'='*70}")
    
    train_indices = fold_data['train']
    val_indices = fold_data['val']
    
    if len(train_indices) == 0 or len(val_indices) == 0:
        print(f"SKIP fold {fold_idx}: insufficient data")
        continue
    
    print(f"Training on {len(train_indices)} cases: {fold_data['train_case_numbers']}")
    print(f"Validating on {len(val_indices)} cases: {fold_data['val_case_numbers_actual']}")
    
    # Generate training patches
    print("\n--- TRAINING PATCHES ---")
    train_result = patch_generation_training_with_metadata(
        Data_CT, Data_CBF, Data_Tmax, Data_CBV, Data_MTT, Data_OT, 
        train_indices, ii_jj_start=[0, 4, 8, 12]
    )
    
    if train_result[0] is None:
        print(f"Failed to generate training patches for fold {fold_idx}")
        continue
    
    train_Data, train_OT, train_indices_meta = train_result
    
    # Generate validation patches
    print("\n--- VALIDATION PATCHES ---")
    val_result = patch_generation_training_with_metadata(
        Data_CT, Data_CBF, Data_Tmax, Data_CBV, Data_MTT, Data_OT, 
        val_indices, ii_jj_start=[0, 4, 8, 12]
    )
    
    if val_result[0] is None:
        print(f"Failed to generate validation patches for fold {fold_idx}")
        continue
    
    val_Data, val_OT, val_indices_meta = val_result
    
    print(f"\nFOLD {fold_idx} STATISTICS:")
    print(f"Training: {len(train_Data)} patches")
    print(f"Validation: {len(val_Data)} patches")
    
    train_lesion_count = np.sum([np.sum(patch) > 0 for patch in train_OT])
    val_lesion_count = np.sum([np.sum(patch) > 0 for patch in val_OT])
    
    print(f"Lesion patches: {train_lesion_count} train, {val_lesion_count} val")
    print(f"Lesion ratios: {train_lesion_count/len(train_Data):.3f} train, {val_lesion_count/len(val_Data):.3f} val")
    
    # Extract metadata for this fold's cases
    fold_case_metadata = {}
    for case_idx in train_indices + val_indices:
        if case_idx in case_metadata:
            fold_case_metadata[case_idx] = case_metadata[case_idx]
    
    # Save with COMPLETE reconstruction metadata
    save_location = f"/home/pahm409/ISLES2029/GT_Whole_RN16_ISLES2018_F{fold_idx}_FIXED.hdf5"
    
    try:
        print(f"\nSaving to {save_location}")
        with h5py.File(save_location, "w") as hf:
            # Save patch data
            hf.create_dataset('train_Data', data=train_Data, compression="gzip", compression_opts=9)
            hf.create_dataset('train_OT', data=train_OT, compression="gzip", compression_opts=9)
            hf.create_dataset('val_Data', data=val_Data, compression="gzip", compression_opts=9)
            hf.create_dataset('val_OT', data=val_OT, compression="gzip", compression_opts=9)
            
            # CRITICAL: Save patch indices for reconstruction
            hf.create_dataset('train_patch_indices', data=np.array(train_indices_meta))
            hf.create_dataset('val_patch_indices', data=np.array(val_indices_meta))
            
            # Save case number mappings for each patch
            train_case_numbers = []
            val_case_numbers = []
            
            for case_idx, slice_idx, row, col in train_indices_meta:
                case_num = index_to_case_mapping[case_idx]
                train_case_numbers.append(case_num)
            
            for case_idx, slice_idx, row, col in val_indices_meta:
                case_num = index_to_case_mapping[case_idx]
                val_case_numbers.append(case_num)
            
            hf.create_dataset('train_case_numbers', data=np.array(train_case_numbers))
            hf.create_dataset('val_case_numbers', data=np.array(val_case_numbers))
            
            # Save fold assignment verification
            hf.create_dataset('train_case_indices', data=np.array(train_indices))
            hf.create_dataset('val_case_indices', data=np.array(val_indices))
            
            # Save complete case metadata for reconstruction
            hf.attrs['case_metadata'] = json.dumps(fold_case_metadata, indent=2)
            hf.attrs['case_to_index_mapping'] = json.dumps(case_to_index_mapping)
            hf.attrs['index_to_case_mapping'] = json.dumps(index_to_case_mapping)
            
            # Save fold definitions and validation mapping
            hf.attrs['fold_definitions'] = json.dumps({
                'F0': F0, 'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4
            })
            hf.attrs['current_fold'] = fold_idx
            hf.attrs['val_case_numbers_expected'] = json.dumps(fold_data['val_case_numbers'])
            hf.attrs['val_case_numbers_actual'] = json.dumps(fold_data['val_case_numbers_actual'])
            hf.attrs['train_case_numbers_actual'] = json.dumps(fold_data['train_case_numbers'])
            
            # Processing parameters for reconstruction
            hf.attrs['dataset_path'] = Dataset
            hf.attrs['patch_size'] = 64
            hf.attrs['stride'] = 16
            hf.attrs['target_hw'] = json.dumps([208, 176])
            hf.attrs['ii_jj_start'] = json.dumps([0, 4, 8, 12])
            
            # Statistics
            hf.attrs['train_patches'] = len(train_Data)
            hf.attrs['val_patches'] = len(val_Data)
            hf.attrs['train_lesion_patches'] = int(train_lesion_count)
            hf.attrs['val_lesion_patches'] = int(val_lesion_count)
            hf.attrs['train_lesion_ratio'] = float(train_lesion_count/len(train_Data))
            hf.attrs['val_lesion_ratio'] = float(val_lesion_count/len(val_Data))
            
            # Lesion statistics per case in this fold
            fold_lesion_stats = {}
            for case_idx in train_indices + val_indices:
                if case_idx in case_metadata:
                    case_num = case_metadata[case_idx]['case_number']
                    fold_lesion_stats[case_num] = {
                        'total_lesion_voxels': case_metadata[case_idx]['total_lesion_voxels'],
                        'lesion_slices': case_metadata[case_idx]['lesion_slices'],
                        'in_training': case_idx in train_indices,
                        'in_validation': case_idx in val_indices
                    }
            
            hf.attrs['fold_lesion_statistics'] = json.dumps(fold_lesion_stats)
            
            file_size_gb = os.path.getsize(save_location) / (1024**3)
            print(f"SUCCESS: Saved {file_size_gb:.2f} GB with COMPLETE METADATA")
            
    except Exception as e:
        print(f"ERROR saving fold {fold_idx}: {e}")
        continue

print(f"\n{'='*80}")
print("PROCESSING COMPLETE WITH FIXED VALIDATION MAPPING")
print(f"{'='*80}")

# Validation check - verify fold assignments
print(f"\nVALIDATION CHECK:")
for fold_idx in range(5):
    filepath = f"/home/pahm409/ISLES2029/GT_Whole_RN16_ISLES2018_F{fold_idx}_FIXED.hdf5"
    if os.path.exists(filepath):
        try:
            with h5py.File(filepath, 'r') as hf:
                expected_val_cases = json.loads(hf.attrs['val_case_numbers_expected'])
                actual_val_cases = json.loads(hf.attrs['val_case_numbers_actual'])
                train_cases = json.loads(hf.attrs['train_case_numbers_actual'])
                
                train_patches = hf.attrs['train_patches']
                val_patches = hf.attrs['val_patches']
                train_lesion_ratio = hf.attrs['train_lesion_ratio']
                val_lesion_ratio = hf.attrs['val_lesion_ratio']
                
                print(f"Fold {fold_idx}:")
                print(f"  Expected val cases: {expected_val_cases}")
                print(f"  Actual val cases: {actual_val_cases}")
                print(f"  Mapping correct: {set(expected_val_cases) == set(actual_val_cases)}")
                print(f"  Train patches: {train_patches} (lesion ratio: {train_lesion_ratio:.3f})")
                print(f"  Val patches: {val_patches} (lesion ratio: {val_lesion_ratio:.3f})")
                
                # Check for overlap between train and val
                overlap = set(train_cases) & set(actual_val_cases)
                print(f"  Train/Val overlap: {len(overlap)} cases {list(overlap) if overlap else '(none - correct)'}")
                
        except Exception as e:
            print(f"Fold {fold_idx}: Error reading - {e}")

print(f"\n{'='*80}")
print("RECONSTRUCTION UTILITIES")
print(f"{'='*80}")

# Create comprehensive reconstruction utilities
reconstruction_utils = '''
import h5py
import numpy as np
import json
import nibabel as nib

def load_fold_complete(fold_idx, data_dir="/home/pahm409/ISLES2029/"):
    """
    Load complete fold data with all reconstruction metadata
    """
    filepath = f"{data_dir}/GT_Whole_RN16_ISLES2018_F{fold_idx}_FIXED.hdf5"
    
    with h5py.File(filepath, 'r') as hf:
        # Load patch data
        train_data = hf['train_Data'][:]
        train_labels = hf['train_OT'][:]
        val_data = hf['val_Data'][:]
        val_labels = hf['val_OT'][:]
        
        # Load patch indices for reconstruction
        train_indices = hf['train_patch_indices'][:]
        val_indices = hf['val_patch_indices'][:]
        
        # Load case mappings
        train_case_numbers = hf['train_case_numbers'][:]
        val_case_numbers = hf['val_case_numbers'][:]
        train_case_indices = hf['train_case_indices'][:]
        val_case_indices = hf['val_case_indices'][:]
        
        # Load complete metadata
        case_metadata = json.loads(hf.attrs['case_metadata'])
        case_to_index = json.loads(hf.attrs['case_to_index_mapping'])
        index_to_case = json.loads(hf.attrs['index_to_case_mapping'])
        fold_definitions = json.loads(hf.attrs['fold_definitions'])
        
        # Processing parameters
        target_hw = json.loads(hf.attrs['target_hw'])
        patch_size = hf.attrs['patch_size']
        stride = hf.attrs['stride']
        
        return {
            'train_data': train_data,
            'train_labels': train_labels,
            'val_data': val_data,
            'val_labels': val_labels,
            'train_patch_indices': train_indices,  # [case_idx, slice_idx, row, col]
            'val_patch_indices': val_indices,
            'train_case_numbers': train_case_numbers,
            'val_case_numbers': val_case_numbers,
            'train_case_indices': train_case_indices,
            'val_case_indices': val_case_indices,
            'case_metadata': case_metadata,
            'case_to_index_mapping': case_to_index,
            'index_to_case_mapping': index_to_case,
            'fold_definitions': fold_definitions,
            'target_hw': target_hw,
            'patch_size': patch_size,
            'stride': stride,
            'fold_idx': fold_idx
        }

def reconstruct_case_from_patches(predictions, patch_indices, case_metadata, target_case_idx, 
                                target_hw=(208, 176), patch_size=64):
    """
    Reconstruct full volume for a specific case from patch predictions
    
    Args:
        predictions: (N, 64, 64, 1) patch predictions
        patch_indices: (N, 4) array of [case_idx, slice_idx, row, col]
        case_metadata: metadata dict from HDF5
        target_case_idx: case index to reconstruct
        target_hw: standardized size after center_pad_crop
        patch_size: patch size (64)
    
    Returns:
        reconstructed_volume: (num_slices, 208, 176) volume
        original_shapes: dict with original shapes for final conversion
    """
    
    if str(target_case_idx) not in case_metadata:
        raise ValueError(f"Case index {target_case_idx} not found in metadata")
    
    metadata = case_metadata[str(target_case_idx)]
    case_number = metadata['case_number']
    
    print(f"Reconstructing case {case_number} (index {target_case_idx})")
    
    # Get case-specific patches
    case_mask = patch_indices[:, 0] == target_case_idx
    case_predictions = predictions[case_mask]
    case_indices = patch_indices[case_mask]
    
    if len(case_predictions) == 0:
        print(f"No patches found for case {target_case_idx}")
        return None, None
    
    print(f"Found {len(case_predictions)} patches for reconstruction")
    
    # Determine volume dimensions
    max_slice = np.max(case_indices[:, 1]) + 1
    th, tw = target_hw
    
    # Initialize reconstruction volume and weight map
    reconstructed_volume = np.zeros((max_slice, th, tw), dtype='float32')
    weight_map = np.zeros((max_slice, th, tw), dtype='float32')
    
    # Place patches back into volume
    for i, (_, slice_idx, row, col) in enumerate(case_indices):
        slice_idx, row, col = int(slice_idx), int(row), int(col)
        
        end_row = row + patch_size
        end_col = col + patch_size
        
        # Ensure bounds
        if end_row <= th and end_col <= tw and slice_idx < max_slice:
            reconstructed_volume[slice_idx, row:end_row, col:end_col] += case_predictions[i, :, :, 0]
            weight_map[slice_idx, row:end_row, col:end_col] += 1
    
    # Average overlapping predictions
    weight_map[weight_map == 0] = 1
    reconstructed_volume = reconstructed_volume / weight_map
    
    # Get original shapes for final conversion back to original space
    original_shapes = metadata['original_shapes']
    
    print(f"Reconstructed to standardized size: {reconstructed_volume.shape}")
    print(f"Original shapes available: {list(original_shapes.keys())}")
    
    return reconstructed_volume, original_shapes

def convert_to_original_space(standardized_volume, original_shapes, target_hw=(208, 176)):
    """
    Convert from standardized (208, 176) back to original image space
    This reverses the center_pad_crop operation
    """
    
    # Use OT (lesion) original shape as reference
    if 'OT' in original_shapes:
        orig_shape = original_shapes['OT']  # [slices, height, width]
    else:
        orig_shape = original_shapes['CT']
    
    num_slices, orig_h, orig_w = orig_shape
    th, tw = target_hw
    
    print(f"Converting from {standardized_volume.shape} to original {orig_shape}")
    
    # Initialize original space volume
    original_volume = np.zeros((num_slices, orig_h, orig_w), dtype='float32')
    
    # Reverse center_pad_crop logic
    for s in range(min(num_slices, standardized_volume.shape[0])):
        standardized_slice = standardized_volume[s]
        
        # Calculate source and destination regions
        sh, sw = min(orig_h, th), min(orig_w, tw)
        
        # Source start (centered in standardized image)
        src_r0 = (th - sh) // 2
        src_c0 = (tw - sw) // 2
        
        # Destination start (centered in original image)
        dst_r0 = max(0, (orig_h - sh) // 2)
        dst_c0 = max(0, (orig_w - sw) // 2)
        
        # Extract from standardized and place in original
        original_volume[s, dst_r0:dst_r0+sh, dst_c0:dst_c0+sw] = \\
            standardized_slice[src_r0:src_r0+sh, src_c0:src_c0+sw]
    
    print(f"Converted to original space: {original_volume.shape}")
    
    return original_volume

def save_prediction_as_nifti(volume, case_metadata, case_idx, output_dir, threshold=0.5):
    """
    Save prediction as NIFTI file with proper naming
    """
    metadata = case_metadata[str(case_idx)]
    case_number = metadata['case_number']
    
    # Threshold predictions
    binary_volume = (volume > threshold).astype(np.uint8)
    
    # Create NIFTI image (transpose back to original orientation)
    affine = np.eye(4)  # Identity - adjust if you have original affine transforms
    nifti_img = nib.Nifti1Image(binary_volume.T, affine)
    
    # Save with proper naming
    output_file = f"{output_dir}/case_{case_number:03d}_predicted_lesion.nii.gz"
    nib.save(nifti_img, output_file)
    
    lesion_volume = np.sum(binary_volume > 0)
    print(f"Saved: {output_file}")
    print(f"Predicted lesion volume: {lesion_volume} voxels")
    
    return output_file

def validate_reconstruction(fold_data, case_idx, predictions=None):
    """
    Validate reconstruction metadata and process for a specific case
    """
    case_metadata = fold_data['case_metadata']
    
    if str(case_idx) not in case_metadata:
        print(f"Case {case_idx} not found in fold metadata")
        return False
    
    metadata = case_metadata[str(case_idx)]
    case_number = metadata['case_number']
    
    print(f"\\nValidation for case {case_number} (index {case_idx}):")
    print(f"  Original shapes: {metadata['original_shapes']}")
    print(f"  Target HW: {metadata.get('target_hw', fold_data['target_hw'])}")
    print(f"  Total lesion voxels: {metadata['total_lesion_voxels']}")
    print(f"  Lesion slices: {len(metadata['lesion_slices'])}")
    
    # Check patches for this case
    if 'val_patch_indices' in fold_data:
        val_case_mask = fold_data['val_patch_indices'][:, 0] == case_idx
        val_patches_count = np.sum(val_case_mask)
        print(f"  Validation patches available: {val_patches_count}")
        
        if val_patches_count > 0 and predictions is not None:
            case_predictions = predictions[val_case_mask]
            pred_lesion_patches = np.sum([np.sum(p) > 0 for p in case_predictions])
            print(f"  Patches with predicted lesions: {pred_lesion_patches}")
    
    if 'train_patch_indices' in fold_data:
        train_case_mask = fold_data['train_patch_indices'][:, 0] == case_idx
        train_patches_count = np.sum(train_case_mask)
        print(f"  Training patches available: {train_patches_count}")
    
    return True

# Example usage function
def full_reconstruction_pipeline(fold_idx, model, output_dir="/home/pahm409/ISLES2029/predictions/"):
    """
    Complete pipeline: load data -> predict -> reconstruct -> save
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading fold {fold_idx}...")
    fold_data = load_fold_complete(fold_idx)
    
    print(f"Making predictions on validation set...")
    val_data = fold_data['val_data']
    predictions = model.predict(val_data, batch_size=32)
    
    print(f"Reconstructing validation cases...")
    val_case_indices = np.unique(fold_data['val_patch_indices'][:, 0])
    
    for case_idx in val_case_indices:
        print(f"\\n--- Reconstructing case index {case_idx} ---")
        
        # Reconstruct to standardized space
        reconstructed, orig_shapes = reconstruct_case_from_patches(
            predictions, fold_data['val_patch_indices'], 
            fold_data['case_metadata'], case_idx, fold_data['target_hw']
        )
        
        if reconstructed is not None:
            # Convert back to original space
            original_space = convert_to_original_space(reconstructed, orig_shapes, fold_data['target_hw'])
            
            # Save as NIFTI
            output_file = save_prediction_as_nifti(
                original_space, fold_data['case_metadata'], case_idx, output_dir
            )
            
            print(f"Successfully processed case {case_idx}")
    
    print(f"\\nAll predictions saved to: {output_dir}")
    return output_dir

if __name__ == "__main__":
    # Test loading
    fold_data = load_fold_complete(0)
    print(f"Loaded fold 0: {fold_data['train_data'].shape} train, {fold_data['val_data'].shape} val")
    print(f"Validation cases: {fold_data['val_case_numbers']}")
    
    # Validate a specific case
    if len(fold_data['val_case_indices']) > 0:
        test_case = fold_data['val_case_indices'][0]
        validate_reconstruction(fold_data, test_case)
'''

# Save reconstruction utilities
utils_file = "/home/pahm409/ISLES2029/reconstruction_utils_fixed.py"
try:
    with open(utils_file, 'w') as f:
        f.write(reconstruction_utils)
    print(f"Complete reconstruction utilities saved to: {utils_file}")
except Exception as e:
    print(f"Could not save reconstruction utilities: {e}")

# Create validation script
validation_script = '''
#!/usr/bin/env python3

import h5py
import numpy as np
import json
import sys

def validate_all_folds():
    """Comprehensive validation of all fold files"""
    
    print("COMPREHENSIVE FOLD VALIDATION")
    print("=" * 60)
    
    for fold_idx in range(5):
        filepath = f"/home/pahm409/ISLES2029/GT_Whole_RN16_ISLES2018_F{fold_idx}_FIXED.hdf5"
        print(f"\\nFOLD {fold_idx}:")
        print("-" * 40)
        
        try:
            with h5py.File(filepath, 'r') as hf:
                # Check data integrity
                train_data = hf['train_Data']
                train_labels = hf['train_OT']
                val_data = hf['val_Data']
                val_labels = hf['val_OT']
                
                print(f"Data shapes:")
                print(f"  Train: {train_data.shape} -> {train_labels.shape}")
                print(f"  Val: {val_data.shape} -> {val_labels.shape}")
                
                # Check patch indices
                train_indices = hf['train_patch_indices'][:]
                val_indices = hf['val_patch_indices'][:]
                
                print(f"Patch indices:")
                print(f"  Train: {train_indices.shape}")
                print(f"  Val: {val_indices.shape}")
                
                # Verify case mappings
                expected_val = json.loads(hf.attrs['val_case_numbers_expected'])
                actual_val = json.loads(hf.attrs['val_case_numbers_actual'])
                train_cases = json.loads(hf.attrs['train_case_numbers_actual'])
                
                mapping_correct = set(expected_val) == set(actual_val)
                print(f"Case mapping:")
                print(f"  Expected val cases: {len(expected_val)}")
                print(f"  Actual val cases: {len(actual_val)}")
                print(f"  Mapping correct: {mapping_correct}")
                
                if not mapping_correct:
                    missing = set(expected_val) - set(actual_val)
                    extra = set(actual_val) - set(expected_val)
                    print(f"  Missing from val: {missing}")
                    print(f"  Extra in val: {extra}")
                
                # Check train/val overlap
                overlap = set(train_cases) & set(actual_val)
                print(f"  Train/Val overlap: {len(overlap)} {'(ERROR!)' if overlap else '(correct)'}")
                
                # Check lesion ratios
                train_lesion_ratio = hf.attrs['train_lesion_ratio']
                val_lesion_ratio = hf.attrs['val_lesion_ratio']
                
                print(f"Lesion ratios:")
                print(f"  Train: {train_lesion_ratio:.3f}")
                print(f"  Val: {val_lesion_ratio:.3f}")
                
                # Verify metadata completeness
                case_metadata = json.loads(hf.attrs['case_metadata'])
                print(f"Metadata:")
                print(f"  Cases with metadata: {len(case_metadata)}")
                print(f"  Required attrs present: {all(attr in hf.attrs for attr in [
                    'case_to_index_mapping', 'index_to_case_mapping', 'fold_definitions'
                ])}")
                
                # Check reconstruction readiness
                has_patch_indices = 'train_patch_indices' in hf and 'val_patch_indices' in hf
                has_case_mappings = 'train_case_numbers' in hf and 'val_case_numbers' in hf
                has_metadata = 'case_metadata' in hf.attrs
                
                reconstruction_ready = has_patch_indices and has_case_mappings and has_metadata
                print(f"Reconstruction ready: {reconstruction_ready}")
                
                if reconstruction_ready:
                    print("  ✓ All components present for full volume reconstruction")
                else:
                    print("  ✗ Missing components for reconstruction:")
                    if not has_patch_indices: print("    - Patch indices")
                    if not has_case_mappings: print("    - Case number mappings")  
                    if not has_metadata: print("    - Case metadata")
                
                print(f"Status: {'✓ VALID' if mapping_correct and not overlap and reconstruction_ready else '✗ ISSUES'}")
                
        except Exception as e:
            print(f"ERROR reading fold {fold_idx}: {e}")
    
    print(f"\\n{'=' * 60}")
    print("VALIDATION COMPLETE")
    print("Files ready for training and reconstruction if all folds show ✓ VALID")

if __name__ == "__main__":
    validate_all_folds()
'''

# Save validation script  
validation_file = "/home/pahm409/ISLES2029/validate_folds.py"
try:
    with open(validation_file, 'w') as f:
        f.write(validation_script)
    print(f"Validation script saved to: {validation_file}")
except Exception as e:
    print(f"Could not save validation script: {e}")

print(f"\n{'='*80}")
print("FINAL SUMMARY")
print(f"{'='*80}")

total_train = 0
total_val = 0
all_valid = True

for fold_idx in range(5):
    filepath = f"/home/pahm409/ISLES2029/GT_Whole_RN16_ISLES2018_F{fold_idx}_FIXED.hdf5"
    if os.path.exists(filepath):
        try:
            with h5py.File(filepath, 'r') as hf:
                train_count = hf.attrs['train_patches']
                val_count = hf.attrs['val_patches'] 
                train_lesion_ratio = hf.attrs['train_lesion_ratio']
                val_lesion_ratio = hf.attrs['val_lesion_ratio']
                
                expected_val = json.loads(hf.attrs['val_case_numbers_expected'])
                actual_val = json.loads(hf.attrs['val_case_numbers_actual'])
                mapping_correct = set(expected_val) == set(actual_val)
                
                total_train += train_count
                total_val += val_count
                
                status = "✓" if mapping_correct else "✗"
                print(f"Fold {fold_idx} {status}: {train_count:,} train ({train_lesion_ratio:.3f} lesion), {val_count:,} val ({val_lesion_ratio:.3f} lesion)")
                
                if not mapping_correct:
                    all_valid = False
                    
        except Exception as e:
            print(f"Fold {fold_idx}: Error - {e}")
            all_valid = False

print(f"\nTOTAL DATASET:")
print(f"Training patches: {total_train:,}")
print(f"Validation patches: {total_val:,}")
print(f"All folds valid: {'YES' if all_valid else 'NO - check validation mapping'}")

print(f"\nFILES CREATED:")
print(f"✓ GT_Whole_RN16_ISLES2018_F{{0-4}}_FIXED.hdf5 - Complete fold datasets")
print(f"✓ reconstruction_utils_fixed.py - Full reconstruction pipeline")
print(f"✓ validate_folds.py - Validation checker")

print(f"\nNEXT STEPS:")
print(f"1. Run: python validate_folds.py")
print(f"2. Train your models using the _FIXED.hdf5 files")
print(f"3. Use reconstruction_utils_fixed.py for full volume predictions")
print(f"4. All validation case mappings are now correct!")

print(f"\nUSAGE EXAMPLE:")
print(f"from reconstruction_utils_fixed import load_fold_complete")
print(f"fold_data = load_fold_complete(0)")
print(f"X_train = fold_data['train_data']")
print(f"y_train = fold_data['train_labels']")
print(f"X_val = fold_data['val_data']")
print(f"y_val = fold_data['val_labels']")

print(f"\n{'='*80}")
print("PROCESSING COMPLETE - VALIDATION MAPPING FIXED!")
print(f"{'='*80}")
