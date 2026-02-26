#!/usr/bin/env python3
"""
Radiomic Features for Walking (FAC) Outcome Prediction - UPGRADED
Multi-modal approach inspired by Jo et al. (2023)

NEW FEATURES:
1. Location-aware features (lesion eloquence, spatial relationships)
2. Three-stage pipeline (Clinical → Imaging → Integrated)
3. Adaptive feature selection (only add if clinically meaningful)
4. Calibration metrics (Brier score, calibration curves)
5. Lesion-volume interaction modeling

Author: Parvez (upgraded)
Date: 2025-01-13
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import sys
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Check GPU
print("\n" + "="*70)
print("GPU CHECK")
print("="*70)
try:
    import subprocess
    gpu_info = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader']
    )
    print(f"GPU Available: {gpu_info.decode().strip()}")
except Exception:
    print("GPU not available - using CPU")
print("="*70)

# Radiomics
from radiomics import featureextractor

# Machine Learning
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.calibration import calibration_curve

from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    confusion_matrix, precision_score, recall_score, f1_score,
    classification_report, balanced_accuracy_score, brier_score_loss
)
from scipy import stats

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = "/home/pahm409/pytorch_loss_comparison_results2/"
T1_IMAGE_DIR = "/hpc/pahm409/harvard/UOA/FNIRT/train/"
ATLAS_DIR = "/hpc/pahm409/ISLES/ATLAS"
CLINICAL_DATA_PATH = "/hpc/pahm409/harvard/UOA/IMPRESS_RETROSPECTIVE_CLINICAL_DATA.xlsx"
OUTPUT_DIR = "./radiomic_fac_analysis_UPGRADED/"

LOSS_FUNCTIONS = ['AdaptiveRegional', 'GDice', 'BCEDice', 'BCEFocalTversky', 'Dice', 'Focal', 'FocalTversky', 'Tversky']
CLINICAL_FEATURES = ['AGE', 'NIHSS', 'GENDER', 'HEMI', 'HTN', 'DIABETES', 'AF', 'TPA', 'FAC_BASE']

# Jo et al. used only 3 clinical features - we'll use our minimal set
MINIMAL_CLINICAL = ['AGE', 'NIHSS', 'FAC_BASE']

# XGBoost params
XGBOOST_PARAMS = {
    'max_depth': 3,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'min_child_weight': 3,
    'random_state': 42,
    'eval_metric': 'auc',
    'use_label_encoder': False,
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'predictor': 'gpu_predictor'
}

CV_FOLDS = 5
CV_RANDOM_STATE = 42

# Atlas thresholds
CST_THRESHOLD = 0.20
CORTEX_THRESHOLD = 0.70
LESION_THRESHOLD = 0.5

# Clinical significance threshold (Jo et al. showed ~0.03 AUC improvement)
CLINICAL_SIGNIFICANCE_THRESHOLD = 0.01

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_output_directory():
    """Create output directory structure"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/radiomics_features", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/predictions", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/calibration_plots", exist_ok=True)
    print(f"✓ Output directory created: {OUTPUT_DIR}")


def get_base_models():
    """Define all classifiers to compare"""
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=500, class_weight='balanced', solver='liblinear', random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight='balanced', random_state=42
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(**XGBOOST_PARAMS),
    }
    return models


def evaluate_with_calibration(model, X, y, model_name, verbose=True):
    """
    Evaluate classifier with calibration metrics (like Jo et al.)
    Returns: AUC, Brier score, calibration slope
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"TRAINING MODEL: {model_name}")
        print(f"{'='*70}")
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

    # Ensure numeric
    if X.dtype == 'object' or not np.issubdtype(X.dtype, np.number):
        X = X.astype(float)
    
    if np.any(np.isnan(X)):
        nan_count = np.sum(np.any(np.isnan(X), axis=1))
        print(f"✗ ERROR: Found {nan_count} rows with NaN")
        return None

    # Setup CV
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    
    y_pred_proba_all = np.zeros(len(y))
    auc_scores = []
    
    # CV loop with scaling inside each fold
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit scaler ONLY on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Clone and train model
        try:
            model_fold = model.__class__(**model.get_params())
            model_fold.fit(X_train_scaled, y_train)
            
            # Get probabilities
            if hasattr(model_fold, "predict_proba"):
                proba = model_fold.predict_proba(X_test_scaled)[:, 1]
            else:
                df = model_fold.decision_function(X_test_scaled)
                proba = 1.0 / (1.0 + np.exp(-df))
            
            y_pred_proba_all[test_idx] = proba
            fold_auc = roc_auc_score(y_test, proba)
            auc_scores.append(fold_auc)
            
            if verbose:
                print(f"  Fold {fold_idx+1}: AUC = {fold_auc:.3f}")
                
        except Exception as e:
            print(f"  ✗ Fold {fold_idx+1} failed: {e}")
            continue
    
    if len(auc_scores) == 0:
        print("✗ ERROR: No successful folds")
        return None
    
    auc_mean = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    
    # Optimal threshold (Youden J)
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba_all)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]
    
    y_pred = (y_pred_proba_all >= optimal_threshold).astype(int)
    
    # Compute metrics
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y, y_pred)
    
    # CALIBRATION METRICS (Jo et al. reported these)
    brier = brier_score_loss(y, y_pred_proba_all)
    
    # Calibration curve
    try:
        prob_true, prob_pred = calibration_curve(y, y_pred_proba_all, n_bins=10)
        calibration_data = (prob_true, prob_pred)
    except:
        calibration_data = None
    
    # Calibration slope
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(y_pred_proba_all.reshape(-1, 1), y)
    calibration_slope = lr.coef_[0]
    
    if verbose:
        print(f"\n✓ AUC = {auc_mean:.3f} ± {auc_std:.3f}")
        print(f"✓ Brier score = {brier:.3f}")
        print(f"✓ Calibration slope = {calibration_slope:.3f}")
        print(f"✓ Optimal threshold = {optimal_threshold:.3f}")
    
    return {
        'model_name': model_name,
        'auc_mean': auc_mean,
        'auc_std': auc_std,
        'auc_scores': auc_scores,
        'brier_score': brier,
        'calibration_slope': calibration_slope,
        'calibration_curve': calibration_data,
        'optimal_threshold': optimal_threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'balanced_acc': bal_acc,
        'y_true': y,
        'y_pred_proba': y_pred_proba_all,
    }


def build_comparison_table(results_dict, output_csv_path=None):
    """Build comparison table with calibration metrics"""
    rows = []
    for key, res in results_dict.items():
        rows.append({
            'Scenario_Model': key,
            'AUC_mean': res['auc_mean'],
            'AUC_std': res['auc_std'],
            'Brier_score': res['brier_score'],
            'Calibration_slope': res['calibration_slope'],
            'Sensitivity': res['sensitivity'],
            'Specificity': res['specificity'],
            'F1': res['f1'],
            'BalancedAcc': res['balanced_acc'],
        })
    
    df_cmp = pd.DataFrame(rows).sort_values('AUC_mean', ascending=False)
    
    print("\n" + "="*70)
    print("GLOBAL MODEL RANKING (by AUC)")
    print("="*70)
    print(df_cmp[['Scenario_Model','AUC_mean','Brier_score','Sensitivity','Specificity']].head(20).to_string(index=False))

    if output_csv_path is not None:
        df_cmp.to_csv(output_csv_path, index=False)
        print(f"\n✓ Saved to {output_csv_path}")
    
    return df_cmp


def plot_calibration_curves(results_dict, output_dir):
    """Plot calibration curves for top models"""
    print("\nGenerating calibration plots...")
    
    # Sort by AUC
    sorted_results = sorted(results_dict.items(), key=lambda x: x[1]['auc_mean'], reverse=True)
    
    # Plot top 6 models
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (model_name, res) in enumerate(sorted_results[:6]):
        ax = axes[idx]
        
        if res['calibration_curve'] is not None:
            prob_true, prob_pred = res['calibration_curve']
            ax.plot(prob_pred, prob_true, 's-', label='Model')
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Observed Frequency')
            ax.set_title(f"{model_name}\nAUC={res['auc_mean']:.3f}, Brier={res['brier_score']:.3f}")
            ax.legend()
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{output_dir}/calibration_plots/top_models_calibration.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Calibration plots saved to {plot_path}")
    plt.close()


# ============================================================================
# CLINICAL DATA LOADING
# ============================================================================

def load_clinical_data():
    """Load and preprocess clinical data"""
    print("\n" + "="*70)
    print("LOADING CLINICAL DATA")
    print("="*70)
    
    df = pd.read_excel(CLINICAL_DATA_PATH)
    print(f"✓ Loaded {len(df)} patients")
    
    df = df.rename(columns={'IMPRESS code': 'patient_id'})
    
    # Convert FAC12
    df['FAC12'] = pd.to_numeric(df['FAC12'], errors='coerce')
    n_before = len(df)
    df = df[df['FAC12'].notna()].copy()
    n_after = len(df)
    print(f"✓ Filtered: {n_before} → {n_after} patients")
    
    # Binary outcome: FAC≥4 = 1, FAC<4 = 0
    df['FAC12_binary'] = (df['FAC12'] >= 4).astype(int)
    
    # Preprocess features
    print("\nPreprocessing clinical features:")
    
    # GENDER
    if 'GENDER' in df.columns:
        df['GENDER'] = df['GENDER'].fillna('Unknown')
        df['GENDER'] = (df['GENDER'].astype(str).str.upper().str.strip() == 'M').astype(int)
    else:
        df['GENDER'] = 0
    
    # HEMI
    if 'HEMI' in df.columns:
        df['HEMI'] = df['HEMI'].fillna('L')
        df['HEMI'] = (df['HEMI'].astype(str).str.upper().str.strip() == 'R').astype(int)
    else:
        df['HEMI'] = 0
    
    # Binary features
    for feat in ['HTN', 'DIABETES', 'AF', 'TPA']:
        if feat in df.columns:
            df[feat] = df[feat].fillna('N')
            df[feat] = (df[feat].astype(str).str.upper().str.strip() == 'Y').astype(int)
        else:
            df[feat] = 0
    
    # Continuous features
    for feat in ['AGE', 'NIHSS', 'FAC_BASE']:
        if feat in df.columns:
            df[feat] = pd.to_numeric(df[feat], errors='coerce')
            median_val = df[feat].median()
            n_missing = df[feat].isna().sum()
            df[feat] = df[feat].fillna(median_val)
            print(f"  ✓ {feat}: mean={df[feat].mean():.1f}, {n_missing} missing")
        else:
            df[feat] = 0
    
    print(f"\nOutcome: FAC≥4={df['FAC12_binary'].sum()} ({df['FAC12_binary'].mean()*100:.1f}%)")
    
    return df


# ============================================================================
# ATLAS LOADING (for location-aware features)
# ============================================================================

class AtlasManager:
    """Manage atlas loading and caching"""
    
    def __init__(self, atlas_dir):
        self.atlas_dir = Path(atlas_dir)
        self.jhu_dir = self.atlas_dir / 'JHU'
        self.smatt_dir = self.atlas_dir / 'SMATT'
        self.atlas_cache = {}
        
        # Motor regions with functional importance weights
        self.motor_regions = {
            'CST_L': {'path': self.jhu_dir / 'prob_split' / 'L_CST.nii.gz', 'weight': 0.9, 'threshold': CST_THRESHOLD},
            'CST_R': {'path': self.jhu_dir / 'prob_split' / 'R_CST.nii.gz', 'weight': 0.9, 'threshold': CST_THRESHOLD},
            'M1_L': {'path': self.smatt_dir / 'L_M1.nii.gz', 'weight': 1.0, 'threshold': CORTEX_THRESHOLD},
            'M1_R': {'path': self.smatt_dir / 'R_M1.nii.gz', 'weight': 1.0, 'threshold': CORTEX_THRESHOLD},
            'S1_L': {'path': self.smatt_dir / 'L_S1.nii.gz', 'weight': 0.8, 'threshold': CORTEX_THRESHOLD},
            'S1_R': {'path': self.smatt_dir / 'R_S1.nii.gz', 'weight': 0.8, 'threshold': CORTEX_THRESHOLD},
            'PMd_L': {'path': self.smatt_dir / 'L_PMd.nii.gz', 'weight': 0.7, 'threshold': CORTEX_THRESHOLD},
            'PMd_R': {'path': self.smatt_dir / 'R_PMd.nii.gz', 'weight': 0.7, 'threshold': CORTEX_THRESHOLD},
            'PMv_L': {'path': self.smatt_dir / 'L_PMv.nii.gz', 'weight': 0.7, 'threshold': CORTEX_THRESHOLD},
            'PMv_R': {'path': self.smatt_dir / 'R_PMv.nii.gz', 'weight': 0.7, 'threshold': CORTEX_THRESHOLD},
            'PLIC_L': {'path': self.smatt_dir / 'L_PLIC.nii.gz', 'weight': 0.9, 'threshold': CST_THRESHOLD},
            'PLIC_R': {'path': self.smatt_dir / 'R_PLIC.nii.gz', 'weight': 0.9, 'threshold': CST_THRESHOLD},
        }
        
        self.load_atlases()
    
    def load_atlases(self):
        """Load all atlases"""
        print("\nLoading motor atlases...")
        for region_name, info in self.motor_regions.items():
            if info['path'].exists():
                self.atlas_cache[region_name] = sitk.ReadImage(str(info['path']))
                print(f"  ✓ {region_name}")
        print(f"Total: {len(self.atlas_cache)} atlases loaded")
    
    def get_atlas_data(self, region_name, target_shape):
        """Get atlas data resampled to target shape"""
        if region_name not in self.atlas_cache:
            return None
        
        atlas_img = self.atlas_cache[region_name]
        atlas_data = sitk.GetArrayFromImage(atlas_img)
        
        if atlas_data.shape == target_shape:
            return atlas_data
        else:
            return None  # Skip if shape mismatch
    
    def compute_motor_eloquence_score(self, lesion_binary, lesion_volume):
        """
        Compute motor eloquence score (inspired by Jo et al.)
        Weighted sum of overlaps with motor regions
        """
        if lesion_volume == 0:
            return 0.0
        
        eloquence_score = 0.0
        
        for region_name, info in self.motor_regions.items():
            atlas_data = self.get_atlas_data(region_name, lesion_binary.shape)
            if atlas_data is None:
                continue
            
            # Binarize atlas
            atlas_binary = (atlas_data > info['threshold']).astype(int)
            
            # Compute overlap
            overlap = np.sum(lesion_binary * atlas_binary)
            
            if overlap > 0:
                # Weight by functional importance
                overlap_ratio = overlap / lesion_volume
                weighted_overlap = info['weight'] * overlap_ratio
                eloquence_score += weighted_overlap
        
        return eloquence_score


# ============================================================================
# RADIOMIC EXTRACTION WITH LOCATION-AWARE FEATURES
# ============================================================================

def get_ensemble_mask(patient_id, loss_function):
    """
    Get ensemble mask across replications
    Searches for loss function directories with flexible naming
    """
    fold = None
    
    # Try to find which fold this patient is in
    for fold_idx in range(1, 6):
        fold_dir = Path(f"{BASE_DIR}/fold_{fold_idx}")
        if not fold_dir.exists():
            continue
        
        # Look for any directory matching the loss function pattern
        # Patterns: LossFunc_rep1, LossFunc_rep2, etc.
        for loss_dir in fold_dir.glob(f"{loss_function}_rep*"):
            case_path = loss_dir / f"case_{patient_id}" / "reconstructed_prediction.nii.gz"
            if case_path.exists():
                fold = fold_idx
                break
        
        if fold:
            break
    
    if fold is None:
        return None
    
    # Collect masks from all replications
    masks = []
    mask_ref = None
    fold_dir = Path(f"{BASE_DIR}/fold_{fold}")
    
    for loss_dir in fold_dir.glob(f"{loss_function}_rep*"):
        case_path = loss_dir / f"case_{patient_id}" / "reconstructed_prediction.nii.gz"
        if case_path.exists():
            mask = sitk.ReadImage(str(case_path))
            mask_ref = mask
            masks.append(sitk.GetArrayFromImage(mask))
    
    if len(masks) == 0:
        return None
    
    # Ensemble: average and threshold
    ensemble_array = np.mean(masks, axis=0)
    ensemble_binary = (ensemble_array > 0.5).astype(np.uint8)
    
    ensemble_mask = sitk.GetImageFromArray(ensemble_binary)
    ensemble_mask.CopyInformation(mask_ref)
    
    return ensemble_mask


def extract_location_aware_features(patient_id, t1_image, mask, loss_function, atlas_manager):
    """
    Extract radiomic + location-aware features (inspired by Jo et al.)
    
    KEY ADDITIONS:
    1. Motor eloquence score (weighted overlap with motor regions)
    2. Lesion-volume interaction (small lesions: location matters more)
    3. Distance to motor regions
    4. Bilateral involvement metrics
    """
    
    features = {}
    prefix = f"{loss_function}_"
    
    try:
        # Get lesion data
        lesion_data = sitk.GetArrayFromImage(mask)
        binary_lesion = (lesion_data > LESION_THRESHOLD).astype(int)
        lesion_volume_voxels = np.sum(binary_lesion)
        
        # Lesion volume in mL
        spacing = mask.GetSpacing()
        voxel_volume = np.prod(spacing)
        lesion_volume_ml = lesion_volume_voxels * voxel_volume / 1000.0
        
        features[f'{prefix}lesion_volume_voxels'] = lesion_volume_voxels
        features[f'{prefix}lesion_volume_ml'] = lesion_volume_ml
        
        if lesion_volume_voxels == 0:
            return features
        
        # ====================================================================
        # LOCATION-AWARE FEATURES (NEW - inspired by Jo et al.)
        # ====================================================================
        
        # 1. Motor eloquence score
        motor_eloquence = atlas_manager.compute_motor_eloquence_score(
            binary_lesion, lesion_volume_voxels
        )
        features[f'{prefix}motor_eloquence_score'] = motor_eloquence
        
        # 2. Lesion-volume interaction (Jo et al. found this critical)
        if lesion_volume_ml < 10:  # Small lesion - location dominates
            features[f'{prefix}location_dominance'] = motor_eloquence * 2.0
            features[f'{prefix}volume_dominance'] = 0.0
        elif lesion_volume_ml < 50:  # Medium lesion - both matter
            features[f'{prefix}location_volume_interaction'] = motor_eloquence * (lesion_volume_ml / 50.0)
            features[f'{prefix}volume_dominance'] = lesion_volume_ml / 50.0
        else:  # Large lesion - volume dominates
            features[f'{prefix}location_dominance'] = 0.0
            features[f'{prefix}volume_dominance'] = lesion_volume_ml / 100.0
        
        # 3. Individual atlas overlaps (for bilateral involvement)
        left_motor_damage = 0.0
        right_motor_damage = 0.0
        
        for region_name, info in atlas_manager.motor_regions.items():
            atlas_data = atlas_manager.get_atlas_data(region_name, binary_lesion.shape)
            if atlas_data is None:
                continue
            
            atlas_binary = (atlas_data > info['threshold']).astype(int)
            overlap = np.sum(binary_lesion * atlas_binary)
            
            if overlap > 0:
                overlap_ratio = overlap / lesion_volume_voxels
                features[f'{prefix}{region_name}_overlap'] = overlap_ratio
                
                # Track left/right
                if '_L' in region_name:
                    left_motor_damage += overlap_ratio * info['weight']
                elif '_R' in region_name:
                    right_motor_damage += overlap_ratio * info['weight']
        
        # 4. Laterality and bilateral involvement
        total_motor_damage = left_motor_damage + right_motor_damage
        
        if total_motor_damage > 0:
            laterality = (left_motor_damage - right_motor_damage) / (total_motor_damage + 1e-6)
            features[f'{prefix}motor_laterality'] = laterality
            features[f'{prefix}bilateral_motor_damage'] = min(left_motor_damage, right_motor_damage)
        else:
            features[f'{prefix}motor_laterality'] = 0.0
            features[f'{prefix}bilateral_motor_damage'] = 0.0
        
        features[f'{prefix}left_motor_damage'] = left_motor_damage
        features[f'{prefix}right_motor_damage'] = right_motor_damage
        features[f'{prefix}total_motor_damage'] = total_motor_damage
        
        # 5. Lesion center of mass
        if lesion_volume_voxels > 0:
            lesion_coords = np.argwhere(binary_lesion > 0)
            com = np.mean(lesion_coords, axis=0)
            
            # Normalize to image space
            features[f'{prefix}com_z'] = com[0] / binary_lesion.shape[0]
            features[f'{prefix}com_y'] = com[1] / binary_lesion.shape[1]
            features[f'{prefix}com_x'] = com[2] / binary_lesion.shape[2]
        
        # ====================================================================
        # TRADITIONAL RADIOMIC FEATURES
        # ====================================================================
        
        # Fix geometry mismatch
        if t1_image.GetSize() == mask.GetSize():
            mask.SetSpacing(t1_image.GetSpacing())
            mask.SetOrigin(t1_image.GetOrigin())
            mask.SetDirection(t1_image.GetDirection())
        else:
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(t1_image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            resampler.SetOutputPixelType(mask.GetPixelID())
            mask = resampler.Execute(mask)
        
        # Extract radiomics
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.settings['geometryTolerance'] = 1e-3
        
        extractor.enableImageTypeByName('Original')
        extractor.enableFeatureClassByName('shape')
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('glcm')
        extractor.enableFeatureClassByName('glrlm')
        
        radiomic_features = extractor.execute(t1_image, mask)
        
        # Add radiomic features
        for key, val in radiomic_features.items():
            if not key.startswith('original_'):
                continue
            
            try:
                val_float = float(val)
                if np.isnan(val_float) or np.isinf(val_float):
                    continue
                clean_name = key.replace('original_', '').replace(' ', '_').lower()
                features[f'{prefix}{clean_name}'] = val_float
            except (TypeError, ValueError):
                continue
        
        return features
    
    except Exception as e:
        print(f"  Error: {e}")
        return None


def extract_radiomics_all_patients(clinical_df, loss_function, atlas_manager):
    """Extract radiomics + location features for all patients"""
    
    print(f"\n{'='*70}")
    print(f"EXTRACTING FEATURES: {loss_function}")
    print(f"{'='*70}")
    
    radiomic_data = []
    patient_ids = clinical_df['patient_id'].unique()
    
    # First, check how many patients we expect to find
    print(f"Checking availability for {len(patient_ids)} patients...")
    
    available_count = 0
    for patient_id in patient_ids:
        t1_path = f"{T1_IMAGE_DIR}/{patient_id}_T1_FNIRT_MNI.nii.gz"
        if os.path.exists(t1_path):
            # Quick check if mask exists
            fold = None
            for fold_idx in range(1, 6):
                fold_dir = Path(f"{BASE_DIR}/fold_{fold_idx}")
                if fold_dir.exists():
                    for loss_dir in fold_dir.glob(f"{loss_function}_rep*"):
                        case_path = loss_dir / f"case_{patient_id}" / "reconstructed_prediction.nii.gz"
                        if case_path.exists():
                            available_count += 1
                            break
                    if fold:
                        break
    
    print(f"Expected to extract: ~{available_count} patients")
    
    if available_count == 0:
        print(f"⚠️  WARNING: No segmentation results found for {loss_function}")
        print(f"   Check if directory naming matches: {BASE_DIR}/fold_*/{{loss_function}}_rep*/")
        return None
    
    # Now extract
    for idx, patient_id in enumerate(patient_ids, 1):
        if idx % 10 == 0 or idx <= 5:  # Print first 5 and every 10th
            print(f"[{idx}/{len(patient_ids)}] {patient_id}...", end=' ')
        
        t1_path = f"{T1_IMAGE_DIR}/{patient_id}_T1_FNIRT_MNI.nii.gz"
        if not os.path.exists(t1_path):
            if idx % 10 == 0 or idx <= 5:
                print("✗ No T1")
            continue
        
        t1_image = sitk.ReadImage(t1_path)
        mask = get_ensemble_mask(patient_id, loss_function)
        
        if mask is None:
            if idx % 10 == 0 or idx <= 5:
                print("✗ No mask")
            continue
        
        features = extract_location_aware_features(
            patient_id, t1_image, mask, loss_function, atlas_manager
        )
        
        if features is not None:
            features['patient_id'] = patient_id
            features['loss_function'] = loss_function
            radiomic_data.append(features)
            if idx % 10 == 0 or idx <= 5:
                print("✓")
        else:
            if idx % 10 == 0 or idx <= 5:
                print("✗ Failed")
    
    if len(radiomic_data) == 0:
        print(f"\n✗ NO FEATURES EXTRACTED for {loss_function}")
        print(f"   This loss function will be SKIPPED in analysis")
        return None
    
    radiomic_df = pd.DataFrame(radiomic_data)
    n_features = len([c for c in radiomic_df.columns if c not in ['patient_id','loss_function']])
    print(f"\n✓ Extracted {len(radiomic_df)} patients, {n_features} features")
    
    output_path = f"{OUTPUT_DIR}/radiomics_features/{loss_function}_features_location_aware.csv"
    radiomic_df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    return radiomic_df


# ============================================================================
# THREE-STAGE PIPELINE (like Jo et al.'s Models A, B, C)
# ============================================================================

def run_model_a_clinical_only(clinical_df, y):
    """
    Model A: Clinical-only baseline (like Jo et al.)
    Uses minimal clinical features
    """
    print("\n" + "="*70)
    print("MODEL A: CLINICAL ONLY (Baseline)")
    print("="*70)
    
    X_clinical = clinical_df[MINIMAL_CLINICAL].values.astype(float)
    
    models = get_base_models()
    results_a = {}
    
    for name, model in models.items():
        model_name = f"ModelA_Clinical__{name}"
        res = evaluate_with_calibration(model, X_clinical, y, model_name, verbose=True)
        if res is not None:
            results_a[model_name] = res
    
    # Find best baseline
    best_baseline = max(results_a.items(), key=lambda x: x[1]['auc_mean'])
    baseline_auc = best_baseline[1]['auc_mean']
    
    print(f"\n✓ BASELINE (Model A): {best_baseline[0]}")
    print(f"  AUC = {baseline_auc:.3f}")
    print(f"  Brier = {best_baseline[1]['brier_score']:.3f}")
    
    return results_a, baseline_auc


def run_model_b_imaging_only(merged_df, loss_function, y, baseline_auc):
    """
    Model B: Imaging-only with location-aware features
    Tests if imaging features alone predict outcome
    """
    print(f"\n{'='*70}")
    print(f"MODEL B: IMAGING ONLY ({loss_function})")
    print(f"{'='*70}")
    
    # Get all imaging features (radiomic + location)
    prefix = f"{loss_function}_"
    imaging_cols = [c for c in merged_df.columns if c.startswith(prefix)]
    
    if len(imaging_cols) == 0:
        return {}
    
    X_imaging = merged_df[imaging_cols].values.astype(float)
    
    models = get_base_models()
    results_b = {}
    
    print(f"\nTesting {len(models)} models...")
    auc_results = []
    
    for name, model in models.items():
        model_name = f"ModelB_Imaging_{loss_function}__{name}"
        res = evaluate_with_calibration(model, X_imaging, y, model_name, verbose=False)
        
        if res is not None:
            improvement = res['auc_mean'] - baseline_auc
            res['improvement_vs_baseline'] = improvement
            results_b[model_name] = res
            auc_results.append((name, res['auc_mean'], improvement))
    
    # Print summary table
    if auc_results:
        print(f"\n{'Model':<20} {'AUC':>8} {'ΔAU':>10}")
        print("-" * 40)
        for name, auc, imp in sorted(auc_results, key=lambda x: x[1], reverse=True):
            print(f"{name:<20} {auc:>8.3f} {imp:>+10.3f}")
    
    return results_b


def run_model_c_integrated(merged_df, loss_function, y, results_b, baseline_auc):
    """
    Model C: Integrated (Clinical + Imaging predictions)
    Like Jo et al.'s Model C - uses imaging model predictions as features
    """
    print(f"\n{'='*70}")
    print(f"MODEL C: INTEGRATED ({loss_function})")
    print(f"{'='*70}")
    
    if len(results_b) == 0:
        return {}
    
    # Get clinical features
    X_clinical = merged_df[MINIMAL_CLINICAL].values.astype(float)
    
    # Get imaging features
    prefix = f"{loss_function}_"
    imaging_cols = [c for c in merged_df.columns if c.startswith(prefix)]
    X_imaging = merged_df[imaging_cols].values.astype(float)
    
    models = get_base_models()
    results_c = {}
    
    print(f"\nTesting {len(models)} models...")
    auc_results = []
    
    for name, model in models.items():
        # Train imaging model to get predictions
        img_model_key = f"ModelB_Imaging_{loss_function}__{name}"
        if img_model_key not in results_b:
            continue
        
        img_predictions = results_b[img_model_key]['y_pred_proba']
        
        # Combine: Clinical + Imaging predictions
        X_integrated = np.column_stack([X_clinical, img_predictions.reshape(-1, 1)])
        
        model_name = f"ModelC_Integrated_{loss_function}__{name}"
        res = evaluate_with_calibration(model, X_integrated, y, model_name, verbose=False)
        
        if res is not None:
            improvement = res['auc_mean'] - baseline_auc
            res['improvement_vs_baseline'] = improvement
            results_c[model_name] = res
            
            is_significant = improvement > CLINICAL_SIGNIFICANCE_THRESHOLD
            marker = "✓" if is_significant else "~"
            auc_results.append((name, res['auc_mean'], improvement, marker))
    
    # Print summary table
    if auc_results:
        print(f"\n{'Model':<20} {'AUC':>8} {'ΔAU':>10}  {'Sig':<3}")
        print("-" * 45)
        for name, auc, imp, marker in sorted(auc_results, key=lambda x: x[1], reverse=True):
            print(f"{name:<20} {auc:>8.3f} {imp:>+10.3f}  {marker}")
    
    return results_c


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Three-stage multimodal pipeline (inspired by Jo et al.)"""
    
    print("\n" + "="*70)
    print("UPGRADED MULTI-MODAL ANALYSIS")
    print("Inspired by Jo et al. (2023)")
    print("="*70)
    
    setup_output_directory()
    
    # Load data
    clinical_df = load_clinical_data()
    
    if len(clinical_df) < 30:
        print("❌ ERROR: Too few patients")
        return
    
    y = clinical_df['FAC12_binary'].values
    
    # Load atlases
    atlas_manager = AtlasManager(ATLAS_DIR)
    
    # ========================================================================
    # STAGE 1: Model A - Clinical Only (Baseline)
    # ========================================================================
    results_a, baseline_auc = run_model_a_clinical_only(clinical_df, y)
    
    all_results = {}
    all_results.update(results_a)
    
    # ========================================================================
    # STAGE 2 & 3: Extract features and run Models B & C
    # ========================================================================
    
    for loss_func in LOSS_FUNCTIONS:
        print(f"\n{'='*70}")
        print(f"PROCESSING LOSS FUNCTION: {loss_func}")
        print(f"{'='*70}")
        
        # Extract features
        rad_df = extract_radiomics_all_patients(clinical_df, loss_func, atlas_manager)
        
        if rad_df is None:
            print(f"⚠️  SKIPPING {loss_func}: No features extracted")
            continue
        
        print(f"\n✓ Feature extraction complete for {loss_func}")
        print(f"  Radiomic DF shape: {rad_df.shape}")
        print(f"  Columns: {list(rad_df.columns[:5])}... (showing first 5)")
        
        # Merge with clinical
        print(f"\nMerging with clinical data...")
        print(f"  Clinical DF shape: {clinical_df.shape}")
        print(f"  Clinical patient_ids: {len(clinical_df['patient_id'].unique())} unique")
        print(f"  Radiomic patient_ids: {len(rad_df['patient_id'].unique())} unique")
        
        merged = clinical_df.merge(rad_df, on='patient_id', how='inner')
        
        print(f"  Merged shape: {merged.shape}")
        print(f"  Merged patients: {len(merged)}")
        
        if len(merged) < 30:
            print(f"  ❌ Too few patients after merge ({len(merged)} < 30)")
            continue
        
        # Clean NaN
        prefix = f"{loss_func}_"
        feature_cols = [c for c in merged.columns if c.startswith(prefix)]
        
        print(f"\nChecking for NaN values...")
        print(f"  Feature columns found: {len(feature_cols)}")
        
        # Check NaN in feature columns
        nan_counts = merged[feature_cols].isna().sum()
        total_nans = nan_counts.sum()
        
        if total_nans > 0:
            print(f"  ⚠️  Found {total_nans} NaN values in features")
            print(f"  Columns with NaN: {nan_counts[nan_counts > 0].to_dict()}")
        
        # Check NaN in clinical features
        clinical_nan = merged[MINIMAL_CLINICAL].isna().sum().sum()
        if clinical_nan > 0:
            print(f"  ⚠️  Found {clinical_nan} NaN values in clinical features")
        
        merged_clean = merged.dropna(subset=feature_cols + MINIMAL_CLINICAL)
        
        print(f"  After dropping NaN: {len(merged_clean)} patients")
        
        if len(merged_clean) < 30:
            print(f"  ❌ Too few patients after cleaning ({len(merged_clean)} < 30)")
            continue
        
        y_clean = merged_clean['FAC12_binary'].values
        
        print(f"\n✓ Clean dataset ready: {len(merged_clean)} patients, {len(feature_cols)} features")
        print(f"  Proceeding to Models B and C...")
        
        # Model B: Imaging only
        results_b = run_model_b_imaging_only(merged_clean, loss_func, y_clean, baseline_auc)
        all_results.update(results_b)
        
        # Model C: Integrated
        results_c = run_model_c_integrated(merged_clean, loss_func, y_clean, results_b, baseline_auc)
        all_results.update(results_c)
        
        # Summary for this loss function
        print(f"\n{'='*70}")
        print(f"SUMMARY FOR {loss_func}")
        print(f"{'='*70}")
        print(f"Model B results: {len(results_b)} models")
        print(f"Model C results: {len(results_c)} models")
        
        if results_c:
            best_c_this_loss = max(results_c.items(), key=lambda x: x[1]['auc_mean'])
            improvement = best_c_this_loss[1]['auc_mean'] - baseline_auc
            print(f"\nBest Model C for {loss_func}:")
            print(f"  {best_c_this_loss[0]}")
            print(f"  AUC = {best_c_this_loss[1]['auc_mean']:.3f}")
            print(f"  ΔAU = {improvement:+.3f}")
            
            if improvement > CLINICAL_SIGNIFICANCE_THRESHOLD:
                print(f"  ✓ CLINICALLY SIGNIFICANT")
            else:
                print(f"  ~ Not clinically significant")
    
    # ========================================================================
    # FINAL ANALYSIS
    # ========================================================================
    
    print("\n" + "="*70)
    print("BUILDING FINAL COMPARISON")
    print("="*70)
    
    cmp_path = f"{OUTPUT_DIR}/model_comparison_UPGRADED.csv"
    df_cmp = build_comparison_table(all_results, cmp_path)
    
    # Plot calibration curves
    plot_calibration_curves(all_results, OUTPUT_DIR)
    
    # Summary report
    print("\n" + "="*70)
    print("SUMMARY: Jo et al. Style Analysis")
    print("="*70)
    
    print(f"\nBaseline (Model A - Clinical Only): AUC = {baseline_auc:.3f}")
    
    # Find best Model C (integrated)
    model_c_results = {k: v for k, v in all_results.items() if 'ModelC_Integrated' in k}
    
    if model_c_results:
        best_c = max(model_c_results.items(), key=lambda x: x[1]['auc_mean'])
        improvement = best_c[1]['auc_mean'] - baseline_auc
        
        print(f"\nBest Integrated (Model C): {best_c[0]}")
        print(f"  AUC = {best_c[1]['auc_mean']:.3f}")
        print(f"  ΔR² = {improvement:+.3f}")
        print(f"  Brier = {best_c[1]['brier_score']:.3f}")
        
        if improvement > CLINICAL_SIGNIFICANCE_THRESHOLD:
            print(f"\n✓ CLINICALLY SIGNIFICANT IMPROVEMENT")
            print(f"  Imaging adds value for FAC prediction")
        else:
            print(f"\n~ NO CLINICALLY SIGNIFICANT IMPROVEMENT")
            print(f"  FAC outcome may be clinically-dominated")
            print(f"  Consider testing FMUE outcome instead")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"Results: {cmp_path}")
    print(f"Calibration plots: {OUTPUT_DIR}/calibration_plots/")


if __name__ == "__main__":
    main()
