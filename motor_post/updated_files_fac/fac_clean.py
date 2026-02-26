#!/usr/bin/env python3
"""
Clean Radiomic Analysis for FAC Outcome - Jo et al. (2023) Approach
NO ATLASES - Pure radiomics + clinical integration

This is the CORRECT approach:
- Model A: Clinical only (Age, NIHSS)
- Model B: Radiomics only (shape, texture from lesion)
- Model C: Integrated (Clinical + Radiomics predictions)

Author: Parvez (cleaned)
Date: 2025-01-13
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, 
    precision_score, f1_score, balanced_accuracy_score, 
    brier_score_loss, classification_report
)
from sklearn.calibration import calibration_curve

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = "/home/pahm409/pytorch_loss_comparison_results2/"
T1_IMAGE_DIR = "/hpc/pahm409/harvard/UOA/FNIRT/train/"
CLINICAL_DATA_PATH = "/hpc/pahm409/harvard/UOA/IMPRESS_RETROSPECTIVE_CLINICAL_DATA.xlsx"
OUTPUT_DIR = "./radiomic_fac_CLEAN/"

LOSS_FUNCTIONS = ['AdaptiveRegional', 'GDice', 'BCEDice', 'BCEFocalTversky', 'Dice', 'Focal', 'FocalTversky', 'Tversky']

# Jo et al. approach: Minimal clinical features
MINIMAL_CLINICAL = ['AGE', 'NIHSS']  # Like Jo et al. (Age, NIHSS, END)

# Radiomic extraction mode
# For n=57 patients, MINIMAL mode is STRONGLY recommended
EXTRACTION_MODE = 'minimal'  # Options: 'minimal' (~20 features), 'core' (~90), 'comprehensive' (~150)

# Feature selection: Test SMALL k values for n=57
# For n=57 patients with 5-fold CV (~11 per fold):
#   - Top-3:  19 patients/feature (VERY SAFE)
#   - Top-5:  11.4 patients/feature (SAFE)
#   - Top-7:  8.1 patients/feature (OK)
#   - Top-10: 5.7 patients/feature (RISKY)
FEATURE_SELECTION_K_VALUES = [3, 5, 7]  # Start with VERY small k values

# XGBoost params
XGBOOST_PARAMS = {
    'max_depth': 3,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'random_state': 42,
    'eval_metric': 'auc',
    'use_label_encoder': False,
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'predictor': 'gpu_predictor'
}

CV_FOLDS = 5
CV_RANDOM_STATE = 42
CLINICAL_SIGNIFICANCE_THRESHOLD = 0.01

# ============================================================================
# SETUP
# ============================================================================

def setup_output_directory():
    """Create output directory structure"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/radiomics_features", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/results", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}")


def get_models():
    """
    Define classifiers optimized for SMALL datasets (n=57)
    
    Key: Strong regularization to prevent overfitting
    """
    from sklearn.neural_network import MLPClassifier
    import lightgbm as lgb
    
    return {
        # Logistic Regression with STRONG L2 regularization
        "LogisticRegression": LogisticRegression(
            penalty='l2',
            C=0.1,  # Strong regularization (default=1.0)
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear',
            random_state=42
        ),
        
        # Random Forest with MUCH FEWER trees and constraints
        "RandomForest": RandomForestClassifier(
            n_estimators=50,  # Reduced from 300
            max_depth=3,      # Shallow trees
            min_samples_split=10,  # Need 10 samples to split
            min_samples_leaf=5,    # Need 5 samples per leaf
            class_weight='balanced',
            random_state=42
        ),
        
        # LightGBM with strong regularization (better than XGBoost for small data)
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.01,  # Very slow learning
            num_leaves=8,        # Few leaves
            min_child_samples=10,  # Need 10 samples per leaf
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,       # L1 regularization
            reg_lambda=1.0,      # L2 regularization
            class_weight='balanced',
            random_state=42,
            verbose=-1
        ),
        
        # Simple MLP with strong regularization
        "MLP": MLPClassifier(
            hidden_layer_sizes=(10,),  # Single hidden layer with 10 neurons
            activation='relu',
            solver='adam',
            alpha=1.0,  # Strong L2 regularization (default=0.0001)
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=42
        ),
    }


# ============================================================================
# CLINICAL DATA
# ============================================================================

def load_clinical_data():
    """Load and preprocess clinical data"""
    print("\n" + "="*70)
    print("LOADING CLINICAL DATA")
    print("="*70)
    
    df = pd.read_excel(CLINICAL_DATA_PATH)
    df = df.rename(columns={'IMPRESS code': 'patient_id'})
    
    # Filter valid FAC12
    df['FAC12'] = pd.to_numeric(df['FAC12'], errors='coerce')
    df = df[df['FAC12'].notna()].copy()
    df['FAC12_binary'] = (df['FAC12'] >= 4).astype(int)
    
    # Preprocess minimal clinical features
    for feat in MINIMAL_CLINICAL:
        if feat in df.columns:
            df[feat] = pd.to_numeric(df[feat], errors='coerce')
            median_val = df[feat].median()
            df[feat] = df[feat].fillna(median_val)
            print(f"  ✓ {feat}: mean={df[feat].mean():.1f}")
    
    print(f"\n✓ {len(df)} patients")
    print(f"  Outcome: FAC≥4 = {df['FAC12_binary'].sum()} ({df['FAC12_binary'].mean()*100:.1f}%)")
    
    return df


# ============================================================================
# RADIOMIC EXTRACTION (NO ATLASES!)
# ============================================================================

def get_ensemble_mask(patient_id, loss_function):
    """Get ensemble mask across replications"""
    fold = None
    
    for fold_idx in range(1, 6):
        fold_dir = Path(f"{BASE_DIR}/fold_{fold_idx}")
        if not fold_dir.exists():
            continue
        
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
    
    # Ensemble
    ensemble_array = np.mean(masks, axis=0)
    ensemble_binary = (ensemble_array > 0.5).astype(np.uint8)
    
    ensemble_mask = sitk.GetImageFromArray(ensemble_binary)
    ensemble_mask.CopyInformation(mask_ref)
    
    return ensemble_mask


def extract_radiomics_simple(patient_id, t1_image, mask, loss_function, comprehensive=False, minimal=False):
    """
    Extract radiomics features - NO atlases!
    
    Args:
        comprehensive (bool): Extract all feature classes (~150 features)
        minimal (bool): Extract only essential features (~20 features)
                       This is BEST for n=57 patients!
    
    Following Jo et al. approach:
    - Shape features (volume, surface area, sphericity)
    - First-order features (intensity statistics)
    - Texture features (GLCM, GLRLM, GLSZM, etc.)
    
    These capture lesion characteristics without needing location info
    """
    
    features = {'patient_id': patient_id, 'loss_function': loss_function}
    prefix = f"{loss_function}_"
    
    try:
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
        
        # Configure radiomics extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.settings['geometryTolerance'] = 1e-3
        
        # Enable feature classes based on mode
        extractor.enableImageTypeByName('Original')
        
        if minimal:
            # MINIMAL mode: Only most clinically relevant features (~20 features)
            # Best for n=57 patients!
            extractor.enableFeatureClassByName('shape')        # ~14: Volume, sphericity, etc.
            extractor.enableFeatureClassByName('firstorder')   # ~18: Mean, median, entropy
            # Skip texture features entirely - too many and often noisy
            
        elif comprehensive:
            # COMPREHENSIVE mode: All features (~150)
            extractor.enableFeatureClassByName('shape')
            extractor.enableFeatureClassByName('firstorder')
            extractor.enableFeatureClassByName('glcm')
            extractor.enableFeatureClassByName('glrlm')
            extractor.enableFeatureClassByName('glszm')
            extractor.enableFeatureClassByName('gldm')
            extractor.enableFeatureClassByName('ngtdm')
            
        else:
            # CORE mode: Essential features (~90)
            extractor.enableFeatureClassByName('shape')
            extractor.enableFeatureClassByName('firstorder')
            extractor.enableFeatureClassByName('glcm')
            extractor.enableFeatureClassByName('glrlm')
            extractor.enableFeatureClassByName('glszm')
        
        # Extract
        radiomic_features = extractor.execute(t1_image, mask)
        
        # Store only numeric features
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
        print(f"    Error: {e}")
        return None


def extract_all_patients(clinical_df, loss_function, mode='minimal'):
    """
    Extract radiomics for all patients
    
    Args:
        mode (str): 'minimal' (~20 features), 'core' (~90), or 'comprehensive' (~150)
    """
    
    print(f"\n{'='*70}")
    print(f"EXTRACTING RADIOMICS: {loss_function}")
    print(f"Mode: {mode.upper()}")
    print(f"{'='*70}")
    
    radiomic_data = []
    patient_ids = clinical_df['patient_id'].unique()
    
    for idx, patient_id in enumerate(patient_ids, 1):
        if idx % 10 == 0 or idx <= 5:
            print(f"[{idx}/{len(patient_ids)}] {patient_id}...", end=' ')
        
        # Load T1
        t1_path = f"{T1_IMAGE_DIR}/{patient_id}_T1_FNIRT_MNI.nii.gz"
        if not os.path.exists(t1_path):
            if idx % 10 == 0 or idx <= 5:
                print("✗ No T1")
            continue
        
        t1_image = sitk.ReadImage(t1_path)
        
        # Get ensemble mask
        mask = get_ensemble_mask(patient_id, loss_function)
        if mask is None:
            if idx % 10 == 0 or idx <= 5:
                print("✗ No mask")
            continue
        
        # Extract radiomics based on mode
        comprehensive = (mode == 'comprehensive')
        minimal = (mode == 'minimal')
        features = extract_radiomics_simple(patient_id, t1_image, mask, loss_function, 
                                           comprehensive=comprehensive, minimal=minimal)
        
        if features is not None and len(features) > 3:  # More than just ID and loss_func
            radiomic_data.append(features)
            if idx % 10 == 0 or idx <= 5:
                print("✓")
        else:
            if idx % 10 == 0 or idx <= 5:
                print("✗ Failed")
    
    if len(radiomic_data) == 0:
        print(f"✗ No features extracted for {loss_function}")
        return None
    
    radiomic_df = pd.DataFrame(radiomic_data)
    n_features = len([c for c in radiomic_df.columns if c not in ['patient_id','loss_function']])
    print(f"\n✓ Extracted {len(radiomic_df)} patients, {n_features} radiomic features")
    
    # Save
    output_path = f"{OUTPUT_DIR}/radiomics_features/{loss_function}_{mode}.csv"
    radiomic_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    
    return radiomic_df


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X, y, model_name, verbose=False):
    """Evaluate model with proper CV (no data leakage)"""
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*70}")
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    # Ensure numeric
    X = X.astype(float)
    
    if np.any(np.isnan(X)):
        print(f"✗ NaN in features")
        return None
    
    # CV with scaling inside each fold
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    
    y_pred_proba_all = np.zeros(len(y))
    auc_scores = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale ONLY on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        try:
            model_fold = model.__class__(**model.get_params())
            model_fold.fit(X_train_scaled, y_train)
            
            if hasattr(model_fold, "predict_proba"):
                proba = model_fold.predict_proba(X_test_scaled)[:, 1]
            else:
                df = model_fold.decision_function(X_test_scaled)
                proba = 1.0 / (1.0 + np.exp(-df))
            
            y_pred_proba_all[test_idx] = proba
            auc_scores.append(roc_auc_score(y_test, proba))
            
        except Exception as e:
            if verbose:
                print(f"  Fold {fold_idx+1} failed: {e}")
            continue
    
    if len(auc_scores) == 0:
        return None
    
    auc_mean = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    
    # Optimal threshold
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba_all)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]
    y_pred = (y_pred_proba_all >= optimal_threshold).astype(int)
    
    # Metrics
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y, y_pred)
    brier = brier_score_loss(y, y_pred_proba_all)
    
    if verbose:
        print(f"✓ AUC = {auc_mean:.3f} ± {auc_std:.3f}")
        print(f"  Brier = {brier:.3f}")
        print(f"  Sensitivity = {sensitivity:.3f}, Specificity = {specificity:.3f}")
    
    return {
        'model_name': model_name,
        'auc_mean': auc_mean,
        'auc_std': auc_std,
        'auc_scores': auc_scores,
        'brier_score': brier,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'balanced_acc': bal_acc,
        'y_true': y,
        'y_pred_proba': y_pred_proba_all,
    }


# ============================================================================
# THREE-STAGE PIPELINE (Jo et al.)
# ============================================================================

def run_model_a(clinical_df, y):
    """Model A: Clinical only"""
    print("\n" + "="*70)
    print("MODEL A: CLINICAL ONLY")
    print("="*70)
    
    X_clinical = clinical_df[MINIMAL_CLINICAL].values.astype(float)
    
    models = get_models()
    results = {}
    
    print(f"\nFeatures: {MINIMAL_CLINICAL}")
    print(f"Testing {len(models)} models...")
    
    for name, model in models.items():
        res = evaluate_model(model, X_clinical, y, f"ModelA_{name}", verbose=False)
        if res is not None:
            results[f"ModelA_{name}"] = res
            print(f"  {name:<20} AUC = {res['auc_mean']:.3f} ± {res['auc_std']:.3f}")
    
    # Find best
    best = max(results.items(), key=lambda x: x[1]['auc_mean'])
    baseline_auc = best[1]['auc_mean']
    
    print(f"\n✓ BASELINE: {best[0]}, AUC = {baseline_auc:.3f}")
    
    return results, baseline_auc


def run_model_b(merged_df, loss_function, y, baseline_auc):
    """Model B: Radiomics only"""
    print(f"\n{'='*70}")
    print(f"MODEL B: RADIOMICS ONLY ({loss_function})")
    print(f"{'='*70}")
    
    prefix = f"{loss_function}_"
    radiomic_cols = [c for c in merged_df.columns if c.startswith(prefix)]
    
    if len(radiomic_cols) == 0:
        print("✗ No radiomic features found")
        return {}
    
    X_radiomics = merged_df[radiomic_cols].values.astype(float)
    
    models = get_models()
    results = {}
    
    print(f"\nTesting {len(models)} models with {len(radiomic_cols)} radiomic features...")
    
    for name, model in models.items():
        res = evaluate_model(model, X_radiomics, y, f"ModelB_{loss_function}_{name}", verbose=False)
        if res is not None:
            improvement = res['auc_mean'] - baseline_auc
            res['improvement_vs_baseline'] = improvement
            results[f"ModelB_{loss_function}_{name}"] = res
            print(f"  {name:<20} AUC = {res['auc_mean']:.3f} (Δ = {improvement:+.3f})")
    
    return results


def select_top_k_features(X, y, feature_names, k=10):
    """
    Select top-k most informative features using Mutual Information
    
    This is CRITICAL for small datasets (n=57):
    - Reduces overfitting
    - Improves generalization
    - Matches Jo et al.'s approach (minimal features)
    """
    from sklearn.feature_selection import mutual_info_classif
    
    # Compute MI scores
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Get top-k indices
    top_k_idx = np.argsort(mi_scores)[-k:]
    top_k_features = [feature_names[i] for i in top_k_idx]
    top_k_scores = mi_scores[top_k_idx]
    
    print(f"\n  Top-{k} Selected Features (by Mutual Information):")
    for feat, score in sorted(zip(top_k_features, top_k_scores), key=lambda x: x[1], reverse=True):
        print(f"    {feat}: {score:.4f}")
    
    return top_k_idx, top_k_features


def run_model_c_with_feature_selection(merged_df, loss_function, y, results_b, baseline_auc, k_values=[5, 10, 20]):
    """
    Model C: Integrated with PROPER feature selection (no data leakage)
    Feature selection happens INSIDE each CV fold
    """
    print(f"\n{'='*70}")
    print(f"MODEL C: INTEGRATED WITH FEATURE SELECTION ({loss_function})")
    print(f"{'='*70}")
    
    if len(results_b) == 0:
        return {}
    
    X_clinical = merged_df[MINIMAL_CLINICAL].values.astype(float)
    
    prefix = f"{loss_function}_"
    radiomic_cols = [c for c in merged_df.columns if c.startswith(prefix)]
    X_radiomics_all = merged_df[radiomic_cols].values.astype(float)
    
    models = get_models()
    results = {}
    
    # Test different k values
    for k in k_values:
        if k > len(radiomic_cols):
            print(f"\n⚠️  k={k} exceeds available features ({len(radiomic_cols)}), skipping")
            continue
        
        print(f"\n{'='*70}")
        print(f"Testing with Top-{k} Radiomic Features")
        print(f"{'='*70}")
        
        # Do feature selection on FULL dataset just to show which features are selected
        # (for interpretation only, NOT used in model)
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X_radiomics_all, y, random_state=42)
        top_k_idx = np.argsort(mi_scores)[-k:]
        top_k_features = [radiomic_cols[i] for i in top_k_idx]
        top_k_scores = mi_scores[top_k_idx]
        
        print(f"\n  Top-{k} Features (for interpretation):")
        for feat, score in sorted(zip(top_k_features, top_k_scores), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {feat}: {score:.4f}")
        if k > 5:
            print(f"    ... and {k-5} more")
        
        print(f"\nTraining models with PROPER cross-validation...")
        print(f"(Feature selection done INSIDE each fold - no data leakage)")
        
        for name, model in models.items():
            model_name = f"ModelC_{loss_function}_Top{k}_{name}"
            
            # Proper CV with feature selection INSIDE each fold
            res = evaluate_model_with_feature_selection(
                model, X_clinical, X_radiomics_all, y, k, model_name
            )
            
            if res is not None:
                improvement = res['auc_mean'] - baseline_auc
                res['improvement_vs_baseline'] = improvement
                res['k_features'] = k
                res['selected_features'] = top_k_features  # For interpretation only
                results[model_name] = res
                
                marker = "✓" if improvement > CLINICAL_SIGNIFICANCE_THRESHOLD else "~"
                print(f"  [{marker}] {name:<20} AUC = {res['auc_mean']:.3f} (Δ = {improvement:+.3f})")
    
    return results


def evaluate_model_with_feature_selection(model, X_clinical, X_radiomics_all, y, k, model_name):
    """
    Evaluate model with feature selection INSIDE CV loop (no data leakage)
    
    Critical: Feature selection must happen on training data only!
    """
    from sklearn.feature_selection import mutual_info_classif
    
    # Ensure numeric
    X_clinical = X_clinical.astype(float)
    X_radiomics_all = X_radiomics_all.astype(float)
    
    if np.any(np.isnan(X_clinical)) or np.any(np.isnan(X_radiomics_all)):
        print(f"✗ NaN in features")
        return None
    
    # CV with feature selection INSIDE each fold
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    
    y_pred_proba_all = np.zeros(len(y))
    auc_scores = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_clinical)):
        # Split data
        X_clin_train, X_clin_test = X_clinical[train_idx], X_clinical[test_idx]
        X_rad_train, X_rad_test = X_radiomics_all[train_idx], X_radiomics_all[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # STEP 1: Select top-k features on TRAINING data only
        mi_scores = mutual_info_classif(X_rad_train, y_train, random_state=42)
        top_k_idx = np.argsort(mi_scores)[-k:]
        
        X_rad_train_selected = X_rad_train[:, top_k_idx]
        X_rad_test_selected = X_rad_test[:, top_k_idx]
        
        # STEP 2: Train imaging model on TRAINING data only
        scaler_rad = StandardScaler()
        X_rad_train_scaled = scaler_rad.fit_transform(X_rad_train_selected)
        X_rad_test_scaled = scaler_rad.transform(X_rad_test_selected)
        
        try:
            img_model = model.__class__(**model.get_params())
            img_model.fit(X_rad_train_scaled, y_train)
            
            # Get imaging predictions
            if hasattr(img_model, "predict_proba"):
                img_pred_train = img_model.predict_proba(X_rad_train_scaled)[:, 1]
                img_pred_test = img_model.predict_proba(X_rad_test_scaled)[:, 1]
            else:
                img_pred_train = img_model.decision_function(X_rad_train_scaled)
                img_pred_test = img_model.decision_function(X_rad_test_scaled)
                # Normalize to 0-1
                img_pred_train = 1.0 / (1.0 + np.exp(-img_pred_train))
                img_pred_test = 1.0 / (1.0 + np.exp(-img_pred_test))
        except Exception as e:
            continue
        
        # STEP 3: Combine clinical + imaging predictions
        X_integrated_train = np.column_stack([X_clin_train, img_pred_train.reshape(-1, 1)])
        X_integrated_test = np.column_stack([X_clin_test, img_pred_test.reshape(-1, 1)])
        
        # STEP 4: Scale clinical features
        scaler_integrated = StandardScaler()
        X_integrated_train_scaled = scaler_integrated.fit_transform(X_integrated_train)
        X_integrated_test_scaled = scaler_integrated.transform(X_integrated_test)
        
        # STEP 5: Train final integrated model
        try:
            final_model = model.__class__(**model.get_params())
            final_model.fit(X_integrated_train_scaled, y_train)
            
            if hasattr(final_model, "predict_proba"):
                proba = final_model.predict_proba(X_integrated_test_scaled)[:, 1]
            else:
                df = final_model.decision_function(X_integrated_test_scaled)
                proba = 1.0 / (1.0 + np.exp(-df))
            
            y_pred_proba_all[test_idx] = proba
            auc_scores.append(roc_auc_score(y_test, proba))
            
        except Exception as e:
            continue
    
    if len(auc_scores) == 0:
        return None
    
    auc_mean = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    
    # Calculate other metrics
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba_all)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]
    y_pred = (y_pred_proba_all >= optimal_threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y, y_pred)
    brier = brier_score_loss(y, y_pred_proba_all)
    
    return {
        'model_name': model_name,
        'auc_mean': auc_mean,
        'auc_std': auc_std,
        'auc_scores': auc_scores,
        'brier_score': brier,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'balanced_acc': bal_acc,
        'y_true': y,
        'y_pred_proba': y_pred_proba_all,
    }


# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

def build_comparison_table(all_results, output_path):
    """Build comparison table"""
    rows = []
    for key, res in all_results.items():
        rows.append({
            'Model': key,
            'AUC_mean': res['auc_mean'],
            'AUC_std': res['auc_std'],
            'Brier': res['brier_score'],
            'Sensitivity': res['sensitivity'],
            'Specificity': res['specificity'],
            'F1': res['f1'],
        })
    
    df = pd.DataFrame(rows).sort_values('AUC_mean', ascending=False)
    
    print("\n" + "="*70)
    print("TOP 15 MODELS")
    print("="*70)
    print(df[['Model', 'AUC_mean', 'Brier', 'Sensitivity', 'Specificity']].head(15).to_string(index=False))
    
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")
    
    return df


def plot_results(all_results, baseline_auc, output_dir):
    """Plot comparison"""
    
    # Extract Model C results
    model_c = {k: v for k, v in all_results.items() if 'ModelC' in k}
    
    if len(model_c) == 0:
        print("No Model C results to plot")
        return
    
    # Prepare data
    loss_funcs = []
    improvements = []
    aucs = []
    
    for loss_func in LOSS_FUNCTIONS:
        loss_models = {k: v for k, v in model_c.items() if loss_func in k}
        if loss_models:
            best = max(loss_models.items(), key=lambda x: x[1]['auc_mean'])
            loss_funcs.append(loss_func)
            improvements.append(best[1]['auc_mean'] - baseline_auc)
            aucs.append(best[1]['auc_mean'])
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: AUC comparison
    ax1 = axes[0]
    colors = ['#2ecc71' if imp > CLINICAL_SIGNIFICANCE_THRESHOLD else '#95a5a6' for imp in improvements]
    ax1.barh(loss_funcs, aucs, color=colors)
    ax1.axvline(x=baseline_auc, color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline_auc:.3f})')
    ax1.set_xlabel('AUC', fontsize=12)
    ax1.set_title('Model C Performance by Loss Function', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Improvement
    ax2 = axes[1]
    colors = ['#27ae60' if imp > CLINICAL_SIGNIFICANCE_THRESHOLD else '#e67e22' for imp in improvements]
    ax2.barh(loss_funcs, improvements, color=colors)
    ax2.axvline(x=CLINICAL_SIGNIFICANCE_THRESHOLD, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax2.set_xlabel('Improvement (ΔR²)', fontsize=12)
    ax2.set_title('Clinical Value Added', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{output_dir}/comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {plot_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main pipeline"""
    
    print("\n" + "="*70)
    print("CLEAN RADIOMIC ANALYSIS (Jo et al. Approach)")
    print("NO ATLASES - Pure Radiomics")
    print("="*70)
    
    setup_output_directory()
    
    # Load clinical data
    clinical_df = load_clinical_data()
    
    if len(clinical_df) < 30:
        print("❌ Too few patients")
        return
    
    y = clinical_df['FAC12_binary'].values
    
    # Model A: Clinical baseline
    results_a, baseline_auc = run_model_a(clinical_df, y)
    
    all_results = {}
    all_results.update(results_a)
    
    # Models B & C for each loss function
    for loss_func in LOSS_FUNCTIONS:
        print(f"\n{'='*70}")
        print(f"PROCESSING: {loss_func}")
        print(f"{'='*70}")
        
        # Extract radiomics
        rad_df = extract_all_patients(clinical_df, loss_func, mode=EXTRACTION_MODE)
        
        if rad_df is None:
            print(f"⚠️  Skipping {loss_func}")
            continue
        
        # Merge
        merged = clinical_df.merge(rad_df, on='patient_id', how='inner')
        
        if len(merged) < 30:
            print(f"❌ Too few patients after merge: {len(merged)}")
            continue
        
        # Clean
        prefix = f"{loss_func}_"
        feature_cols = [c for c in merged.columns if c.startswith(prefix)]
        
        # Fill NaN with 0 (should not happen with pure radiomics, but safety check)
        merged_clean = merged.copy()
        merged_clean[feature_cols] = merged_clean[feature_cols].fillna(0)
        merged_clean[MINIMAL_CLINICAL] = merged_clean[MINIMAL_CLINICAL].fillna(
            merged_clean[MINIMAL_CLINICAL].median()
        )
        
        y_clean = merged_clean['FAC12_binary'].values
        
        print(f"\n✓ Ready: {len(merged_clean)} patients, {len(feature_cols)} features")
        
        # Model B: Radiomics only (all features)
        results_b = run_model_b(merged_clean, loss_func, y_clean, baseline_auc)
        all_results.update(results_b)
        
        # Model C: Integrated with feature selection (test multiple k values)
        results_c = run_model_c_with_feature_selection(
            merged_clean, loss_func, y_clean, results_b, baseline_auc, 
            k_values=FEATURE_SELECTION_K_VALUES
        )
        all_results.update(results_c)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    output_csv = f"{OUTPUT_DIR}/results/comparison_CLEAN.csv"
    df_results = build_comparison_table(all_results, output_csv)
    
    plot_results(all_results, baseline_auc, OUTPUT_DIR)
    
    # Find best Model C
    model_c_results = {k: v for k, v in all_results.items() if 'ModelC' in k}
    
    if model_c_results:
        best_c = max(model_c_results.items(), key=lambda x: x[1]['auc_mean'])
        improvement = best_c[1]['auc_mean'] - baseline_auc
        
        print(f"\n{'='*70}")
        print("BEST MODEL")
        print(f"{'='*70}")
        print(f"Baseline (Clinical): {baseline_auc:.3f}")
        print(f"Best Integrated:     {best_c[0]}")
        print(f"  AUC = {best_c[1]['auc_mean']:.3f}")
        print(f"  ΔR² = {improvement:+.3f}")
        print(f"  Brier = {best_c[1]['brier_score']:.3f}")
        
        if improvement > CLINICAL_SIGNIFICANCE_THRESHOLD:
            print(f"\n✓ CLINICALLY SIGNIFICANT IMPROVEMENT!")
            print(f"  Radiomics add value for FAC prediction")
        else:
            print(f"\n~ No clinically significant improvement")
            print(f"  Clinical features (Age, NIHSS) are sufficient")
            print(f"  This is a VALID finding with small sample (n={len(clinical_df)})")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"Results: {output_csv}")


if __name__ == "__main__":
    main()
