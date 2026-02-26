#!/usr/bin/env python3
"""
Radiomic Features for Walking (FAC) Outcome Prediction
Multi-classifier comparison across clinical and radiomic features

FIXED VERSION - NO DATA LEAKAGE
- Proper nested CV for feature selection
- Scaling inside CV folds
- Feature selection on training data only

Author: Parvez
Date: 2025-12-21
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

# Check GPU availability
print("\n" + "="*70)
print("GPU CHECK")
print("="*70)
try:
    import subprocess
    gpu_info = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader']
    )
    print(f"GPU Available: {gpu_info.decode().strip()}")
    print("XGBoost will use GPU acceleration")
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
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    confusion_matrix, precision_score, recall_score, f1_score,
    classification_report, balanced_accuracy_score
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
CLINICAL_DATA_PATH = "/hpc/pahm409/harvard/UOA/IMPRESS_RETROSPECTIVE_CLINICAL_DATA.xlsx"
OUTPUT_DIR = "./radiomic_fac_analysis_FIXED/"

LOSS_FUNCTIONS = ['AdaptiveRegional', 'GDice', 'BCEDice', 'BCEFocalTversky', 'Dice', 'Focal', 'FocalTversky', 'Tversky']
CLINICAL_FEATURES = ['AGE', 'NIHSS', 'GENDER', 'HEMI', 'HTN', 'DIABETES', 'AF', 'TPA', 'FAC_BASE']

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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_output_directory():
    """Create output directory structure"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/radiomics_features", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/predictions", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
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
        "DecisionTree": DecisionTreeClassifier(
            class_weight='balanced', random_state=42
        ),
        "GaussianNB": GaussianNB(),
        "SVM_RBF": SVC(
            kernel='rbf', probability=True, class_weight='balanced', random_state=42
        ),
        "SVM_Linear": SVC(
            kernel='linear', probability=True, class_weight='balanced', random_state=42
        ),
        "kNN_k5": KNeighborsClassifier(n_neighbors=5),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=500, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(**XGBOOST_PARAMS),
    }
    return models


def evaluate_classifier_no_leakage(model, X, y, model_name, verbose=True):
    """
    FIXED: Evaluate classifier with proper CV - NO DATA LEAKAGE
    - Scaling happens inside each CV fold
    - No pre-scaling on entire dataset
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"TRAINING MODEL: {model_name}")
        print(f"{'='*70}")
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

    # Ensure X is numeric
    if X.dtype == 'object' or not np.issubdtype(X.dtype, np.number):
        if verbose:
            print("⚠ Converting X to numeric...")
        try:
            X = X.astype(float)
        except (ValueError, TypeError):
            X = np.array([[float(val) if val is not None else np.nan for val in row] for row in X])
    
    if np.any(np.isnan(X)):
        nan_count = np.sum(np.any(np.isnan(X), axis=1))
        print(f"✗ ERROR: Found {nan_count} rows with NaN")
        return None

    # Setup CV
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    
    # Storage for predictions and fold AUCs
    y_pred_proba_all = np.zeros(len(y))
    auc_scores = []
    
    # FIXED: Manual CV loop with scaling INSIDE each fold
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # FIXED: Fit scaler ONLY on training data
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
    
    # Find optimal threshold (Youden J) on FULL predictions
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
    
    # Per-class metrics
    cls_report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    cls0 = cls_report.get('0', {})
    cls1 = cls_report.get('1', {})
    
    if verbose:
        print(f"\n✓ AUC = {auc_mean:.3f} ± {auc_std:.3f}")
        print(f"✓ Optimal threshold (Youden J) = {optimal_threshold:.3f}")
        print("\n✓ Classification Metrics:")
        print(f"  Sensitivity (FAC≥4): {sensitivity:.3f}")
        print(f"  Specificity (FAC<4): {specificity:.3f}")
        print(f"  Precision:           {precision:.3f}")
        print(f"  F1-score:            {f1:.3f}")
        print(f"  Balanced Accuracy:   {bal_acc:.3f}")
    
    return {
        'model_name': model_name,
        'auc_mean': auc_mean,
        'auc_std': auc_std,
        'auc_scores': auc_scores,
        'optimal_threshold': optimal_threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'balanced_acc': bal_acc,
        'class0_precision': cls0.get('precision', 0),
        'class0_recall': cls0.get('recall', 0),
        'class0_f1': cls0.get('f1-score', 0),
        'class1_precision': cls1.get('precision', 0),
        'class1_recall': cls1.get('recall', 0),
        'class1_f1': cls1.get('f1-score', 0),
        'y_true': y,
        'y_pred_proba': y_pred_proba_all,
    }


def evaluate_classifier_with_nested_feature_selection(
    model, X_all_features, y, feature_names, k_features, model_name, 
    selection_method='mutual_info', verbose=True
):
    """
    FIXED: Nested CV with feature selection INSIDE outer loop
    
    Outer loop: Performance estimation
    Inner loop (implicit): Feature selection on training data only
    
    Args:
        model: Sklearn-compatible classifier
        X_all_features: Full feature matrix (n_samples, n_features)
        y: Labels
        feature_names: List of feature names
        k_features: Number of features to select, or 'beat_clinical' for adaptive selection
        model_name: Name for logging
        selection_method: 'mutual_info' or 'beat_clinical'
        verbose: Print progress
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"NESTED CV: {model_name}")
        print(f"Feature selection: {k_features if isinstance(k_features, str) else f'Top-{k_features}'}")
        print(f"{'='*70}")
        print(f"Total features: {X_all_features.shape[1]}, Samples: {X_all_features.shape[0]}")
    
    # Ensure numeric
    if X_all_features.dtype == 'object' or not np.issubdtype(X_all_features.dtype, np.number):
        X_all_features = X_all_features.astype(float)
    
    if np.any(np.isnan(X_all_features)):
        print(f"✗ ERROR: NaN values in features")
        return None
    
    # Setup CV
    kf_outer = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    
    y_pred_proba_all = np.zeros(len(y))
    auc_scores = []
    selected_features_per_fold = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf_outer.split(X_all_features)):
        X_train, X_test = X_all_features[train_idx], X_all_features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # STEP 1: Feature selection ONLY on training data
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        
        if k_features == 'beat_clinical':
            # For 'beat_clinical', we need clinical MI threshold
            # This should be passed as a parameter, but for now we'll use top 20
            k_eff = 20
            print(f"  Note: 'beat_clinical' mode using top-{k_eff} (clinical threshold not passed)")
        elif isinstance(k_features, str):
            k_eff = 20  # Default
        else:
            k_eff = min(k_features, len(mi_scores))
        
        # Select top-k features based on MI
        top_k_idx = np.argsort(mi_scores)[-k_eff:]
        selected_features_per_fold.append([feature_names[i] for i in top_k_idx])
        
        # STEP 2: Use only selected features
        X_train_selected = X_train[:, top_k_idx]
        X_test_selected = X_test[:, top_k_idx]
        
        # STEP 3: Scale ONLY on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # STEP 4: Train and predict
        try:
            model_fold = model.__class__(**model.get_params())
            model_fold.fit(X_train_scaled, y_train)
            
            if hasattr(model_fold, "predict_proba"):
                proba = model_fold.predict_proba(X_test_scaled)[:, 1]
            else:
                df = model_fold.decision_function(X_test_scaled)
                proba = 1.0 / (1.0 + np.exp(-df))
            
            y_pred_proba_all[test_idx] = proba
            fold_auc = roc_auc_score(y_test, proba)
            auc_scores.append(fold_auc)
            
            if verbose:
                print(f"  Fold {fold_idx+1}: AUC = {fold_auc:.3f}, Features selected = {len(top_k_idx)}")
        
        except Exception as e:
            print(f"  ✗ Fold {fold_idx+1} failed: {e}")
            continue
    
    if len(auc_scores) == 0:
        print("✗ ERROR: No successful folds")
        return None
    
    auc_mean = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    
    # Compute optimal threshold and metrics
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
    
    cls_report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    cls0 = cls_report.get('0', {})
    cls1 = cls_report.get('1', {})
    
    # Feature stability analysis
    all_selected = []
    for feat_list in selected_features_per_fold:
        all_selected.extend(feat_list)
    feature_counts = pd.Series(all_selected).value_counts()
    
    if verbose:
        print(f"\n✓ AUC = {auc_mean:.3f} ± {auc_std:.3f}")
        print(f"✓ Feature stability: {len(feature_counts)} unique features selected across folds")
        print(f"  Most stable (selected in all 5 folds): {(feature_counts == 5).sum()} features")
        print(f"  Top 5 most stable features:")
        for feat, count in feature_counts.head(5).items():
            print(f"    {feat}: {count}/5 folds")
    
    return {
        'model_name': model_name,
        'auc_mean': auc_mean,
        'auc_std': auc_std,
        'auc_scores': auc_scores,
        'optimal_threshold': optimal_threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'balanced_acc': bal_acc,
        'class0_precision': cls0.get('precision', 0),
        'class0_recall': cls0.get('recall', 0),
        'class0_f1': cls0.get('f1-score', 0),
        'class1_precision': cls1.get('precision', 0),
        'class1_recall': cls1.get('recall', 0),
        'class1_f1': cls1.get('f1-score', 0),
        'y_true': y,
        'y_pred_proba': y_pred_proba_all,
        'selected_features_per_fold': selected_features_per_fold,
        'feature_stability': feature_counts,
    }


def run_all_classifiers(X, y, scenario_name, verbose=True):
    """Run all classifiers - FIXED version with no leakage"""
    models = get_base_models()
    all_results = {}
    
    for name, model in models.items():
        full_name = f"{scenario_name}__{name}"
        res = evaluate_classifier_no_leakage(model, X, y, full_name, verbose=verbose)
        if res is not None:
            all_results[full_name] = res
    
    return all_results


def run_all_classifiers_with_feature_selection(
    X_all, y, feature_names, scenario_name, k_features, verbose=True
):
    """Run all classifiers with nested feature selection - FIXED"""
    models = get_base_models()
    all_results = {}
    
    for name, model in models.items():
        full_name = f"{scenario_name}__{name}"
        res = evaluate_classifier_with_nested_feature_selection(
            model, X_all, y, feature_names, k_features, full_name, verbose=verbose
        )
        if res is not None:
            all_results[full_name] = res
    
    return all_results


def build_comparison_table(results_dict, output_csv_path=None):
    """Build comparison table from all results"""
    rows = []
    for key, res in results_dict.items():
        rows.append({
            'Scenario_Model': key,
            'AUC_mean': res['auc_mean'],
            'AUC_std': res['auc_std'],
            'Sensitivity': res['sensitivity'],
            'Specificity': res['specificity'],
            'F1': res['f1'],
            'BalancedAcc': res['balanced_acc'],
            'Class0_F1': res['class0_f1'],
            'Class1_F1': res['class1_f1'],
        })
    
    df_cmp = pd.DataFrame(rows).sort_values('AUC_mean', ascending=False)
    
    print("\n" + "="*70)
    print("GLOBAL MODEL RANKING (by AUC)")
    print("="*70)
    print(df_cmp[['Scenario_Model','AUC_mean','AUC_std','Sensitivity','Specificity','F1','BalancedAcc']].head(20).to_string(index=False))

    if output_csv_path is not None:
        df_cmp.to_csv(output_csv_path, index=False)
        print(f"\n✓ Saved to {output_csv_path}")
    
    return df_cmp


def plot_auc_ranking(df_cmp, output_path, top_n=20):
    """Plot AUC ranking"""
    plt.figure(figsize=(12, 8))
    df_sorted = df_cmp.head(top_n).sort_values('AUC_mean', ascending=True)
    
    plt.barh(range(len(df_sorted)), df_sorted['AUC_mean'])
    plt.yticks(range(len(df_sorted)), df_sorted['Scenario_Model'], fontsize=8)
    plt.xlabel('Mean AUC', fontsize=12)
    plt.title(f'Top {top_n} Classifier Rankings (FIXED - No Data Leakage)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Ranking plot saved to {output_path}")
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
    
    # Convert FAC12, filter invalid
    df['FAC12'] = pd.to_numeric(df['FAC12'], errors='coerce')
    n_before = len(df)
    df = df[df['FAC12'].notna()].copy()
    n_after = len(df)
    print(f"✓ Filtered: {n_before} → {n_after} patients")
    
    # Binary outcome: FAC≥4 = 1, FAC<4 = 0
    df['FAC12_binary'] = (df['FAC12'] >= 4).astype(int)
    
    # Preprocess features
    print("\nPreprocessing clinical features:")
    
    # GENDER (M=1, F=0)
    if 'GENDER' in df.columns:
        df['GENDER'] = df['GENDER'].fillna('Unknown')
        df['GENDER'] = (df['GENDER'].astype(str).str.upper().str.strip() == 'M').astype(int)
        print(f"  ✓ GENDER: Male={df['GENDER'].sum()}, Female={(1-df['GENDER']).sum()}")
    else:
        df['GENDER'] = 0
    
    # HEMI (R=1, L=0)
    if 'HEMI' in df.columns:
        df['HEMI'] = df['HEMI'].fillna('L')
        df['HEMI'] = (df['HEMI'].astype(str).str.upper().str.strip() == 'R').astype(int)
        print(f"  ✓ HEMI: Right={df['HEMI'].sum()}, Left={(1-df['HEMI']).sum()}")
    else:
        df['HEMI'] = 0
    
    # Binary features (Y=1, N=0)
    for feat in ['HTN', 'DIABETES', 'AF', 'TPA']:
        if feat in df.columns:
            df[feat] = df[feat].fillna('N')
            df[feat] = (df[feat].astype(str).str.upper().str.strip() == 'Y').astype(int)
            print(f"  ✓ {feat}: {df[feat].sum()} positive ({df[feat].mean()*100:.1f}%)")
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
    
    print(f"\nOutcome: FAC≥4={df['FAC12_binary'].sum()} ({df['FAC12_binary'].mean()*100:.1f}%), FAC<4={len(df)-df['FAC12_binary'].sum()}")
    
    return df


# ============================================================================
# RADIOMIC EXTRACTION (unchanged - no leakage here)
# ============================================================================

def get_ensemble_mask(patient_id, loss_function):
    """Get ensemble mask across replications"""
    fold = None
    for fold_idx in range(1, 6):
        mask_path = f"{BASE_DIR}/fold_{fold_idx}/{loss_function}_rep1/case_{patient_id}/reconstructed_prediction.nii.gz"
        if os.path.exists(mask_path):
            fold = fold_idx
            break
    
    if fold is None:
        return None
    
    masks = []
    mask_ref = None
    for rep in range(1, 4):
        mask_path = f"{BASE_DIR}/fold_{fold}/{loss_function}_rep{rep}/case_{patient_id}/reconstructed_prediction.nii.gz"
        if os.path.exists(mask_path):
            mask = sitk.ReadImage(mask_path)
            mask_ref = mask
            masks.append(sitk.GetArrayFromImage(mask))
    
    if len(masks) == 0:
        return None
    
    ensemble_array = np.mean(masks, axis=0)
    ensemble_binary = (ensemble_array > 0.5).astype(np.uint8)
    
    ensemble_mask = sitk.GetImageFromArray(ensemble_binary)
    ensemble_mask.CopyInformation(mask_ref)
    
    return ensemble_mask


def extract_radiomics_for_patient(patient_id, t1_image, mask, loss_function):
    """Extract radiomic features for a single patient"""
    
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
    
    # Extract features
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings['geometryTolerance'] = 1e-3
    
    extractor.enableImageTypeByName('Original')
    extractor.enableFeatureClassByName('shape')
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('gldm')
    extractor.enableFeatureClassByName('ngtdm')
    
    try:
        features = extractor.execute(t1_image, mask)
        
        radiomic_features = {
            'patient_id': patient_id,
            'loss_function': loss_function
        }
        
        # Only include numeric 'original_*' features
        for key, val in features.items():
            if not key.startswith('original_'):
                continue
            
            try:
                val_float = float(val)
                if np.isnan(val_float) or np.isinf(val_float):
                    continue
                clean_name = key.replace('original_', '').replace(' ', '_').lower()
                radiomic_features[clean_name] = val_float
            except (TypeError, ValueError):
                continue
        
        return radiomic_features
    
    except Exception as e:
        print(f"  Error: {e}")
        return None


def extract_radiomics_all_patients(clinical_df, loss_function):
    """Extract radiomics for all patients"""
    
    print(f"\n{'='*70}")
    print(f"EXTRACTING RADIOMICS: {loss_function}")
    print(f"{'='*70}")
    
    radiomic_data = []
    patient_ids = clinical_df['patient_id'].unique()
    
    for idx, patient_id in enumerate(patient_ids, 1):
        print(f"[{idx}/{len(patient_ids)}] {patient_id}...", end=' ')
        
        t1_path = f"{T1_IMAGE_DIR}/{patient_id}_T1_FNIRT_MNI.nii.gz"
        if not os.path.exists(t1_path):
            print("✗ No T1")
            continue
        
        t1_image = sitk.ReadImage(t1_path)
        mask = get_ensemble_mask(patient_id, loss_function)
        
        if mask is None:
            print("✗ No mask")
            continue
        
        features = extract_radiomics_for_patient(patient_id, t1_image, mask, loss_function)
        if features is not None:
            radiomic_data.append(features)
            print("✓")
        else:
            print("✗ Failed")
    
    if len(radiomic_data) == 0:
        return None
    
    radiomic_df = pd.DataFrame(radiomic_data)
    print(f"\n✓ Extracted {len(radiomic_df)} patients, {len([c for c in radiomic_df.columns if c not in ['patient_id','loss_function']])} features")
    
    output_path = f"{OUTPUT_DIR}/radiomics_features/{loss_function}_radiomics.csv"
    radiomic_df.to_csv(output_path, index=False)
    
    return radiomic_df


# ============================================================================
# MAIN PIPELINE - FIXED VERSION
# ============================================================================

def main():
    """Full multi-model pipeline - FIXED for no data leakage"""
    
    print("\n" + "="*70)
    print("MULTI-CLASSIFIER ANALYSIS: CLINICAL + RADIOMICS")
    print("FIXED VERSION - NO DATA LEAKAGE")
    print("="*70)
    
    setup_output_directory()
    
    # Load data
    clinical_df = load_clinical_data()
    
    if len(clinical_df) < 30:
        print("❌ ERROR: Too few patients")
        return
    
    y = clinical_df['FAC12_binary'].values
    all_results = {}
    
    # ========================================================================
    # Clinical-only models - FIXED (scaling inside CV)
    # ========================================================================
    print("\n" + "="*70)
    print("RUNNING: CLINICAL ONLY (FIXED - No Leakage)")
    print("="*70)
    
    X_clinical = clinical_df[CLINICAL_FEATURES].values.astype(float)
    clinical_results = run_all_classifiers(X_clinical, y, 'ClinicalOnly')
    all_results.update(clinical_results)
    
    # ========================================================================
    # Extract radiomics (unchanged - no leakage here)
    # ========================================================================
    radiomic_dfs = {}
    for loss_func in LOSS_FUNCTIONS:
        rad_df = extract_radiomics_all_patients(clinical_df, loss_func)
        if rad_df is not None:
            radiomic_dfs[loss_func] = rad_df
    
    if len(radiomic_dfs) == 0:
        print("❌ No radiomics extracted")
    else:
        # ====================================================================
        # Radiomics models - FIXED with nested feature selection
        # ====================================================================
        for loss_func, rad_df in radiomic_dfs.items():
            print(f"\n{'='*70}")
            print(f"PROCESSING: {loss_func} (FIXED - Nested CV)")
            print(f"{'='*70}")
            
            # Merge
            radiomic_cols_from_file = [c for c in rad_df.columns if c not in ['patient_id', 'loss_function']]
            merged = clinical_df.merge(rad_df, on='patient_id', how='inner')
            
            print(f"After merge: {len(merged)} patients")
            
            if len(merged) < 30:
                print(f"  ❌ Too few patients after merge - skipping")
                continue
            
            radiomic_cols = [c for c in radiomic_cols_from_file if c in merged.columns]
            print(f"Using {len(radiomic_cols)} radiomic features")
            
            if len(radiomic_cols) == 0:
                print(f"  ⚠ No radiomic features - skipping")
                continue
            
            # Clean NaN
            merged_clean = merged.dropna(subset=radiomic_cols)
            print(f"Clean dataset: {len(merged_clean)} patients")
            
            if len(merged_clean) < 30:
                print(f"  ❌ Too few patients after cleaning - skipping")
                continue
            
            y_clean = merged_clean['FAC12_binary'].values.astype(int)
            
            # FIXED: Nested feature selection for different k values
            for k_setting in ['Top10', 'Top20', 'Top50']:
                print(f"\n{'='*70}")
                print(f"NESTED FEATURE SELECTION: {k_setting} from {loss_func}")
                print(f"{'='*70}")
                
                if k_setting == 'Top10':
                    k = 10
                elif k_setting == 'Top20':
                    k = 20
                elif k_setting == 'Top50':
                    k = 50
                
                # Radiomics-only with NESTED feature selection
                X_rad_all = merged_clean[radiomic_cols].values.astype(float)
                
                rad_results = run_all_classifiers_with_feature_selection(
                    X_rad_all, y_clean, radiomic_cols, 
                    f"RadOnly_{loss_func}_{k_setting}", k, verbose=False
                )
                all_results.update(rad_results)
                
                # Clinical + Radiomics with NESTED feature selection
                # We need to handle this carefully - select radiomics, then combine with clinical
                print(f"\nRunning Clinical+Radiomics with nested selection...")
                
                # For Clinical+Radiomics, we do feature selection on radiomics only,
                # then combine with all clinical features
                clin_rad_results = {}
                models = get_base_models()
                
                for model_name, model in models.items():
                    full_name = f"Clin+Rad_{loss_func}_{k_setting}__{model_name}"
                    
                    # Manual nested CV for Clinical+Radiomics
                    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
                    y_pred_proba_all = np.zeros(len(y_clean))
                    auc_scores = []
                    
                    X_clinical_clean = merged_clean[CLINICAL_FEATURES].values.astype(float)
                    
                    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_rad_all)):
                        # Split radiomics
                        X_rad_train, X_rad_test = X_rad_all[train_idx], X_rad_all[test_idx]
                        X_clin_train, X_clin_test = X_clinical_clean[train_idx], X_clinical_clean[test_idx]
                        y_train, y_test = y_clean[train_idx], y_clean[test_idx]
                        
                        # Select radiomics features on training only
                        mi_scores = mutual_info_classif(X_rad_train, y_train, random_state=42)
                        top_k_idx = np.argsort(mi_scores)[-k:]
                        
                        X_rad_train_selected = X_rad_train[:, top_k_idx]
                        X_rad_test_selected = X_rad_test[:, top_k_idx]
                        
                        # Combine clinical + selected radiomics
                        X_train_combined = np.hstack([X_clin_train, X_rad_train_selected])
                        X_test_combined = np.hstack([X_clin_test, X_rad_test_selected])
                        
                        # Scale
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train_combined)
                        X_test_scaled = scaler.transform(X_test_combined)
                        
                        # Train and predict
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
                            print(f"    Fold {fold_idx+1} failed: {e}")
                            continue
                    
                    if len(auc_scores) > 0:
                        # Compute metrics
                        auc_mean = np.mean(auc_scores)
                        auc_std = np.std(auc_scores)
                        
                        fpr, tpr, thresholds = roc_curve(y_clean, y_pred_proba_all)
                        j_scores = tpr - fpr
                        best_idx = np.argmax(j_scores)
                        optimal_threshold = thresholds[best_idx]
                        y_pred = (y_pred_proba_all >= optimal_threshold).astype(int)
                        
                        tn, fp, fn, tp = confusion_matrix(y_clean, y_pred).ravel()
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        precision = precision_score(y_clean, y_pred, zero_division=0)
                        f1 = f1_score(y_clean, y_pred, zero_division=0)
                        bal_acc = balanced_accuracy_score(y_clean, y_pred)
                        
                        cls_report = classification_report(y_clean, y_pred, output_dict=True, zero_division=0)
                        cls0 = cls_report.get('0', {})
                        cls1 = cls_report.get('1', {})
                        
                        clin_rad_results[full_name] = {
                            'model_name': full_name,
                            'auc_mean': auc_mean,
                            'auc_std': auc_std,
                            'auc_scores': auc_scores,
                            'optimal_threshold': optimal_threshold,
                            'sensitivity': sensitivity,
                            'specificity': specificity,
                            'precision': precision,
                            'f1': f1,
                            'balanced_acc': bal_acc,
                            'class0_precision': cls0.get('precision', 0),
                            'class0_recall': cls0.get('recall', 0),
                            'class0_f1': cls0.get('f1-score', 0),
                            'class1_precision': cls1.get('precision', 0),
                            'class1_recall': cls1.get('recall', 0),
                            'class1_f1': cls1.get('f1-score', 0),
                            'y_true': y_clean,
                            'y_pred_proba': y_pred_proba_all,
                        }
                
                all_results.update(clin_rad_results)
    
    # ========================================================================
    # Global ranking
    # ========================================================================
    print("\n" + "="*70)
    print("BUILDING GLOBAL COMPARISON")
    print("="*70)
    
    cmp_path = f"{OUTPUT_DIR}/model_comparison_all_FIXED.csv"
    df_cmp = build_comparison_table(all_results, cmp_path)
    
    ranking_plot = f"{OUTPUT_DIR}/figures/classifier_ranking_FIXED.png"
    plot_auc_ranking(df_cmp, ranking_plot)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE! (FIXED - NO DATA LEAKAGE)")
    print("="*70)
    print(f"Results: {cmp_path}")
    print(f"Plot: {ranking_plot}")
    print("\nNOTE: AUC values will be LOWER than the leaky version.")
    print("This is CORRECT - they represent true generalization performance.")


if __name__ == "__main__":
    main()
