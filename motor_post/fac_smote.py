#!/usr/bin/env python3
"""
COMPREHENSIVE LOSS FUNCTION VALIDATION FOR FAC PREDICTION
WITH SMOTE-ENN CLASS IMBALANCE HANDLING

PURPOSE: Definitively determine which loss function produces 
         the best clinically valid segmentations for FAC prediction

Author: Parvez
Date: 2025-12-21
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("COMPREHENSIVE LOSS FUNCTION VALIDATION")
print("="*80)

# Radiomics
from radiomics import featureextractor

# Machine Learning
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, f1_score, balanced_accuracy_score,
    classification_report
)

# Imbalanced learning
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = "/home/pahm409/pytorch_loss_comparison_results2/"
T1_IMAGE_DIR = "/hpc/pahm409/harvard/UOA/FNIRT/train/"
CLINICAL_DATA_PATH = "/hpc/pahm409/harvard/UOA/IMPRESS_RETROSPECTIVE_CLINICAL_DATA.xlsx"
OUTPUT_DIR = "./FINAL_LOSS_VALIDATION_FAC/"

# ALL LOSS FUNCTIONS TO COMPARE
LOSS_FUNCTIONS = [
    'AdaptiveRegional',
    'GDice', 
    'Dice',
    'Tversky',
    'BCEDice',
    'BCETversky',
    'FocalTversky',
    'BCEFocalTversky',
    'Focal'
]

CLINICAL_FEATURES = ['AGE', 'NIHSS', 'GENDER', 'HEMI', 'HTN', 'DIABETES', 'AF', 'TPA', 'FAC_BASE']

# Best performing models from previous analysis
BEST_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=500, solver='liblinear', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM_Linear": SVC(kernel='linear', probability=True, random_state=42),
    "XGBoost": xgb.XGBClassifier(
        max_depth=3, learning_rate=0.05, n_estimators=100,
        random_state=42, eval_metric='auc', use_label_encoder=False
    ),
}

CV_FOLDS = 5
CV_RANDOM_STATE = 42
N_TOP_RADIOMICS = 20  # Top radiomics to select

# ============================================================================
# SETUP
# ============================================================================

def setup_output_directory():
    """Create output directory structure"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/radiomics", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/results", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}\n")


# ============================================================================
# EVALUATION WITH SMOTE-ENN
# ============================================================================

def evaluate_with_smote(model, X, y, model_name, verbose=False):
    """
    Evaluate model with SMOTE-ENN balancing
    SMOTE applied INSIDE each CV fold (no leakage)
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    
    if np.any(np.isnan(X)):
        return None
    
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    y_pred_proba_all = np.zeros(len(y))
    auc_scores = []
    
    smote_success = 0
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Apply SMOTE-ENN on training only
        try:
            minority_count = np.bincount(y_train).min()
            k_smote = min(5, minority_count - 1)
            
            if k_smote >= 1:
                smote_enn = SMOTEENN(
                    sampling_strategy='auto',
                    smote=SMOTE(k_neighbors=k_smote, random_state=42),
                    enn=EditedNearestNeighbours(n_neighbors=3),
                    random_state=42
                )
                X_train_bal, y_train_bal = smote_enn.fit_resample(X_train, y_train)
                smote_success += 1
            else:
                X_train_bal, y_train_bal = X_train, y_train
        except:
            X_train_bal, y_train_bal = X_train, y_train
        
        # Scale after balancing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_bal)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and predict
        try:
            model_fold = model.__class__(**model.get_params())
            
            # Special handling for XGBoost scale_pos_weight
            if 'XGBoost' in model_name:
                n_neg = np.sum(y_train_bal == 0)
                n_pos = np.sum(y_train_bal == 1)
                if n_pos > 0:
                    model_fold.set_params(scale_pos_weight=n_neg/n_pos)
            
            model_fold.fit(X_train_scaled, y_train_bal)
            
            if hasattr(model_fold, "predict_proba"):
                proba = model_fold.predict_proba(X_test_scaled)[:, 1]
            else:
                df = model_fold.decision_function(X_test_scaled)
                proba = 1.0 / (1.0 + np.exp(-df))
            
            y_pred_proba_all[test_idx] = proba
            fold_auc = roc_auc_score(y_test, proba)
            auc_scores.append(fold_auc)
        
        except Exception as e:
            if verbose:
                print(f"  Fold {fold_idx+1} failed: {e}")
            continue
    
    if len(auc_scores) == 0:
        return None
    
    # Compute metrics
    auc_mean = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    
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
    
    return {
        'model_name': model_name,
        'auc_mean': auc_mean,
        'auc_std': auc_std,
        'auc_scores': auc_scores,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'balanced_acc': bal_acc,
        'smote_success_rate': smote_success / CV_FOLDS,
        'y_true': y,
        'y_pred_proba': y_pred_proba_all,
    }


# ============================================================================
# CLINICAL DATA LOADING
# ============================================================================

def load_clinical_data():
    """Load and preprocess clinical data"""
    print("="*80)
    print("LOADING CLINICAL DATA")
    print("="*80)
    
    df = pd.read_excel(CLINICAL_DATA_PATH)
    df = df.rename(columns={'IMPRESS code': 'patient_id'})
    
    df['FAC12'] = pd.to_numeric(df['FAC12'], errors='coerce')
    df = df[df['FAC12'].notna()].copy()
    df['FAC12_binary'] = (df['FAC12'] >= 4).astype(int)
    
    # Preprocess features
    if 'GENDER' in df.columns:
        df['GENDER'] = (df['GENDER'].astype(str).str.upper().str.strip() == 'M').astype(int)
    else:
        df['GENDER'] = 0
    
    if 'HEMI' in df.columns:
        df['HEMI'] = (df['HEMI'].astype(str).str.upper().str.strip() == 'R').astype(int)
    else:
        df['HEMI'] = 0
    
    for feat in ['HTN', 'DIABETES', 'AF', 'TPA']:
        if feat in df.columns:
            df[feat] = (df[feat].astype(str).str.upper().str.strip() == 'Y').astype(int)
        else:
            df[feat] = 0
    
    for feat in ['AGE', 'NIHSS', 'FAC_BASE']:
        if feat in df.columns:
            df[feat] = pd.to_numeric(df[feat], errors='coerce')
            df[feat] = df[feat].fillna(df[feat].median())
        else:
            df[feat] = 0
    
    n_dep = (df['FAC12_binary'] == 0).sum()
    n_ind = (df['FAC12_binary'] == 1).sum()
    
    print(f"Total patients: {len(df)}")
    print(f"FAC < 4 (dependent):   {n_dep} ({n_dep/len(df)*100:.1f}%)")
    print(f"FAC ≥ 4 (independent): {n_ind} ({n_ind/len(df)*100:.1f}%)")
    print(f"Imbalance ratio: {n_ind/n_dep:.2f}:1\n")
    
    return df


# ============================================================================
# RADIOMIC EXTRACTION
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
    """Extract radiomic features"""
    
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
    
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings['geometryTolerance'] = 1e-3
    
    extractor.enableImageTypeByName('Original')
    extractor.enableFeatureClassByName('shape')
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    
    try:
        features = extractor.execute(t1_image, mask)
        radiomic_features = {'patient_id': patient_id, 'loss_function': loss_function}
        
        for key, val in features.items():
            if not key.startswith('original_'):
                continue
            try:
                val_float = float(val)
                if np.isnan(val_float) or np.isinf(val_float):
                    continue
                clean_name = key.replace('original_', '').replace(' ', '_').lower()
                radiomic_features[clean_name] = val_float
            except:
                continue
        
        return radiomic_features
    except:
        return None


def extract_radiomics_all_patients(clinical_df, loss_function):
    """Extract radiomics for all patients"""
    
    print(f"Extracting radiomics: {loss_function}")
    
    radiomic_data = []
    patient_ids = clinical_df['patient_id'].unique()
    
    success = 0
    for patient_id in patient_ids:
        t1_path = f"{T1_IMAGE_DIR}/{patient_id}_T1_FNIRT_MNI.nii.gz"
        if not os.path.exists(t1_path):
            continue
        
        t1_image = sitk.ReadImage(t1_path)
        mask = get_ensemble_mask(patient_id, loss_function)
        
        if mask is None:
            continue
        
        features = extract_radiomics_for_patient(patient_id, t1_image, mask, loss_function)
        if features is not None:
            radiomic_data.append(features)
            success += 1
    
    if len(radiomic_data) == 0:
        return None
    
    radiomic_df = pd.DataFrame(radiomic_data)
    n_features = len([c for c in radiomic_df.columns if c not in ['patient_id', 'loss_function']])
    
    print(f"  ✓ {success}/{len(patient_ids)} patients, {n_features} features")
    
    # Save
    output_path = f"{OUTPUT_DIR}/radiomics/{loss_function}_radiomics.csv"
    radiomic_df.to_csv(output_path, index=False)
    
    return radiomic_df


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Comprehensive loss function validation"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE LOSS FUNCTION VALIDATION FOR FAC")
    print("="*80 + "\n")
    
    setup_output_directory()
    
    # Load clinical data
    clinical_df = load_clinical_data()
    y = clinical_df['FAC12_binary'].values
    
    # ========================================================================
    # Step 1: Clinical-only baseline
    # ========================================================================
    print("="*80)
    print("STEP 1: CLINICAL-ONLY BASELINE (with SMOTE)")
    print("="*80 + "\n")
    
    X_clinical = clinical_df[CLINICAL_FEATURES].values.astype(float)
    
    clinical_results = {}
    for model_name, model in BEST_MODELS.items():
        print(f"Testing {model_name}...", end=' ')
        res = evaluate_with_smote(model, X_clinical, y, f"Clinical__{model_name}")
        if res:
            clinical_results[model_name] = res
            print(f"AUC={res['auc_mean']:.3f}, BalAcc={res['balanced_acc']:.3f}")
        else:
            print("Failed")
    
    # Best clinical model
    best_clinical_model = max(clinical_results.items(), 
                             key=lambda x: x[1]['balanced_acc'])
    
    print(f"\n✓ Best clinical: {best_clinical_model[0]}")
    print(f"  AUC = {best_clinical_model[1]['auc_mean']:.3f} ± {best_clinical_model[1]['auc_std']:.3f}")
    print(f"  Balanced Acc = {best_clinical_model[1]['balanced_acc']:.3f}")
    print(f"  Sensitivity = {best_clinical_model[1]['sensitivity']:.3f}")
    print(f"  Specificity = {best_clinical_model[1]['specificity']:.3f}")
    
    # ========================================================================
    # Step 2: Extract radiomics for ALL loss functions
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: EXTRACTING RADIOMICS FOR ALL LOSS FUNCTIONS")
    print("="*80 + "\n")
    
    radiomic_dfs = {}
    for loss_func in LOSS_FUNCTIONS:
        rad_df = extract_radiomics_all_patients(clinical_df, loss_func)
        if rad_df is not None:
            radiomic_dfs[loss_func] = rad_df
    
    print(f"\n✓ Extracted radiomics for {len(radiomic_dfs)}/{len(LOSS_FUNCTIONS)} loss functions\n")
    
    # ========================================================================
    # Step 3: Evaluate each loss function
    # ========================================================================
    print("="*80)
    print("STEP 3: EVALUATING EACH LOSS FUNCTION (Clinical + Radiomics)")
    print("="*80 + "\n")
    
    loss_results = {}
    
    for loss_func, rad_df in radiomic_dfs.items():
        print(f"\n{'='*80}")
        print(f"LOSS FUNCTION: {loss_func}")
        print(f"{'='*80}")
        
        # Merge with clinical
        radiomic_cols = [c for c in rad_df.columns if c not in ['patient_id', 'loss_function']]
        merged = clinical_df.merge(rad_df, on='patient_id', how='inner')
        merged_clean = merged.dropna(subset=radiomic_cols)
        
        print(f"Patients: {len(merged_clean)}")
        print(f"Radiomics: {len(radiomic_cols)} features")
        
        if len(merged_clean) < 30:
            print("⚠ Too few patients - skipping")
            continue
        
        y_clean = merged_clean['FAC12_binary'].values.astype(int)
        
        # Select top radiomics by MI
        X_rad_all = merged_clean[radiomic_cols].values.astype(float)
        mi_scores = mutual_info_classif(X_rad_all, y_clean, random_state=42)
        top_idx = np.argsort(mi_scores)[-N_TOP_RADIOMICS:]
        top_cols = [radiomic_cols[i] for i in top_idx]
        
        print(f"Selected top {len(top_cols)} radiomics by mutual information")
        
        # Combined features
        X_combined = merged_clean[CLINICAL_FEATURES + top_cols].values.astype(float)
        
        # Test all models
        loss_model_results = {}
        for model_name, model in BEST_MODELS.items():
            print(f"  Testing {model_name}...", end=' ')
            res = evaluate_with_smote(model, X_combined, y_clean, 
                                    f"{loss_func}__{model_name}")
            if res:
                loss_model_results[model_name] = res
                print(f"AUC={res['auc_mean']:.3f}, BalAcc={res['balanced_acc']:.3f}")
            else:
                print("Failed")
        
        # Best model for this loss
        if len(loss_model_results) > 0:
            best_model = max(loss_model_results.items(), 
                           key=lambda x: x[1]['balanced_acc'])
            
            loss_results[loss_func] = {
                'best_model': best_model[0],
                'best_result': best_model[1],
                'all_models': loss_model_results
            }
            
            print(f"\n  ✓ Best for {loss_func}: {best_model[0]}")
            print(f"    AUC = {best_model[1]['auc_mean']:.3f} ± {best_model[1]['auc_std']:.3f}")
            print(f"    Balanced Acc = {best_model[1]['balanced_acc']:.3f}")
            print(f"    Sensitivity = {best_model[1]['sensitivity']:.3f}")
            print(f"    Specificity = {best_model[1]['specificity']:.3f}")
    
    # ========================================================================
    # Step 4: Final Rankings
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL RANKINGS: LOSS FUNCTION COMPARISON")
    print("="*80 + "\n")
    
    # Build ranking table
    ranking_data = []
    
    # Add clinical baseline
    ranking_data.append({
        'Loss_Function': 'Clinical-only',
        'Best_Model': best_clinical_model[0],
        'AUC_mean': best_clinical_model[1]['auc_mean'],
        'AUC_std': best_clinical_model[1]['auc_std'],
        'Balanced_Acc': best_clinical_model[1]['balanced_acc'],
        'Sensitivity': best_clinical_model[1]['sensitivity'],
        'Specificity': best_clinical_model[1]['specificity'],
        'F1': best_clinical_model[1]['f1']
    })
    
    # Add each loss function
    for loss_func, loss_data in loss_results.items():
        best_res = loss_data['best_result']
        ranking_data.append({
            'Loss_Function': loss_func,
            'Best_Model': loss_data['best_model'],
            'AUC_mean': best_res['auc_mean'],
            'AUC_std': best_res['auc_std'],
            'Balanced_Acc': best_res['balanced_acc'],
            'Sensitivity': best_res['sensitivity'],
            'Specificity': best_res['specificity'],
            'F1': best_res['f1']
        })
    
    # Create DataFrame and sort
    df_ranking = pd.DataFrame(ranking_data)
    df_ranking = df_ranking.sort_values('Balanced_Acc', ascending=False)
    df_ranking['Rank'] = range(1, len(df_ranking) + 1)
    
    # Reorder columns
    df_ranking = df_ranking[['Rank', 'Loss_Function', 'Best_Model', 'AUC_mean', 
                             'AUC_std', 'Balanced_Acc', 'Sensitivity', 
                             'Specificity', 'F1']]
    
    # Save
    ranking_path = f"{OUTPUT_DIR}/results/loss_function_rankings.csv"
    df_ranking.to_csv(ranking_path, index=False)
    
    # Print table
    print(df_ranking.to_string(index=False))
    
    # ========================================================================
    # Step 5: Statistical Comparison
    # ========================================================================
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON: Top Loss vs Clinical Baseline")
    print("="*80 + "\n")
    
    # Get top loss function
    top_loss = df_ranking[df_ranking['Rank'] == 1].iloc[0]
    
    if top_loss['Loss_Function'] != 'Clinical-only':
        top_loss_name = top_loss['Loss_Function']
        top_loss_data = loss_results[top_loss_name]['best_result']
        clinical_data = best_clinical_model[1]
        
        # Paired t-test on AUC scores
        t_stat, p_value = stats.ttest_rel(
            top_loss_data['auc_scores'],
            clinical_data['auc_scores']
        )
        
        # Effect size (Cohen's d)
        diff = np.array(top_loss_data['auc_scores']) - np.array(clinical_data['auc_scores'])
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        
        print(f"Comparison: {top_loss_name} vs Clinical-only")
        print(f"  {top_loss_name} AUC:  {top_loss_data['auc_mean']:.3f} ± {top_loss_data['auc_std']:.3f}")
        print(f"  Clinical AUC:         {clinical_data['auc_mean']:.3f} ± {clinical_data['auc_std']:.3f}")
        print(f"  Difference:           {top_loss_data['auc_mean'] - clinical_data['auc_mean']:+.3f}")
        print(f"  t-statistic:          {t_stat:.3f}")
        print(f"  p-value:              {p_value:.4f}")
        print(f"  Cohen's d:            {cohens_d:.3f}")
        
        if p_value < 0.05:
            print(f"\n  ✓ {top_loss_name} is SIGNIFICANTLY BETTER than clinical-only (p < 0.05)")
        else:
            print(f"\n  ○ {top_loss_name} is NOT significantly different from clinical-only (p ≥ 0.05)")
    
    # ========================================================================
    # Step 6: Visualization
    # ========================================================================
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Plot 1: Ranking bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Balanced Accuracy
    ax1 = axes[0]
    df_plot = df_ranking.sort_values('Balanced_Acc', ascending=True)
    colors = ['red' if x == 'Clinical-only' else 'green' if x == df_ranking.iloc[0]['Loss_Function'] 
              else 'gray' for x in df_plot['Loss_Function']]
    
    ax1.barh(range(len(df_plot)), df_plot['Balanced_Acc'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(df_plot)))
    ax1.set_yticklabels(df_plot['Loss_Function'], fontsize=10)
    ax1.set_xlabel('Balanced Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Loss Function Rankings (Balanced Accuracy)', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(x=0.8, color='blue', linestyle='--', alpha=0.5, label='0.8 threshold')
    ax1.legend()
    
    # AUC
    ax2 = axes[1]
    df_plot2 = df_ranking.sort_values('AUC_mean', ascending=True)
    colors2 = ['red' if x == 'Clinical-only' else 'green' if x == df_ranking.iloc[0]['Loss_Function'] 
               else 'gray' for x in df_plot2['Loss_Function']]
    
    ax2.barh(range(len(df_plot2)), df_plot2['AUC_mean'], color=colors2, alpha=0.7)
    ax2.set_yticks(range(len(df_plot2)))
    ax2.set_yticklabels(df_plot2['Loss_Function'], fontsize=10)
    ax2.set_xlabel('AUC (mean)', fontsize=12, fontweight='bold')
    ax2.set_title('Loss Function Rankings (AUC)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.axvline(x=0.8, color='blue', linestyle='--', alpha=0.5, label='0.8 threshold')
    ax2.legend()
    
    plt.tight_layout()
    plot_path = f"{OUTPUT_DIR}/figures/loss_function_rankings.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Rankings plot saved: {plot_path}")
    
    # Plot 2: Sensitivity vs Specificity
    plt.figure(figsize=(10, 8))
    
    for idx, row in df_ranking.iterrows():
        color = 'red' if row['Loss_Function'] == 'Clinical-only' else \
                'green' if row['Rank'] == 1 else 'gray'
        size = 200 if row['Rank'] == 1 else 100
        
        plt.scatter(row['Specificity'], row['Sensitivity'], 
                   s=size, alpha=0.7, color=color,
                   label=row['Loss_Function'])
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    plt.xlabel('Specificity', fontsize=12, fontweight='bold')
    plt.ylabel('Sensitivity', fontsize=12, fontweight='bold')
    plt.title('Sensitivity vs Specificity Trade-off', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    sens_spec_path = f"{OUTPUT_DIR}/figures/sensitivity_specificity.png"
    plt.savefig(sens_spec_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Sens/Spec plot saved: {sens_spec_path}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  Rankings: {ranking_path}")
    print(f"  Figures: {OUTPUT_DIR}/figures/")
    
    print("\n" + "="*80)
    print("FINAL ANSWER: WHICH LOSS FUNCTION IS BEST?")
    print("="*80)
    
    top3 = df_ranking.head(3)
    print(f"\nTop 3 Loss Functions:")
    for idx, row in top3.iterrows():
        print(f"\n{row['Rank']}. {row['Loss_Function']}")
        print(f"   Best Model: {row['Best_Model']}")
        print(f"   Balanced Acc: {row['Balanced_Acc']:.3f}")
        print(f"   AUC: {row['AUC_mean']:.3f} ± {row['AUC_std']:.3f}")
        print(f"   Sensitivity: {row['Sensitivity']:.3f}")
        print(f"   Specificity: {row['Specificity']:.3f}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
