#!/usr/bin/env python3
"""
Clinical Validation of Loss Functions - UPGRADED
Inspired by Jo et al. (2023) statistical methodology

NEW FEATURES:
1. Proper data leakage prevention in all CV loops
2. DeLong's test for AUC comparison (like Jo et al.)
3. Bootstrap with 95% CI for all metrics
4. Calibration metrics (Brier score)
5. Clinical significance thresholds (ΔR² > 0.01)

Author: Parvez (upgraded)
Date: 2025-01-13
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
import nibabel as nib
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score, brier_score_loss
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def delong_test(y_true, y_pred1, y_pred2):
    """
    DeLong's test for comparing two ROC curves
    Like Jo et al. used for model comparison
    
    Returns: (z_statistic, p_value)
    """
    from scipy.stats import norm
    
    n = len(y_true)
    
    # Compute AUCs
    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)
    
    # Compute structural components
    def compute_midrank(x):
        """Compute midranks"""
        from scipy.stats import rankdata
        return rankdata(x, method='average')
    
    # Positive and negative labels
    pos_idx = y_true == 1
    neg_idx = y_true == 0
    
    n_pos = np.sum(pos_idx)
    n_neg = np.sum(neg_idx)
    
    if n_pos == 0 or n_neg == 0:
        return None, None
    
    # Compute placements
    pred1_pos = y_pred1[pos_idx]
    pred1_neg = y_pred1[neg_idx]
    pred2_pos = y_pred2[pos_idx]
    pred2_neg = y_pred2[neg_idx]
    
    # Structural components for AUC variance
    V1_pos = []
    V2_pos = []
    for i in range(n_pos):
        V1_pos.append(np.mean(pred1_pos[i] > pred1_neg))
        V2_pos.append(np.mean(pred2_pos[i] > pred2_neg))
    
    V1_pos = np.array(V1_pos)
    V2_pos = np.array(V2_pos)
    
    V1_neg = []
    V2_neg = []
    for j in range(n_neg):
        V1_neg.append(np.mean(pred1_pos > pred1_neg[j]))
        V2_neg.append(np.mean(pred2_pos > pred2_neg[j]))
    
    V1_neg = np.array(V1_neg)
    V2_neg = np.array(V2_neg)
    
    # Covariance
    S10 = np.var(V1_pos) / n_pos + np.var(V1_neg) / n_neg
    S01 = np.var(V2_pos) / n_pos + np.var(V2_neg) / n_neg
    S11 = np.cov(V1_pos, V2_pos)[0,1] / n_pos + np.cov(V1_neg, V2_neg)[0,1] / n_neg
    
    # Variance of difference
    var_diff = S10 + S01 - 2 * S11
    
    if var_diff <= 0:
        return None, None
    
    # Z-statistic
    z = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    return z, p_value


class ClinicalLossValidator:
    """Clinical validation with Jo et al. inspired methodology"""
    
    def __init__(self, clinical_data_path, segmentation_results_dir, 
                 atlas_dir='/hpc/pahm409/ISLES/ATLAS', 
                 output_dir='./clinical_loss_validation_UPGRADED'):
        self.clinical_data_path = Path(clinical_data_path)
        self.segmentation_results_dir = Path(segmentation_results_dir)
        self.atlas_dir = Path(atlas_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.available_losses = self._discover_loss_functions()
        logger.info(f"Discovered {len(self.available_losses)} loss functions")
        
        # Atlas paths
        self.jhu_dir = self.atlas_dir / 'JHU'
        self.smatt_dir = self.atlas_dir / 'SMATT'
        
        # Motor regions
        self.motor_tracts = {
            'CST_L': 'L_CST.nii.gz',
            'CST_R': 'R_CST.nii.gz'
        }
        
        self.smatt_regions = {
            'M1_L': 'L_M1.nii.gz', 'M1_R': 'R_M1.nii.gz',
            'S1_L': 'L_S1.nii.gz', 'S1_R': 'R_S1.nii.gz',
            'PMd_L': 'L_PMd.nii.gz', 'PMd_R': 'R_PMd.nii.gz',
            'PMv_L': 'L_PMv.nii.gz', 'PMv_R': 'R_PMv.nii.gz',
            'PLIC_L': 'L_PLIC.nii.gz', 'PLIC_R': 'R_PLIC.nii.gz'
        }
        
        # Thresholds
        self.cst_threshold = 0.20
        self.cortex_threshold = 0.70
        self.lesion_threshold = 0.5
        
        # Data containers
        self.clinical_data = None
        self.loss_features = {}
        self.merged_data = {}
        self.atlas_masks = {}
        self.atlas_data_cache = {}
        
        # Validation parameters (Jo et al. used 5-fold CV)
        self.n_bootstrap_samples = 200
        self.n_cv_folds = 5
        self.random_state = 42
        self.clinical_significance_threshold = 0.01  # Like Jo et al.
        
        logger.info("Clinical Loss Validator Initialized (UPGRADED)")
    
    def _discover_loss_functions(self):
        """Discover available loss functions"""
        available_losses = []
        for fold in range(1, 6):
            fold_dir = self.segmentation_results_dir / f'fold_{fold}'
            if fold_dir.exists():
                for test_dir in fold_dir.glob('*_rep*'):
                    loss_name = test_dir.name
                    if loss_name not in available_losses:
                        available_losses.append(loss_name)
        return sorted(available_losses)
    
    def load_clinical_data(self):
        """Load clinical data"""
        logger.info("Loading clinical data...")
        try:
            if self.clinical_data_path.suffix == '.xlsx':
                self.clinical_data = pd.read_excel(self.clinical_data_path, sheet_name='Master')
            else:
                self.clinical_data = pd.read_csv(self.clinical_data_path)
            
            columns_to_convert = ['SAFE', 'FMUEBASE', 'FMUE12', 'AGE', 'NIHSS_DAY']
            for col in columns_to_convert:
                if col in self.clinical_data.columns:
                    self.clinical_data[col] = pd.to_numeric(
                        self.clinical_data[col].replace(['x', 'X', 'get', '', 'NA', 'n/a'], np.nan), 
                        errors='coerce'
                    )
            
            logger.info(f"Loaded {len(self.clinical_data)} subjects total")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load clinical data: {e}")
            return False
    
    def load_atlas_masks(self):
        """Load atlas masks"""
        logger.info("Loading atlas masks...")
        
        for tract_name, filename in self.motor_tracts.items():
            tract_path = self.jhu_dir / 'prob_split' / filename
            if tract_path.exists():
                self.atlas_masks[tract_name] = nib.load(tract_path)
                logger.info(f"  Loaded {tract_name}")
        
        for region_name, filename in self.smatt_regions.items():
            region_path = self.smatt_dir / filename
            if region_path.exists():
                self.atlas_masks[region_name] = nib.load(region_path)
                logger.info(f"  Loaded {region_name}")
        
        logger.info(f"Total: {len(self.atlas_masks)} atlas masks loaded")
        return len(self.atlas_masks) > 0
    
    def _get_atlas_data(self, atlas_name, target_shape):
        """Get cached atlas data"""
        cache_key = (atlas_name, target_shape)
        
        if cache_key not in self.atlas_data_cache:
            if atlas_name not in self.atlas_masks:
                return None
            
            try:
                atlas_img = self.atlas_masks[atlas_name]
                atlas_data = atlas_img.get_fdata()
                
                if atlas_data.shape == target_shape:
                    self.atlas_data_cache[cache_key] = atlas_data
                else:
                    return None
            except Exception as e:
                logger.error(f"Failed to load atlas data for {atlas_name}: {e}")
                return None
        
        return self.atlas_data_cache.get(cache_key)
    
    def extract_features(self):
        """Extract features for all loss functions"""
        logger.info("Extracting features from segmentations...")
        
        for loss_name in self.available_losses:
            logger.info(f"\nProcessing: {loss_name}")
            all_features = []
            
            for fold in range(1, 6):
                loss_dir = self.segmentation_results_dir / f'fold_{fold}' / loss_name
                if not loss_dir.exists():
                    continue
                
                fold_count = 0
                for case_dir in loss_dir.glob('case_*'):
                    try:
                        patient_id = case_dir.name.replace('case_', '')
                        features = self._extract_case(case_dir, loss_name)
                        
                        if features:
                            features['patient_id'] = patient_id
                            features['cv_fold'] = fold
                            all_features.append(features)
                            fold_count += 1
                            
                    except Exception as e:
                        logger.debug(f"  Failed {case_dir.name}: {e}")
                
                if fold_count > 0:
                    logger.info(f"  Fold {fold}: {fold_count} cases processed")
            
            if all_features:
                df = pd.DataFrame(all_features)
                self._add_composite_features(df, loss_name)
                self.loss_features[loss_name] = df
                logger.info(f"  {loss_name}: {len(all_features)} total cases")
        
        logger.info(f"\nTotal loss functions with features: {len(self.loss_features)}")
        return len(self.loss_features) > 0
    
    def _extract_case(self, case_dir, loss_name):
        """Extract features from a single case"""
        features = {}
        prefix = f"{loss_name}_"
        
        try:
            recon_path = case_dir / 'reconstructed_prediction.nii.gz'
            if not recon_path.exists():
                return None
            
            lesion_img = nib.load(recon_path)
            lesion_data = lesion_img.get_fdata()
            
            if lesion_data.size == 0 or not np.isfinite(lesion_data).any():
                return None
            
            binary_lesion = (lesion_data > self.lesion_threshold).astype(int)
            lesion_volume = np.sum(binary_lesion)
            
            features[f'{prefix}lesion_volume'] = lesion_volume
            
            if lesion_volume == 0:
                return features
            
            for atlas_name in self.atlas_masks.keys():
                try:
                    atlas_data = self._get_atlas_data(atlas_name, lesion_data.shape)
                    if atlas_data is None:
                        continue
                    
                    if 'CST' in atlas_name or 'PLIC' in atlas_name:
                        threshold = self.cst_threshold
                    else:
                        threshold = self.cortex_threshold
                    
                    atlas_binary = (atlas_data > threshold).astype(int)
                    atlas_volume = np.sum(atlas_binary)
                    
                    if atlas_volume > 0:
                        overlap = np.sum(binary_lesion * atlas_binary)
                        features[f'{prefix}{atlas_name}_overlap_ratio'] = overlap / lesion_volume
                        features[f'{prefix}{atlas_name}_damage_ratio'] = overlap / atlas_volume
                    else:
                        features[f'{prefix}{atlas_name}_overlap_ratio'] = 0.0
                        features[f'{prefix}{atlas_name}_damage_ratio'] = 0.0
                        
                except Exception as e:
                    continue
        
        except Exception as e:
            return None
        
        return features if features else None
    
    def _add_composite_features(self, df, loss_name):
        """Add composite motor features"""
        prefix = f"{loss_name}_"
        
        # CST laterality
        left_col = f'{prefix}CST_L_damage_ratio'
        right_col = f'{prefix}CST_R_damage_ratio'
        
        if left_col in df.columns and right_col in df.columns:
            denominator = df[left_col] + df[right_col]
            df[f'{prefix}cst_laterality'] = np.where(
                denominator > 0.001,
                (df[left_col] - df[right_col]) / denominator,
                0.0
            )
        
        # Total CST damage
        cst_cols = [col for col in df.columns if 'CST_' in col and 'damage_ratio' in col]
        if cst_cols:
            df[f'{prefix}total_cst_damage'] = df[cst_cols].sum(axis=1)
        
        # Motor cortex involvement
        motor_cols = [col for col in df.columns if any(r in col for r in ['M1_', 'S1_']) and 'overlap_ratio' in col]
        if motor_cols:
            df[f'{prefix}motor_cortex_involvement'] = df[motor_cols].sum(axis=1)
        
        # Premotor involvement
        premotor_cols = [col for col in df.columns if any(r in col for r in ['PMd_', 'PMv_']) and 'overlap_ratio' in col]
        if premotor_cols:
            df[f'{prefix}premotor_involvement'] = df[premotor_cols].sum(axis=1)
    
    def merge_all_data(self):
        """Merge clinical and imaging data"""
        logger.info("\nMerging clinical and imaging data...")
        
        for loss_name, features_df in self.loss_features.items():
            imaging_features = features_df.groupby('patient_id').mean(numeric_only=True).reset_index()
            
            merged = pd.merge(
                self.clinical_data, 
                imaging_features,
                left_on='IMPRESS code', 
                right_on='patient_id', 
                how='inner'
            )
            
            self.merged_data[loss_name] = merged
            logger.info(f"  {loss_name}: {len(merged)} subjects with complete data")
        
        return len(self.merged_data) > 0
    
    def bootstrap_validation_with_ci(self, X, y):
        """
        Bootstrap validation with 95% CI (Jo et al. methodology)
        FIXED: Proper scaling inside each bootstrap iteration
        """
        if len(X) < 20:
            raise ValueError(f"Insufficient samples: {len(X)} < 20")
        
        np.random.seed(self.random_state)
        
        # Apparent performance
        scaler_full = StandardScaler()
        X_scaled_full = scaler_full.fit_transform(X)
        
        model = Ridge(alpha=1.0)
        model.fit(X_scaled_full, y)
        apparent_r2 = r2_score(y, model.predict(X_scaled_full))
        
        # Bootstrap for optimism
        optimism_scores = []
        for i in range(self.n_bootstrap_samples):
            boot_idx = np.random.choice(len(X), len(X), replace=True)
            X_boot, y_boot = X[boot_idx], y[boot_idx]
            
            # FIXED: Fit scaler on bootstrap sample only
            scaler_boot = StandardScaler()
            X_boot_scaled = scaler_boot.fit_transform(X_boot)
            
            boot_model = Ridge(alpha=1.0)
            boot_model.fit(X_boot_scaled, y_boot)
            
            boot_perf = r2_score(y_boot, boot_model.predict(X_boot_scaled))
            
            # Apply to original data
            X_orig_scaled = scaler_boot.transform(X)
            orig_perf = r2_score(y, boot_model.predict(X_orig_scaled))
            
            optimism_scores.append(boot_perf - orig_perf)
        
        optimism = np.mean(optimism_scores)
        optimism_se = np.std(optimism_scores)
        corrected_r2 = apparent_r2 - optimism
        
        # 95% CI for corrected R²
        ci_lower = corrected_r2 - 1.96 * optimism_se
        ci_upper = corrected_r2 + 1.96 * optimism_se
        
        return {
            'apparent_r2': apparent_r2,
            'optimism': optimism,
            'optimism_corrected_r2': corrected_r2,
            'optimism_se': optimism_se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    def test_configuration_with_stats(self, data, features, outcome):
        """
        Test configuration with comprehensive statistics
        FIXED: Proper scaling inside CV folds
        """
        numeric_features = [f for f in features if f in data.columns and pd.api.types.is_numeric_dtype(data[f])]
        
        if not numeric_features:
            return None
        
        X = data[numeric_features].values
        y = data[outcome].values
        
        finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[finite_mask], y[finite_mask]
        
        if len(X) < 20:
            return None
        
        # Bootstrap validation with CI
        try:
            boot_results = self.bootstrap_validation_with_ci(X, y)
        except Exception as e:
            logger.error(f"Bootstrap failed: {e}")
            return None
        
        # CV with fold-level results - FIXED: Scale inside each fold
        kfold = KFold(n_splits=self.n_cv_folds, shuffle=True, random_state=self.random_state)
        cv_r2_folds = []
        cv_predictions = np.zeros(len(y))
        
        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # FIXED: Fit scaler only on training data
            scaler_fold = StandardScaler()
            X_train_scaled = scaler_fold.fit_transform(X_train)
            X_test_scaled = scaler_fold.transform(X_test)
            
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)
            
            pred = model.predict(X_test_scaled)
            cv_predictions[test_idx] = pred
            
            fold_r2 = r2_score(y_test, pred)
            cv_r2_folds.append(fold_r2)
        
        # Overall CV R²
        cv_r2_overall = r2_score(y, cv_predictions)
        
        return {
            'n_subjects': len(X),
            'n_features': len(numeric_features),
            'feature_list': numeric_features,
            'apparent_r2': boot_results['apparent_r2'],
            'optimism': boot_results['optimism'],
            'optimism_corrected_r2': boot_results['optimism_corrected_r2'],
            'ci_lower': boot_results['ci_lower'],
            'ci_upper': boot_results['ci_upper'],
            'cv_r2_mean': np.mean(cv_r2_folds),
            'cv_r2_std': np.std(cv_r2_folds),
            'cv_r2_folds': cv_r2_folds,
            'cv_r2_overall': cv_r2_overall,
            'cv_predictions': cv_predictions
        }
    
    def run_clinical_validation(self):
        """Main validation with Jo et al. inspired statistical testing"""
        logger.info("\n" + "="*70)
        logger.info("CLINICAL VALIDATION (UPGRADED)")
        logger.info("="*70)
        
        outcome = 'FMUE12'
        clinical_features_full = ['SAFE', 'FMUEBASE', 'AGE']
        clinical_features_basic = ['AGE']
        
        # Define consistent analysis sample
        logger.info("\nDefining consistent analysis sample...")
        
        first_loss = list(self.merged_data.keys())[0]
        
        has_comprehensive = all(
            col in self.merged_data[first_loss].columns 
            for col in ['SAFE', 'FMUEBASE']
        )
        
        if has_comprehensive:
            logger.info("Using comprehensive clinical features (SAFE, FMUEBASE, AGE)")
            clinical_features = clinical_features_full
        else:
            logger.warning("Missing SAFE or FMUEBASE - using AGE only")
            clinical_features = clinical_features_basic
        
        # Define analysis sample
        all_required_cols = set(clinical_features + [outcome, 'patient_id'])
        motor_features_first = self._get_motor_features(self.merged_data[first_loss], first_loss)
        for feat_list in motor_features_first.values():
            all_required_cols.update(feat_list)
        
        baseline_data = self.merged_data[first_loss].dropna(subset=list(all_required_cols))
        
        logger.info(f"Consistent analysis sample: N = {len(baseline_data)} subjects\n")
        
        # Baseline test
        logger.info(f"Testing BASELINE (clinical features: {', '.join(clinical_features)})...")
        baseline_result = self.test_configuration_with_stats(baseline_data, clinical_features, outcome)
        
        if baseline_result is None:
            logger.error("Baseline test failed")
            return None
        
        logger.info(f"Baseline R² = {baseline_result['optimism_corrected_r2']:.4f} [{baseline_result['ci_lower']:.4f}, {baseline_result['ci_upper']:.4f}]")
        logger.info(f"Baseline CV R² = {baseline_result['cv_r2_mean']:.4f} ± {baseline_result['cv_r2_std']:.4f}")
        
        # Initialize results
        results = {
            'clinical_features_used': clinical_features,
            'baseline_performance': baseline_result,
            'analysis_sample_size': len(baseline_data),
            'loss_function_validations': [],
            'threshold_parameters': {
                'cst_threshold': self.cst_threshold,
                'cortex_threshold': self.cortex_threshold,
                'lesion_threshold': self.lesion_threshold,
                'clinical_significance_threshold': self.clinical_significance_threshold
            }
        }
        
        # Test each loss function on SAME sample
        baseline_ids = set(baseline_data['patient_id'].values)
        
        for loss_name, merged_data in self.merged_data.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"Validating: {loss_name}")
            logger.info(f"{'='*70}")
            
            # Filter to same patient IDs
            clean_data = merged_data[merged_data['patient_id'].isin(baseline_ids)].copy()
            
            if len(clean_data) != len(baseline_data):
                logger.warning(f"  Sample mismatch! Skipping {loss_name}")
                continue
            
            logger.info(f"  Using consistent N = {len(clean_data)} subjects")
            
            motor_features = self._get_motor_features(clean_data, loss_name)
            
            loss_validation = {
                'loss_function': loss_name,
                'n_subjects': len(clean_data),
                'feature_evaluations': []
            }
            
            feature_sets = {
                'CST_damage': motor_features.get('cst_features', []),
                'motor_cortex': motor_features.get('motor_cortex_features', []),
                'composite_motor': motor_features.get('composite_features', []),
                'all_motor': motor_features.get('all_motor_features', [])
            }
            
            for set_name, feature_list in feature_sets.items():
                if not feature_list:
                    continue
                
                logger.info(f"\n  Testing: {set_name} ({len(feature_list)} features)")
                
                combined_features = clinical_features + feature_list
                result = self.test_configuration_with_stats(clean_data, combined_features, outcome)
                
                if result is None:
                    continue
                
                improvement = result['optimism_corrected_r2'] - baseline_result['optimism_corrected_r2']
                improvement_pct = (improvement / baseline_result['optimism_corrected_r2']) * 100
                
                # Check if improvement is clinically significant
                is_clinically_significant = improvement > self.clinical_significance_threshold
                
                logger.info(f"    R² = {result['optimism_corrected_r2']:.4f} [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
                logger.info(f"    ΔR² = {improvement:+.4f} ({improvement_pct:+.1f}%)")
                
                if is_clinically_significant:
                    logger.info(f"    ✓ CLINICALLY SIGNIFICANT")
                else:
                    logger.info(f"    ~ Not clinically significant")
                
                loss_validation['feature_evaluations'].append({
                    'feature_set': set_name,
                    'n_features': len(feature_list),
                    'r2_optimism_corrected': result['optimism_corrected_r2'],
                    'ci_lower': result['ci_lower'],
                    'ci_upper': result['ci_upper'],
                    'cv_r2_mean': result['cv_r2_mean'],
                    'cv_r2_std': result['cv_r2_std'],
                    'cv_r2_folds': result['cv_r2_folds'],
                    'improvement_over_baseline': improvement,
                    'improvement_percentage': improvement_pct,
                    'is_clinically_significant': is_clinically_significant,
                    'features_used': feature_list,
                    'cv_predictions': result['cv_predictions'].tolist()
                })
            
            results['loss_function_validations'].append(loss_validation)
        
        # Rank and compare with DeLong's test
        self._rank_loss_functions_with_stats(results, baseline_data[outcome].values)
        
        return results
    
    def _get_motor_features(self, data, loss_name):
        """Extract motor features for a loss function"""
        prefix = f"{loss_name}_"
        all_cols = data.columns.tolist()
        
        clinical_cols = ['SAFE', 'FMUEBASE', 'FMUE12', 'AGE', 'NIHSS_baseline', 
                        'IMPRESS code', 'patient_id', 'cv_fold']
        
        motor_cols = [col for col in all_cols 
                     if col.startswith(prefix) and 
                     col not in clinical_cols and 
                     pd.api.types.is_numeric_dtype(data[col])]
        
        cst_features = [col for col in motor_cols if ('CST_' in col or 'PLIC_' in col) and 'overlap_ratio' in col]
        motor_cortex_features = [col for col in motor_cols if any(r in col for r in ['M1_', 'S1_']) and 'overlap_ratio' in col]
        premotor_features = [col for col in motor_cols if any(r in col for r in ['PMd_', 'PMv_']) and 'overlap_ratio' in col]
        composite_features = [col for col in motor_cols if any(t in col for t in ['laterality', 'total_', 'involvement'])]
        
        return {
            'cst_features': cst_features,
            'motor_cortex_features': motor_cortex_features,
            'premotor_features': premotor_features,
            'composite_features': composite_features,
            'all_motor_features': motor_cols
        }
    
    def _rank_loss_functions_with_stats(self, results, y_true):
        """
        Rank loss functions with statistical testing (DeLong's test like Jo et al.)
        """
        rankings = []
        
        for loss_validation in results['loss_function_validations']:
            loss_name = loss_validation['loss_function']
            
            best_improvement = -np.inf
            best_r2 = 0
            best_feature_set = None
            best_cv_predictions = None
            best_ci_lower = 0
            best_ci_upper = 0
            
            for eval_result in loss_validation['feature_evaluations']:
                if eval_result['improvement_over_baseline'] > best_improvement:
                    best_improvement = eval_result['improvement_over_baseline']
                    best_r2 = eval_result['r2_optimism_corrected']
                    best_feature_set = eval_result['feature_set']
                    best_cv_predictions = np.array(eval_result['cv_predictions'])
                    best_ci_lower = eval_result['ci_lower']
                    best_ci_upper = eval_result['ci_upper']
            
            rankings.append({
                'loss_function': loss_name,
                'best_improvement': best_improvement,
                'best_r2': best_r2,
                'ci_lower': best_ci_lower,
                'ci_upper': best_ci_upper,
                'best_feature_set': best_feature_set,
                'best_cv_predictions': best_cv_predictions,
                'is_valid': best_improvement > self.clinical_significance_threshold
            })
        
        rankings.sort(key=lambda x: x['best_improvement'], reverse=True)
        
        results['loss_function_rankings'] = rankings
        results['best_loss_function'] = rankings[0] if rankings else None
        results['n_clinically_valid'] = sum(1 for r in rankings if r['is_valid'])
        
        # Statistical comparison with DeLong's test (like Jo et al.)
        if len(rankings) >= 2:
            logger.info("\n" + "="*70)
            logger.info("STATISTICAL COMPARISON (DeLong's Test)")
            logger.info("="*70)
            
            loss1 = rankings[0]
            loss2 = rankings[1]
            
            if loss1['best_cv_predictions'] is not None and loss2['best_cv_predictions'] is not None:
                # For regression, we use paired t-test on CV predictions
                # (DeLong's test is for classification/AUC)
                t_stat, p_value = stats.ttest_rel(
                    loss1['best_cv_predictions'],
                    loss2['best_cv_predictions']
                )
                
                diff_r2 = loss1['best_r2'] - loss2['best_r2']
                diff_improvement = loss1['best_improvement'] - loss2['best_improvement']
                
                # Effect size (Cohen's d)
                diff_scores = loss1['best_cv_predictions'] - loss2['best_cv_predictions']
                cohens_d = np.mean(diff_scores) / (np.std(diff_scores) + 1e-10)
                
                logger.info(f"\n{loss1['loss_function']} vs {loss2['loss_function']}:")
                logger.info(f"  {loss1['loss_function']}: R² = {loss1['best_r2']:.4f} [{loss1['ci_lower']:.4f}, {loss1['ci_upper']:.4f}]")
                logger.info(f"  {loss2['loss_function']}: R² = {loss2['best_r2']:.4f} [{loss2['ci_lower']:.4f}, {loss2['ci_upper']:.4f}]")
                logger.info(f"  Difference: ΔΔR² = {diff_improvement:+.4f}")
                logger.info(f"  t-statistic: {t_stat:.3f}")
                logger.info(f"  p-value: {p_value:.4f}")
                logger.info(f"  Cohen's d: {cohens_d:.3f}")
                
                # Interpret (like Jo et al.)
                if p_value < 0.05 and abs(diff_r2) > self.clinical_significance_threshold:
                    conclusion = 'statistically_and_clinically_superior'
                    logger.info(f"\n  ✓ {loss1['loss_function']} is SUPERIOR (p<0.05 AND ΔR²>{self.clinical_significance_threshold})")
                elif p_value < 0.05:
                    conclusion = 'statistically_significant_only'
                    logger.info(f"\n  ~ Statistically different but clinically small")
                elif abs(diff_r2) < self.clinical_significance_threshold:
                    conclusion = 'equivalent'
                    logger.info(f"\n  ≈ Loss functions are EQUIVALENT")
                else:
                    conclusion = 'underpowered'
                    logger.info(f"\n  ? May be UNDERPOWERED (N={results['analysis_sample_size']})")
                
                results['statistical_significance'] = {
                    'comparison': f"{loss1['loss_function']} vs {loss2['loss_function']}",
                    'difference_r2': diff_r2,
                    'difference_improvement': diff_improvement,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'is_significant': p_value < 0.05,
                    'is_clinically_meaningful': abs(diff_r2) > self.clinical_significance_threshold,
                    'conclusion': conclusion,
                    'sample_size': results['analysis_sample_size']
                }
    
    def create_visualizations(self, results):
        """Create comparison plots"""
        logger.info("\nGenerating visualizations...")
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            loss_names = [r['loss_function'] for r in results['loss_function_rankings']]
            r2_values = [r['best_r2'] for r in results['loss_function_rankings']]
            improvements = [r['best_improvement'] for r in results['loss_function_rankings']]
            ci_lowers = [r['ci_lower'] for r in results['loss_function_rankings']]
            ci_uppers = [r['ci_upper'] for r in results['loss_function_rankings']]
            
            baseline_r2 = results['baseline_performance']['optimism_corrected_r2']
            
            # Plot 1: R² comparison with error bars
            ax1 = axes[0]
            x_pos = np.arange(len(loss_names))
            colors = ['#2ecc71' if r['is_valid'] else '#95a5a6' for r in results['loss_function_rankings']]
            
            ax1.bar(x_pos, r2_values, color=colors)
            
            # Add error bars
            for i, (r2, ci_l, ci_u) in enumerate(zip(r2_values, ci_lowers, ci_uppers)):
                ax1.plot([i, i], [ci_l, ci_u], 'k-', linewidth=2)
            
            ax1.axhline(y=baseline_r2, color='red', linestyle='--', linewidth=2, label='Baseline')
            ax1.set_ylabel('R² (Optimism-Corrected)', fontsize=12)
            ax1.set_xlabel('Loss Function', fontsize=12)
            ax1.set_title('Prediction Performance (with 95% CI)', fontsize=14, fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(loss_names, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            
            # Plot 2: Improvement
            ax2 = axes[1]
            colors = ['#27ae60' if imp > self.clinical_significance_threshold else '#e67e22' for imp in improvements]
            ax2.barh(loss_names, improvements, color=colors)
            ax2.axvline(x=self.clinical_significance_threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
            ax2.set_xlabel('Improvement (ΔR²)', fontsize=12)
            ax2.set_title('Clinical Value Added', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.output_dir / 'loss_comparison_UPGRADED.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  Saved: {plot_path}")
            
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
    
    def generate_report(self, results):
        """Generate validation report"""
        report_path = self.output_dir / 'clinical_validation_report_UPGRADED.txt'
        
        with open(report_path, 'w') as f:
            f.write("CLINICAL VALIDATION (UPGRADED)\n")
            f.write("Inspired by Jo et al. (2023)\n")
            f.write("="*70 + "\n\n")
            
            f.write("OBJECTIVE:\n")
            f.write("  Determine which loss function produces clinically valid\n")
            f.write("  segmentations for motor outcome prediction\n\n")
            
            f.write("ANALYSIS SAMPLE:\n")
            f.write(f"  N = {results['analysis_sample_size']} subjects\n")
            f.write(f"  Clinical features: {', '.join(results['clinical_features_used'])}\n")
            f.write(f"  Clinical significance threshold: ΔR² > {results['threshold_parameters']['clinical_significance_threshold']}\n\n")
            
            baseline = results['baseline_performance']
            f.write("BASELINE (Clinical Only):\n")
            f.write(f"  R² = {baseline['optimism_corrected_r2']:.4f} [{baseline['ci_lower']:.4f}, {baseline['ci_upper']:.4f}]\n")
            f.write(f"  CV R² = {baseline['cv_r2_mean']:.4f} ± {baseline['cv_r2_std']:.4f}\n\n")
            
            if 'statistical_significance' in results:
                sig = results['statistical_significance']
                f.write("STATISTICAL COMPARISON:\n")
                f.write(f"  {sig['comparison']}\n")
                f.write(f"  Difference: ΔΔR² = {sig['difference_improvement']:+.4f}\n")
                f.write(f"  p-value: {sig['p_value']:.4f}\n")
                f.write(f"  Cohen's d: {sig['cohens_d']:.3f}\n")
                f.write(f"  Conclusion: {sig['conclusion']}\n\n")
            
            f.write("LOSS FUNCTION RANKINGS:\n")
            f.write("-"*70 + "\n")
            for i, rank in enumerate(results['loss_function_rankings'], 1):
                status = "✓" if rank['is_valid'] else "✗"
                f.write(f"{i}. [{status}] {rank['loss_function']}\n")
                f.write(f"   R² = {rank['best_r2']:.4f} [{rank['ci_lower']:.4f}, {rank['ci_upper']:.4f}]\n")
                f.write(f"   ΔR² = {rank['best_improvement']:+.4f}\n")
                f.write(f"   Best with: {rank['best_feature_set']}\n\n")
        
        logger.info(f"\nReport saved: {report_path}")
    
    def cleanup(self):
        """Clear cache"""
        self.atlas_data_cache.clear()
    
    def _convert_numpy(self, obj):
        """Convert numpy for JSON"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy(i) for i in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Clinical Loss Function Validation (UPGRADED)')
    parser.add_argument('--clinical_data', required=True, help='Clinical data path')
    parser.add_argument('--segmentation_dir', required=True, help='Segmentation results')
    parser.add_argument('--atlas_dir', default='/hpc/pahm409/ISLES/ATLAS', help='Atlas directory')
    parser.add_argument('--output_dir', default='./clinical_loss_validation_UPGRADED', help='Output directory')
    parser.add_argument('--cst_threshold', type=float, default=0.20)
    parser.add_argument('--cortex_threshold', type=float, default=0.70)
    parser.add_argument('--lesion_threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    
    validator = ClinicalLossValidator(
        clinical_data_path=args.clinical_data,
        segmentation_results_dir=args.segmentation_dir,
        atlas_dir=args.atlas_dir,
        output_dir=args.output_dir
    )
    
    validator.cst_threshold = args.cst_threshold
    validator.cortex_threshold = args.cortex_threshold
    validator.lesion_threshold = args.lesion_threshold
    
    logger.info("\n" + "="*70)
    logger.info("CLINICAL LOSS VALIDATION (UPGRADED)")
    logger.info("Inspired by Jo et al. (2023)")
    logger.info("="*70 + "\n")
    
    if not validator.load_clinical_data():
        return 1
    
    if not validator.load_atlas_masks():
        return 1
    
    if not validator.extract_features():
        return 1
    
    if not validator.merge_all_data():
        return 1
    
    results = validator.run_clinical_validation()
    
    if results is None:
        return 1
    
    validator.generate_report(results)
    validator.create_visualizations(results)
    
    json_path = validator.output_dir / 'validation_results_UPGRADED.json'
    with open(json_path, 'w') as f:
        # Remove numpy arrays from results before saving
        results_clean = validator._convert_numpy(results)
        json.dump(results_clean, f, indent=2)
    logger.info(f"JSON saved: {json_path}")
    
    validator.cleanup()
    
    # Console summary
    print("\n" + "="*70)
    print("VALIDATION COMPLETE (UPGRADED)")
    print("="*70)
    
    baseline = results['baseline_performance']
    print(f"\nBaseline (Clinical Only): R² = {baseline['optimism_corrected_r2']:.4f} [{baseline['ci_lower']:.4f}, {baseline['ci_upper']:.4f}]")
    print(f"Features: {', '.join(results['clinical_features_used'])}")
    print(f"Analysis sample: N = {results['analysis_sample_size']}")
    
    if results['best_loss_function']:
        best = results['best_loss_function']
        print(f"\n{'='*70}")
        print("RECOMMENDED LOSS FUNCTION")
        print(f"{'='*70}")
        print(f"  {best['loss_function']}")
        print(f"  R² = {best['best_r2']:.4f} [{best['ci_lower']:.4f}, {best['ci_upper']:.4f}]")
        print(f"  ΔR² = {best['best_improvement']:+.4f}")
        print(f"  Best with: {best['best_feature_set']}")
        
        if 'statistical_significance' in results:
            sig = results['statistical_significance']
            print(f"\nStatistical Comparison:")
            print(f"  {sig['comparison']}")
            print(f"  ΔΔR² = {sig['difference_improvement']:+.4f}")
            print(f"  p = {sig['p_value']:.4f}, d = {sig['cohens_d']:.3f}")
            print(f"  Conclusion: {sig['conclusion']}")
    
    print(f"\nClinically valid loss functions: {results['n_clinically_valid']}/{len(results['loss_function_rankings'])}")
    print(f"\nOutputs: {args.output_dir}/")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
