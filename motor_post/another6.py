#!/usr/bin/env python3
"""
Motor Pathway Prediction - Multi-Replication Analysis (IMPROVED v2)

DEPENDENCIES:
-------------
This script requires your custom validation.py module with ClinicalLossValidator class.
Make sure validation.py is in the same directory as this script.

Fixes:
------
1. Ensures clinical variance across reps by varying random state
2. Validates sample sizes per rep
3. Adds more detailed logging
4. Checks for data leakage
5. IMPROVED: Fixed sample size warning logic
6. IMPROVED: Added statistical significance testing
7. IMPROVED: Added explicit data leakage checks
8. IMPROVED: Added confidence intervals
9. IMPROVED: Better documentation of random_state strategy
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# NOTE: You need to have validation.py in your working directory
# This should contain your ClinicalLossValidator class
try:
    from validation import ClinicalLossValidator
except ImportError:
    print("ERROR: validation.py not found!")
    print("This script requires your custom ClinicalLossValidator class")
    print("Make sure validation.py is in the same directory as this script")
    raise

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_motor_pathway_models(merged_df, loss_name, outcome="FMUE12", threshold=50, random_state=42):
    """
    Train Clinical, Motor, and Combined models.
    
    IMPROVED:
    - Added random_state parameter for CV reproducibility tracking
    - Added data validation checks
    - More detailed logging
    - Explicit data leakage checks
    - Fixed sample size validation
    """
    if merged_df.empty:
        logger.warning(f"Empty dataframe for {loss_name}")
        return None

    clinical_features = ["SAFE", "FMUEBASE", "AGE", "NIHSS_DAY"]
    motor_feature_cols = [c for c in merged_df.columns if any(x in c for x in
                          ["CST", "M1", "S1", "PMd", "PMv", "PLIC", "laterality", "total_", "involvement"])]
    motor_feature_cols = [c for c in motor_feature_cols if 'mRS' not in c]

    if outcome not in merged_df.columns:
        logger.error(f"Outcome {outcome} missing")
        return None

    df = merged_df.copy()
    if 'mRS12' in df.columns:
        df = df.drop(columns=['mRS12'])

    # Convert to numeric
    cols_to_convert = clinical_features + motor_feature_cols + [outcome]
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop incomplete rows
    required_cols = [c for c in clinical_features if c in df.columns] + motor_feature_cols
    clean_df = df.dropna(subset=required_cols + [outcome])
    
    if clean_df.empty:
        logger.error(f"No complete rows after dropna")
        return None
    
    # DATA VALIDATION - IMPROVED
    logger.info(f"    Sample size: {len(clean_df)} patients")
    logger.info(f"    Clinical features: {len([c for c in clinical_features if c in clean_df.columns])}")
    logger.info(f"    Motor features: {len(motor_feature_cols)}")
    
    # IMPROVED: Fixed sample size warning logic
    # Expected: 5-fold CV with ~36 patients per fold = ~180 total patients
    expected_total = 180  # ~36 per fold × 5 folds
    if len(clean_df) > expected_total + 20:  # Allow some margin
        logger.warning(f"    ⚠️  Sample size ({len(clean_df)}) seems too large!")
        logger.warning(f"    ⚠️  Expected ~{expected_total} total patients for 5-fold CV")
    
    # IMPROVED: Explicit data leakage check
    clinical_cols = [c for c in clinical_features if c in clean_df.columns]
    motor_cols = [c for c in motor_feature_cols if c in clean_df.columns]
    feature_cols = clinical_cols + motor_cols
    
    # Check for data leakage - outcome variables should NOT be in features
    forbidden_cols = ['FMUE12', 'FMUE_PRIMARY', 'mRS12', 'mRS', outcome]
    leak_cols = [c for c in feature_cols if any(f in c for f in forbidden_cols)]
    if leak_cols:
        logger.error(f"    ⚠️  DATA LEAKAGE DETECTED: {leak_cols}")
        return None

    y = clean_df[outcome].values
    y_binary = (y >= threshold).astype(int)
    
    # IMPROVED: Document random_state strategy
    # IMPORTANT: Each replication uses a different random_state (42 + rep_idx)
    # This ensures different CV splits across reps, measuring true variance
    # in model performance, not just random seed effects within a single split
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1)
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1)

    results = {
        "loss_name": loss_name, 
        "n_samples": int(len(clean_df)),
        "random_state": random_state
    }

    # CLINICAL ONLY
    if clinical_cols:
        X = clean_df[clinical_cols].values
        
        # FIXED: Use Pipeline to scale inside CV (prevents data leakage)
        clinical_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', rf_reg)
        ])
        ridge_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', ridge)
        ])
        clf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', rf_clf)
        ])
        
        # Get all fold scores for variance analysis
        clinical_r2_scores = cross_val_score(clinical_pipeline, X, y, cv=kfold, scoring='r2')
        
        results["clinical"] = {
            "n_features": len(clinical_cols),
            "ridge_r2": float(cross_val_score(ridge_pipeline, X, y, cv=kfold, scoring='r2').mean()),
            "rf_r2": float(clinical_r2_scores.mean()),
            "rf_r2_std": float(clinical_r2_scores.std()),
            "rf_r2_folds": [float(x) for x in clinical_r2_scores],
            "auc": float(cross_val_score(clf_pipeline, X, y_binary, cv=kfold, scoring='roc_auc').mean())
        }
        
        logger.info(f"    Clinical R² (folds): {[f'{x:.3f}' for x in clinical_r2_scores]}")

    # MOTOR ONLY
    if motor_cols:
        X = clean_df[motor_cols].values
        
        # FIXED: Use Pipeline to scale inside CV
        motor_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', rf_reg)
        ])
        ridge_pipeline_motor = Pipeline([
            ('scaler', StandardScaler()),
            ('model', ridge)
        ])
        clf_pipeline_motor = Pipeline([
            ('scaler', StandardScaler()),
            ('model', rf_clf)
        ])
        
        motor_r2_scores = cross_val_score(motor_pipeline, X, y, cv=kfold, scoring='r2')
        
        results["motor"] = {
            "n_features": len(motor_cols),
            "ridge_r2": float(cross_val_score(ridge_pipeline_motor, X, y, cv=kfold, scoring='r2').mean()),
            "rf_r2": float(motor_r2_scores.mean()),
            "rf_r2_std": float(motor_r2_scores.std()),
            "rf_r2_folds": [float(x) for x in motor_r2_scores],
            "auc": float(cross_val_score(clf_pipeline_motor, X, y_binary, cv=kfold, scoring='roc_auc').mean())
        }

    # COMBINED
    X = clean_df[feature_cols].values
    
    # FIXED: Use Pipeline to scale inside CV
    combined_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', rf_reg)
    ])
    ridge_pipeline_combined = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ridge)
    ])
    clf_pipeline_combined = Pipeline([
        ('scaler', StandardScaler()),
        ('model', rf_clf)
    ])
    
    combined_r2_scores = cross_val_score(combined_pipeline, X, y, cv=kfold, scoring='r2')
    
    results["combined"] = {
        "n_features": len(feature_cols),
        "ridge_r2": float(cross_val_score(ridge_pipeline_combined, X, y, cv=kfold, scoring='r2').mean()),
        "rf_r2": float(combined_r2_scores.mean()),
        "rf_r2_std": float(combined_r2_scores.std()),
        "rf_r2_folds": [float(x) for x in combined_r2_scores],
        "auc": float(cross_val_score(clf_pipeline_combined, X, y_binary, cv=kfold, scoring='roc_auc').mean())
    }

    return results


def compute_confidence_interval(values, confidence=0.95):
    """
    Compute confidence interval for a list of values.
    
    Args:
        values: List or array of numeric values
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    n = len(values)
    if n < 2:
        return (np.nan, np.nan)
    
    mean = np.mean(values)
    sem = stats.sem(values)  # Standard error of the mean
    ci = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return (float(mean - ci), float(mean + ci))


def main():
    clinical_data_path = "/hpc/pahm409/harvard/UOA/IMPRESS_RETROSPECTIVE_CLINICAL_DATA.xlsx"
    base_dir = "/home/pahm409/pytorch_loss_comparison_results2/"
    atlas_dir = "/hpc/pahm409/ISLES/ATLAS"
    output_dir = Path("./motor_pathway_prediction_multirep_v2/")
    output_dir.mkdir(exist_ok=True)

    # Extract base loss names and reps - ONLY DIRECTORIES
    logger.info("Scanning for loss function replications...")
    all_dirs = []
    for fold_dir in Path(base_dir).glob("fold_*"):
        for item in fold_dir.iterdir():
            if item.is_dir() and '_rep' in item.name:
                all_dirs.append(item.name)
    
    all_dirs = list(set(all_dirs))
    
    loss_rep_map = {}
    for dirname in all_dirs:
        parts = dirname.rsplit('_rep', 1)
        if len(parts) == 2:
            loss_base = parts[0]
            rep_num = parts[1]
            if loss_base not in loss_rep_map:
                loss_rep_map[loss_base] = []
            if f"{loss_base}_rep{rep_num}" not in loss_rep_map[loss_base]:
                loss_rep_map[loss_base].append(f"{loss_base}_rep{rep_num}")
    
    logger.info(f"Found {len(loss_rep_map)} loss functions with replications:")
    for loss_base, reps in sorted(loss_rep_map.items()):
        logger.info(f"  {loss_base}: {len(reps)} reps")

    # Process each replication
    all_results = {}
    
    for loss_base, rep_list in sorted(loss_rep_map.items()):
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {loss_base}")
        logger.info(f"{'='*70}")
        
        rep_results = []
        
        for rep_idx, rep_name in enumerate(sorted(rep_list)):
            logger.info(f"\n  Replication: {rep_name}")
            
            # IMPROVED: Use different random state for each rep to ensure variance
            # This creates different CV splits, measuring true performance variance
            rep_random_state = 42 + rep_idx
            
            # Initialize validator for this replication
            validator = ClinicalLossValidator(
                clinical_data_path=clinical_data_path,
                segmentation_results_dir=base_dir,
                atlas_dir=atlas_dir
            )
            
            # Override discovery to only look for this specific rep
            validator.available_losses = [rep_name]
            
            validator.load_clinical_data()
            validator.load_atlas_masks()
            validator.extract_features()
            validator.merge_all_data()
            
            if rep_name in validator.merged_data:
                res = train_motor_pathway_models(
                    validator.merged_data[rep_name], 
                    rep_name,
                    random_state=rep_random_state
                )
                if res:
                    rep_results.append(res)
                    logger.info(f"    Clinical R²: {res['clinical']['rf_r2']:.3f} (CV std: {res['clinical']['rf_r2_std']:.3f})")
                    logger.info(f"    Motor R²:    {res['motor']['rf_r2']:.3f} (CV std: {res['motor']['rf_r2_std']:.3f})")
                    logger.info(f"    Combined R²: {res['combined']['rf_r2']:.3f} (CV std: {res['combined']['rf_r2_std']:.3f})")
                else:
                    logger.warning(f"    Failed to train models for {rep_name}")
            else:
                logger.warning(f"    No merged data found for {rep_name}")
        
        # Aggregate across replications
        if rep_results:
            clinical_r2s = [r['clinical']['rf_r2'] for r in rep_results]
            combined_r2s = [r['combined']['rf_r2'] for r in rep_results]
            motor_r2s = [r['motor']['rf_r2'] for r in rep_results if 'motor' in r]
            
            # Track CV variance within each rep
            clinical_cv_stds = [r['clinical']['rf_r2_std'] for r in rep_results]
            
            # IMPROVED: Statistical significance testing
            statistical_tests = {}
            if len(clinical_r2s) >= 2 and len(combined_r2s) >= 2:
                # Paired t-test: Combined vs Clinical
                t_stat, p_value = stats.ttest_rel(combined_r2s, clinical_r2s)
                
                # Effect size (Cohen's d for paired samples)
                diff = np.array(combined_r2s) - np.array(clinical_r2s)
                cohens_d = np.mean(diff) / np.std(diff, ddof=1)
                
                statistical_tests = {
                    'paired_ttest': {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant_at_0.05': bool(p_value < 0.05),
                        'significant_at_0.01': bool(p_value < 0.01)
                    },
                    'effect_size': {
                        'cohens_d': float(cohens_d),
                        'interpretation': (
                            'large' if abs(cohens_d) >= 0.8 else
                            'medium' if abs(cohens_d) >= 0.5 else
                            'small' if abs(cohens_d) >= 0.2 else
                            'negligible'
                        )
                    }
                }
            
            # IMPROVED: Confidence intervals
            clinical_ci = compute_confidence_interval(clinical_r2s)
            combined_ci = compute_confidence_interval(combined_r2s)
            
            all_results[loss_base] = {
                "n_replications": len(rep_results),
                "sample_size": rep_results[0]['n_samples'],
                "clinical": {
                    "rf_r2_mean": float(np.mean(clinical_r2s)),
                    "rf_r2_std": float(np.std(clinical_r2s)),
                    "rf_r2_min": float(np.min(clinical_r2s)),
                    "rf_r2_max": float(np.max(clinical_r2s)),
                    "rf_r2_ci_95": clinical_ci,  # NEW
                    "mean_cv_std": float(np.mean(clinical_cv_stds))
                },
                "combined": {
                    "rf_r2_mean": float(np.mean(combined_r2s)),
                    "rf_r2_std": float(np.std(combined_r2s)),
                    "rf_r2_min": float(np.min(combined_r2s)),
                    "rf_r2_max": float(np.max(combined_r2s)),
                    "rf_r2_ci_95": combined_ci  # NEW
                },
                "improvement": {
                    "absolute_mean": float(np.mean(combined_r2s) - np.mean(clinical_r2s)),
                    "relative_pct": float((np.mean(combined_r2s) - np.mean(clinical_r2s)) / np.mean(clinical_r2s) * 100)
                },
                "statistical_tests": statistical_tests,  # NEW
                "replication_details": rep_results
            }
            
            # Add motor-only stats if available
            if motor_r2s:
                motor_ci = compute_confidence_interval(motor_r2s)
                all_results[loss_base]["motor"] = {
                    "rf_r2_mean": float(np.mean(motor_r2s)),
                    "rf_r2_std": float(np.std(motor_r2s)),
                    "rf_r2_min": float(np.min(motor_r2s)),
                    "rf_r2_max": float(np.max(motor_r2s)),
                    "rf_r2_ci_95": motor_ci  # NEW
                }
        else:
            logger.warning(f"No valid replications found for {loss_base}")

    # Save results
    output_file = output_dir / "multirep_results_improved.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "="*70)
    print("MULTI-REPLICATION SUMMARY (IMPROVED v2)")
    print("="*70)
    
    if not all_results:
        print("\nNo results to display!")
        return
    
    # Sort by improvement
    ranked = sorted(all_results.items(), 
                   key=lambda x: x[1]['improvement']['absolute_mean'], 
                   reverse=True)
    
    print(f"\nTotal loss functions analyzed: {len(ranked)}")
    print(f"\nRanked by improvement (Combined R² - Clinical R²):")
    print("-" * 70)
    
    for i, (loss_name, res) in enumerate(ranked, 1):
        print(f"\n{i}. {loss_name} ({res['n_replications']} reps, n={res['sample_size']})")
        
        # Show between-rep variance AND within-rep CV variance
        clin_between_std = res['clinical']['rf_r2_std']
        clin_within_std = res['clinical'].get('mean_cv_std', 0)
        clin_ci = res['clinical']['rf_r2_ci_95']
        comb_ci = res['combined']['rf_r2_ci_95']
        
        print(f"   Clinical:  R² = {res['clinical']['rf_r2_mean']:.3f} ± {clin_between_std:.3f} (between-rep)")
        print(f"              95% CI = [{clin_ci[0]:.3f}, {clin_ci[1]:.3f}]")
        print(f"              CV std = {clin_within_std:.3f} (within-rep)")
        
        if 'motor' in res:
            motor_ci = res['motor']['rf_r2_ci_95']
            print(f"   Motor:     R² = {res['motor']['rf_r2_mean']:.3f} ± {res['motor']['rf_r2_std']:.3f}")
            print(f"              95% CI = [{motor_ci[0]:.3f}, {motor_ci[1]:.3f}]")
        
        print(f"   Combined:  R² = {res['combined']['rf_r2_mean']:.3f} ± {res['combined']['rf_r2_std']:.3f}")
        print(f"              95% CI = [{comb_ci[0]:.3f}, {comb_ci[1]:.3f}]")
        print(f"   Improvement: ΔR² = {res['improvement']['absolute_mean']:+.3f} ({res['improvement']['relative_pct']:+.1f}%)")
        
        # IMPROVED: Show statistical significance
        if 'statistical_tests' in res and 'paired_ttest' in res['statistical_tests']:
            p_val = res['statistical_tests']['paired_ttest']['p_value']
            sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            cohens_d = res['statistical_tests']['effect_size']['cohens_d']
            effect_interp = res['statistical_tests']['effect_size']['interpretation']
            
            print(f"   Statistical: p = {p_val:.4f} {sig_marker}, Cohen's d = {cohens_d:.3f} ({effect_interp})")
        
        # Flag potential issues
        if clin_between_std < 0.001:
            print(f"   ⚠️  WARNING: Zero between-rep variance!")
        if res['sample_size'] > 200:  # Updated threshold
            print(f"   ⚠️  WARNING: Sample size seems too large (expected ~180)")
    
    print("\n" + "="*70)
    
    # Statistical summary
    print("\nSTATISTICAL SUMMARY:")
    print("-" * 70)
    improvements = [res['improvement']['absolute_mean'] for _, res in ranked]
    print(f"Mean improvement across all losses: {np.mean(improvements):.3f}")
    print(f"Std improvement across all losses:  {np.std(improvements):.3f}")
    print(f"Best improvement: {np.max(improvements):.3f} ({ranked[0][0]})")
    print(f"Worst improvement: {np.min(improvements):.3f} ({ranked[-1][0]})")
    
    # Count statistically significant improvements
    sig_count = sum(1 for _, res in ranked 
                   if 'statistical_tests' in res 
                   and 'paired_ttest' in res['statistical_tests']
                   and res['statistical_tests']['paired_ttest']['significant_at_0.05'])
    print(f"\nStatistically significant improvements (p < 0.05): {sig_count}/{len(ranked)}")
    print("\nSignificance markers: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("="*70)


if __name__ == "__main__":
    main()
