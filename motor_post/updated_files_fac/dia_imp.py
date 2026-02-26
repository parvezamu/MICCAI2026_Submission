#!/usr/bin/env python3
"""
Diagnostic: Why is there no improvement?
Analyze the actual data to find the root cause
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

print("="*70)
print("DIAGNOSTIC: Why No Improvement from Imaging?")
print("="*70)

# Load clinical data
print("\n1. Loading clinical data...")
df_clin = pd.read_excel("/hpc/pahm409/harvard/UOA/IMPRESS_RETROSPECTIVE_CLINICAL_DATA.xlsx")
df_clin = df_clin.rename(columns={'IMPRESS code': 'patient_id'})
df_clin['FAC12'] = pd.to_numeric(df_clin['FAC12'], errors='coerce')
df_clin = df_clin[df_clin['FAC12'].notna()].copy()
df_clin['FAC12_binary'] = (df_clin['FAC12'] >= 4).astype(int)

# Preprocess features
for feat in ['AGE', 'NIHSS', 'FAC_BASE']:
    if feat in df_clin.columns:
        df_clin[feat] = pd.to_numeric(df_clin[feat], errors='coerce')
        df_clin[feat] = df_clin[feat].fillna(df_clin[feat].median())

# Preprocess binary features
if 'GENDER' in df_clin.columns:
    df_clin['GENDER'] = df_clin['GENDER'].fillna('Unknown')
    df_clin['GENDER'] = (df_clin['GENDER'].astype(str).str.upper().str.strip() == 'M').astype(int)

if 'HEMI' in df_clin.columns:
    df_clin['HEMI'] = df_clin['HEMI'].fillna('L')
    df_clin['HEMI'] = (df_clin['HEMI'].astype(str).str.upper().str.strip() == 'R').astype(int)

for feat in ['HTN', 'DIABETES', 'AF', 'TPA']:
    if feat in df_clin.columns:
        df_clin[feat] = df_clin[feat].fillna('N')
        df_clin[feat] = (df_clin[feat].astype(str).str.upper().str.strip() == 'Y').astype(int)

print(f"   Patients: {len(df_clin)}")
print(f"   Outcome: FAC≥4 = {df_clin['FAC12_binary'].sum()} ({df_clin['FAC12_binary'].mean()*100:.1f}%)")

# Test different clinical feature sets
print("\n2. Testing Clinical Feature Combinations...")
print("="*70)

y = df_clin['FAC12_binary'].values

test_configs = [
    ("Minimal (Age, NIHSS)", ['AGE', 'NIHSS']),
    ("With Baseline (Age, NIHSS, FAC_BASE)", ['AGE', 'NIHSS', 'FAC_BASE']),
    ("All Clinical", ['AGE', 'NIHSS', 'GENDER', 'HEMI', 'HTN', 'DIABETES', 'AF', 'TPA', 'FAC_BASE']),
]

results = []
for name, features in test_configs:
    # Check if features exist
    available_features = [f for f in features if f in df_clin.columns]
    if len(available_features) == 0:
        continue
    
    X = df_clin[available_features].values.astype(float)
    
    # Remove NaN rows
    finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X_clean = X[finite_mask]
    y_clean = y[finite_mask]
    
    if len(X_clean) < 30:
        continue
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # Test with Logistic Regression
    lr = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
    auc_scores = cross_val_score(lr, X_scaled, y_clean, cv=5, scoring='roc_auc')
    auc_mean = auc_scores.mean()
    
    results.append({
        'config': name,
        'n_features': len(available_features),
        'n_samples': len(X_clean),
        'auc_mean': auc_mean,
        'auc_std': auc_scores.std()
    })
    
    print(f"\n{name}:")
    print(f"   Features: {available_features}")
    print(f"   AUC: {auc_mean:.3f} ± {auc_scores.std():.3f}")

# Identify the problem
print("\n" + "="*70)
print("3. ANALYSIS")
print("="*70)

df_results = pd.DataFrame(results)
print(df_results[['config', 'n_features', 'auc_mean', 'auc_std']].to_string(index=False))

if len(results) >= 2:
    baseline_auc = results[0]['auc_mean']  # Minimal features
    with_baseline_auc = results[1]['auc_mean'] if len(results) > 1 else baseline_auc
    
    improvement = with_baseline_auc - baseline_auc
    
    print(f"\n📊 KEY FINDING:")
    print(f"   Minimal features (Age, NIHSS): {baseline_auc:.3f}")
    print(f"   With FAC_BASE added: {with_baseline_auc:.3f}")
    print(f"   Improvement from FAC_BASE: {improvement:+.3f}")
    
    if improvement > 0.05:
        print(f"\n⚠️  PROBLEM IDENTIFIED:")
        print(f"   FAC_BASE is TOO predictive!")
        print(f"   This leaves little room for imaging to add value.")
        print(f"   Baseline walking ability strongly predicts 3-month walking ability.")
        print(f"\n💡 SOLUTION:")
        print(f"   1. Remove FAC_BASE from clinical features")
        print(f"   2. Match Jo et al.'s approach (only Age, NIHSS)")
        print(f"   3. This gives imaging features a chance to help")
    else:
        print(f"\n✓ FAC_BASE provides minimal improvement")
        print(f"  Other factors may be limiting imaging contribution")

# Check feature correlation
print("\n4. Feature Correlation Analysis...")
print("="*70)

if 'FAC_BASE' in df_clin.columns and 'FAC12' in df_clin.columns:
    valid_mask = df_clin[['FAC_BASE', 'FAC12']].notna().all(axis=1)
    if valid_mask.sum() > 10:
        corr = df_clin.loc[valid_mask, ['FAC_BASE', 'FAC12']].corr().iloc[0, 1]
        print(f"Correlation between FAC_BASE and FAC12: {corr:.3f}")
        
        if corr > 0.7:
            print(f"\n⚠️  VERY HIGH CORRELATION!")
            print(f"   Baseline FAC is highly predictive of 3-month FAC")
            print(f"   This is expected but limits imaging contribution")

# Check if imaging features exist
print("\n5. Checking Imaging Features...")
print("="*70)

try:
    # Try to load saved features
    import glob
    feature_files = glob.glob("./radiomic_fac_analysis_UPGRADED/radiomics_features/*.csv")
    
    if feature_files:
        print(f"Found {len(feature_files)} feature files")
        
        # Load first one
        df_feat = pd.read_csv(feature_files[0])
        print(f"\nExample: {feature_files[0]}")
        print(f"  Shape: {df_feat.shape}")
        print(f"  Patients: {len(df_feat)}")
        
        # Check for NaN
        feature_cols = [c for c in df_feat.columns if c not in ['patient_id', 'loss_function']]
        nan_count = df_feat[feature_cols].isna().sum().sum()
        print(f"  NaN values: {nan_count}")
        
        if nan_count > 0:
            print(f"\n⚠️  Found NaN values in imaging features")
            print(f"   This indicates atlas alignment or extraction issues")
    else:
        print("No imaging feature files found yet")
        print("Run the main script first to extract features")
        
except Exception as e:
    print(f"Could not load imaging features: {e}")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)
print("""
Based on this analysis:

1. **If FAC_BASE is very predictive (>0.05 AUC improvement):**
   → Remove FAC_BASE from clinical features
   → Use only: Age, NIHSS (like Jo et al.)
   → This gives imaging a chance to contribute

2. **If imaging features have many NaN:**
   → Fix atlas alignment issues
   → Or remove atlas-based location features
   → Keep only volume + texture radiomics

3. **If clinical baseline is still very strong (>0.80 AUC):**
   → FAC may be clinically-dominated
   → This is a valid finding!
   → Report: "Clinical features sufficient for FAC prediction"

4. **Consider switching to FMUE outcome:**
   → Better anatomical relationship to motor regions
   → Your atlases are designed for upper limb motor
   → Likely to show imaging contribution
""")

print("="*70)
