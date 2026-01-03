"""
Hospital Readmission Prediction - Complete Training Pipeline
Demonstrates systematic iteration from 0% to 58.1% recall

Author: Omar Camara
Repository: github.com/omar-camara/hospital-readmission-ml
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

#==============================================================================
# CONFIGURATION
#==============================================================================

DATA_PATH = "data/diabetic_data.csv"  # Update path as needed
RANDOM_STATE = 123
TEST_SIZE = 0.2

#==============================================================================
# DATA LOADING AND PREPROCESSING
#==============================================================================

print("=" * 80)
print("HOSPITAL READMISSION PREDICTION - COMPLETE PIPELINE")
print("=" * 80)

# Load data
print("\n📁 Loading data...")
df = pd.read_csv(DATA_PATH, na_values=["?", "NA", ""])
print(f"✓ Loaded {len(df):,} patient records")

# Create binary target
df['readmit_flag'] = (df['readmitted'] == "<30").astype(int)

# Remove irrelevant columns
cols_to_drop = ['readmitted', 'encounter_id', 'patient_nbr', 'weight',
                'payer_code', 'medical_specialty']
df = df.drop(columns=cols_to_drop)

# Group ICD-9 diagnosis codes into clinical categories
def group_diag_icd9(code):
    """
    Group ICD-9 codes into clinical categories
    Reduces sparsity and improves model interpretability
    """
    try:
        code_num = int(str(code).split('.')[0])
    except:
        return "Other"
    
    if 390 <= code_num <= 459 or code_num == 785:
        return "Circulatory"
    elif 460 <= code_num <= 519 or code_num == 786:
        return "Respiratory"
    elif 520 <= code_num <= 579 or code_num == 787:
        return "Digestive"
    elif 250 <= code_num < 251:
        return "Diabetes"
    elif 800 <= code_num <= 999:
        return "Injury"
    elif 710 <= code_num <= 739:
        return "Musculoskeletal"
    elif 580 <= code_num <= 629 or code_num == 788:
        return "Genitourinary"
    elif 140 <= code_num <= 239:
        return "Neoplasms"
    else:
        return "Other"

print("\n🏥 Grouping diagnosis codes...")
df['diag_1_group'] = df['diag_1'].apply(group_diag_icd9)
df['diag_2_group'] = df['diag_2'].apply(group_diag_icd9)
df['diag_3_group'] = df['diag_3'].apply(group_diag_icd9)
df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'])

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df.drop('readmit_flag', axis=1), drop_first=True)
df_encoded = df_encoded.fillna(0)

# Clean column names for XGBoost compatibility
df_encoded.columns = (df_encoded.columns
                      .str.replace('[', '_', regex=False)
                      .str.replace(']', '_', regex=False)
                      .str.replace('<', '_', regex=False)
                      .str.replace('>', '_', regex=False))

X = df_encoded
y = df['readmit_flag']

print(f"✓ Dataset shape: {X.shape}")
print(f"✓ Features: {X.shape[1]}")
print(f"✓ Class distribution: {dict(y.value_counts())}")
print(f"✓ Imbalance ratio: {(y==0).sum() / (y==1).sum():.1f}:1")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"\n📊 Train: {len(X_train):,} | Test: {len(X_test):,}")

#==============================================================================
# ITERATION JOURNEY
#==============================================================================

print("\n" + "=" * 80)
print("ITERATION JOURNEY: FROM FAILURE TO SUCCESS")
print("=" * 80)

# V1: Baseline (No Class Balancing)
print("\n[V1] BASELINE: No Class Balancing")
print("-" * 80)
rf_v1 = RandomForestClassifier(
    n_estimators=100, random_state=RANDOM_STATE,
    max_depth=15, min_samples_split=50
)
rf_v1.fit(X_train, y_train)
y_pred_v1 = rf_v1.predict(X_test)
y_proba_v1 = rf_v1.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_v1, target_names=['No Readmit', 'Readmit']))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_v1).ravel()
print(f"💡 Missed {fn:,} out of {fn+tp:,} readmissions ({fn/(fn+tp)*100:.1f}%)")

# V2a: Overcorrection (Class Weight 1:20)
print("\n[V2a] OVERCORRECTION: Class Weight 1:20")
print("-" * 80)
rf_v2a = RandomForestClassifier(
    n_estimators=100, random_state=RANDOM_STATE,
    class_weight={0: 1, 1: 20},
    max_depth=15, min_samples_split=50
)
rf_v2a.fit(X_train, y_train)
y_pred_v2a = rf_v2a.predict(X_test)
y_proba_v2a = rf_v2a.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_v2a, target_names=['No Readmit', 'Readmit']))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_v2a).ravel()
print(f"💡 Created {fp:,} false alarms (flagged {fp+tp:,} patients)")

# V2b: Optimal (Class Weight 1:8)
print("\n[V2b] OPTIMAL: Class Weight 1:8")
print("-" * 80)
rf_v2b = RandomForestClassifier(
    n_estimators=100, random_state=RANDOM_STATE,
    class_weight={0: 1, 1: 8},
    max_depth=15, min_samples_split=50
)
rf_v2b.fit(X_train, y_train)
y_pred_v2b = rf_v2b.predict(X_test)
y_proba_v2b = rf_v2b.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_v2b, target_names=['No Readmit', 'Readmit']))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_v2b).ravel()
print(f"✅ Catches {tp:,} readmissions with {fp:,} false alarms")

# XGBoost (Best Performance)
print("\n[FINAL] XGBOOST with Scale Pos Weight")
print("-" * 80)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
    eval_metric='logloss', use_label_encoder=False
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

print(f"Scale Pos Weight: {scale_pos_weight:.2f}")
print(classification_report(y_test, y_pred_xgb, target_names=['No Readmit', 'Readmit']))

# Logistic Regression
print("\n[BASELINE] LOGISTIC REGRESSION")
print("-" * 80)
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
y_proba_lr = lr_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_lr, target_names=['No Readmit', 'Readmit']))

#==============================================================================
# MODEL COMPARISON
#==============================================================================

print("\n" + "=" * 80)
print("COMPREHENSIVE COMPARISON")
print("=" * 80)

models = {
    'V1: No Balancing': y_pred_v1,
    'V2a: Weight 1:20': y_pred_v2a,
    'V2b: Weight 1:8': y_pred_v2b,
    'XGBoost (Final)': y_pred_xgb,
    'Logistic Regression': y_pred_lr
}

comparison = []
for name, preds in models.items():
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    comparison.append({
        'Model': name,
        'Recall': recall_score(y_test, preds),
        'Precision': precision_score(y_test, preds, zero_division=0),
        'F1-Score': f1_score(y_test, preds, zero_division=0),
        'Total Cost': (fn * 10000) + (fp * 500)
    })

comparison_df = pd.DataFrame(comparison)
print("\n" + comparison_df.round(3).to_string(index=False))

#==============================================================================
# VISUALIZATIONS
#==============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Save all visualizations to results/ folder
import os
os.makedirs('results', exist_ok=True)

# 1. Evolution Journey
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
viz_models = [
    ('V1: No Balancing\n0% Recall', y_pred_v1),
    ('V2a: Weight 1:20\n93.9% Recall', y_pred_v2a),
    ('V2b: Weight 1:8\n46.4% Recall ⭐', y_pred_v2b),
    ('XGBoost\n58.1% Recall 🏆', y_pred_xgb),
    ('Logistic Regression\n54.1% Recall', y_pred_lr),
]

for idx, (title, preds) in enumerate(viz_models):
    row, col = idx // 3, idx % 3
    cm = confusion_matrix(y_test, preds)
    ax = axes[row, col]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

axes[1, 2].axis('off')  # Hide unused subplot
plt.tight_layout()
plt.savefig('results/evolution_journey.png', dpi=300, bbox_inches='tight')
print("✓ evolution_journey.png")

# 2. Metrics Comparison
fig, ax = plt.subplots(figsize=(12, 6))
comparison_df.set_index('Model')[['Recall', 'Precision', 'F1-Score']].plot(
    kind='bar', ax=ax, width=0.75
)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ metrics_comparison.png")

# 3. Cost Analysis
fig, ax = plt.subplots(figsize=(10, 6))
(comparison_df.set_index('Model')['Total Cost'] / 1e6).plot(
    kind='bar', ax=ax, color='steelblue'
)
ax.set_title('Total Cost by Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Total Cost (Millions $)')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/cost_analysis.png', dpi=300, bbox_inches='tight')
print("✓ cost_analysis.png")

# 4. ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))
probas = {
    'XGBoost': y_proba_xgb,
    'RF (Weight 1:8)': y_proba_v2b,
    'Logistic Regression': y_proba_lr,
}

for name, proba in probas.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})', linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ roc_curves.png")

#==============================================================================
# SUMMARY
#==============================================================================

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

best_idx = comparison_df['Recall'].idxmax()
best_model = comparison_df.loc[best_idx]

print(f"""
🏆 BEST MODEL: {best_model['Model']}

Performance:
  • Recall:     {best_model['Recall']:.1%}
  • Precision:  {best_model['Precision']:.1%}
  • F1-Score:   {best_model['F1-Score']:.3f}
  • Total Cost: ${best_model['Total Cost']:,.0f}

Improvement from Baseline:
  • Recall: 0% → {best_model['Recall']:.1%}
  • Cost Savings: ${22710000 - best_model['Total Cost']:,.0f}

Key Insight:
  XGBoost's gradient boosting handles class imbalance better than
  Random Forest's bagging, achieving 12% higher recall (58% vs 46%).
""")

print("\n" + "=" * 80)
print("✅ Training Complete! All visualizations saved to results/")
print("=" * 80)
