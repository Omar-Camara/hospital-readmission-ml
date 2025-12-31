"""
Save Trained Models for Deployment
Generates models_deployment.pkl for Gradio app

Author: Omar Camara
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

#==============================================================================
# CONFIGURATION
#==============================================================================

DATA_PATH = "data/diabetic_data.csv"
OUTPUT_PATH = "models/models_deployment.pkl"
SAMPLE_CSV_PATH = "models/sample_patients.csv"

print("=" * 80)
print("SAVING MODELS FOR DEPLOYMENT")
print("=" * 80)

#==============================================================================
# PREPROCESSING
#==============================================================================

print("\n1. Preprocessing data...")

df = pd.read_csv(DATA_PATH, na_values=["?", "NA", ""])
df['readmit_flag'] = (df['readmitted'] == "<30").astype(int)

cols_to_drop = ['readmitted', 'encounter_id', 'patient_nbr', 'weight',
                'payer_code', 'medical_specialty']
df = df.drop(columns=cols_to_drop)

def group_diag_icd9(code):
    try:
        code_num = int(str(code).split('.')[0])
    except:
        return "Other"
    
    if 390 <= code_num <= 459 or code_num == 785: return "Circulatory"
    elif 460 <= code_num <= 519 or code_num == 786: return "Respiratory"
    elif 520 <= code_num <= 579 or code_num == 787: return "Digestive"
    elif 250 <= code_num < 251: return "Diabetes"
    elif 800 <= code_num <= 999: return "Injury"
    elif 710 <= code_num <= 739: return "Musculoskeletal"
    elif 580 <= code_num <= 629 or code_num == 788: return "Genitourinary"
    elif 140 <= code_num <= 239: return "Neoplasms"
    else: return "Other"

df['diag_1_group'] = df['diag_1'].apply(group_diag_icd9)
df['diag_2_group'] = df['diag_2'].apply(group_diag_icd9)
df['diag_3_group'] = df['diag_3'].apply(group_diag_icd9)
df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'])

df_encoded = pd.get_dummies(df.drop('readmit_flag', axis=1), drop_first=True).fillna(0)

# Clean column names for XGBoost
df_encoded.columns = (df_encoded.columns
                      .str.replace('[', '_', regex=False)
                      .str.replace(']', '_', regex=False)
                      .str.replace('<', '_', regex=False)
                      .str.replace('>', '_', regex=False))

X, y = df_encoded, df['readmit_flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

print(f"✓ Processed {len(X):,} records into {X.shape[1]} features")

#==============================================================================
# TRAIN MODELS
#==============================================================================

print("\n2. Training models...")

# XGBoost
xgb_model = XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    scale_pos_weight=7.96, random_state=123,
    eval_metric='logloss', use_label_encoder=False
)
xgb_model.fit(X_train, y_train)
print("✓ XGBoost trained")

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100, random_state=123,
    class_weight={0: 1, 1: 8},
    max_depth=15, min_samples_split=50
)
rf_model.fit(X_train, y_train)
print("✓ Random Forest trained")

#==============================================================================
# SAVE DEPLOYMENT PACKAGE
#==============================================================================

print("\n3. Creating deployment package...")

deployment_package = {
    'xgb_model': xgb_model,
    'rf_model': rf_model,
    'feature_names': X_train.columns.tolist(),
    'training_stats': {
        'n_samples': len(X_train),
        'n_features': len(X_train.columns),
        'class_ratio': (y_train == 0).sum() / (y_train == 1).sum()
    },
    'categorical_mappings': {
        'age': ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
        'gender': ['Male', 'Female'],
        'insulin': ['No', 'Down', 'Steady', 'Up'],
        'diabetesMed': ['Yes', 'No'],
        'primary_diagnosis': ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes',
                             'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other'],
    }
}

import os
os.makedirs('models', exist_ok=True)

with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(deployment_package, f)

print(f"✓ Saved to {OUTPUT_PATH}")

#==============================================================================
# VERIFY
#==============================================================================

print("\n4. Verifying models...")

with open(OUTPUT_PATH, 'rb') as f:
    loaded = pickle.load(f)

test_sample = X_test.iloc[0:1]
xgb_pred = loaded['xgb_model'].predict_proba(test_sample)[0][1]
rf_pred = loaded['rf_model'].predict_proba(test_sample)[0][1]

print(f"✓ XGBoost test prediction: {xgb_pred:.1%}")
print(f"✓ RF test prediction: {rf_pred:.1%}")

#==============================================================================
# CREATE SAMPLE DATA
#==============================================================================

print("\n5. Creating sample patients...")

sample_patients = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003'],
    'age': ['[70-80)', '[50-60)', '[30-40)'],
    'gender': ['Male', 'Female', 'Male'],
    'time_in_hospital': [7, 3, 2],
    'num_lab_procedures': [60, 40, 30],
    'num_procedures': [2, 1, 0],
    'num_medications': [25, 15, 10],
    'number_diagnoses': [9, 7, 4],
    'insulin': ['Up', 'Steady', 'No'],
    'diabetesMed': ['Yes', 'Yes', 'No'],
    'primary_diagnosis': ['Circulatory', 'Diabetes', 'Injury']
})

sample_patients.to_csv(SAMPLE_CSV_PATH, index=False)
print(f"✓ Saved to {SAMPLE_CSV_PATH}")

print("\n" + "=" * 80)
print("✅ MODELS READY FOR DEPLOYMENT!")
print("=" * 80)
print(f"\nGenerated files:")
print(f"  • {OUTPUT_PATH} (~50MB)")
print(f"  • {SAMPLE_CSV_PATH}")
print("\nNext step: Run Gradio app with these models")
print("=" * 80)
