# Pre-trained Models

The trained models are **not included in this repository** due to file size (50MB+).

## Download Models

**Option 1**: From Hugging Face Space
Download from: [https://huggingface.co/spaces/Username273183/hospital-readmission-predictor/blob/main/models_deployment.pkl](https://huggingface.co/spaces/Username273183/hospital-readmission-predictor/blob/main/models_deployment.pkl)

**Option 2**: Train Yourself
\`\`\`bash
python src/save_models.py
\`\`\`

This will create `models/models_deployment.pkl` locally.

## Model Files

Once downloaded, your `models/` folder should contain:
- `models_deployment.pkl` (~50MB) - XGBoost and Random Forest models
- `sample_patients.csv` (included) - Example data for testing

## Usage

\`\`\`python
import pickle

with open('models/models_deployment.pkl', 'rb') as f:
    models = pickle.load(f)

xgb_model = models['xgb_model']
rf_model = models['rf_model']
\`\`\`
