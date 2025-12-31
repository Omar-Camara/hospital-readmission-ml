# Dataset

## 📥 Download Instructions

This project uses the **Diabetes 130-US Hospitals** dataset from UCI ML Repository.

### Option 1: Direct Download
1. Go to https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
2. Click "Download"
3. Extract `diabetic_data.csv`
4. Place in this `data/` folder

### Option 2: Kaggle
1. Go to https://www.kaggle.com/datasets/dubradave/hospital-readmissions
2. Download dataset
3. Extract and place `diabetic_data.csv` here

### Expected File
```
data/
└── diabetic_data.csv  (~100MB, 101,766 records)
```

## 📊 Dataset Information

**Source**: Beata Strack, Jonathan P. DeShazo, et al. (2014)  
**Title**: Impact of HbA1c Measurement on Hospital Readmission Rates  
**Published in**: BioMed Research International

**Records**: 101,766 patient encounters  
**Features**: 50+ (demographics, diagnoses, medications, procedures)  
**Target**: 30-day readmission (binary)  
**Time Period**: 1999-2008  
**Hospitals**: 130 US hospitals

## 🔒 Privacy Note

This dataset contains de-identified patient records compliant with HIPAA regulations. No protected health information (PHI) is included.

## 📚 Citation

If you use this dataset, please cite:

```
Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J., 
& Clore, J.N. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: 
Analysis of 70,000 Clinical Database Patient Records. BioMed Research International, 
2014, Article ID 781670.
```