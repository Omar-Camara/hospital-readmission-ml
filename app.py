"""
Hospital Readmission Risk Predictor - Advanced Version
Features: Real Models, Batch Predictions, SHAP, Model Comparison

Author: Omar Camara
"""

import gradio as gr
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD MODELS
# ============================================================================

print("Loading models...")
try:
    with open('models_deployment.pkl', 'rb') as f:
        deployment_pkg = pickle.load(f)
    
    xgb_model = deployment_pkg['xgb_model']
    rf_model = deployment_pkg['rf_model']
    feature_names = deployment_pkg['feature_names']
    cat_mappings = deployment_pkg['categorical_mappings']
    
    # Initialize SHAP explainer (compute once)
    X_sample = pd.DataFrame(np.random.randn(100, len(feature_names)), columns=feature_names)
    shap_explainer = shap.TreeExplainer(xgb_model)
    
    MODELS_LOADED = True
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"⚠ Could not load models: {e}")
    print("Running in demo mode with simplified predictions")
    MODELS_LOADED = False
    xgb_model = None
    rf_model = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_feature_vector(age, gender, time_in_hospital, num_lab_procedures,
                         num_procedures, num_medications, number_diagnoses,
                         insulin, diabetesMed, primary_diagnosis):
    """Convert user inputs to model feature vector"""
    
    if not MODELS_LOADED:
        return None
    
    # Create base dataframe with all features set to 0
    features = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Set numerical features
    features['time_in_hospital'] = time_in_hospital
    features['num_lab_procedures'] = num_lab_procedures
    features['num_procedures'] = num_procedures
    features['num_medications'] = num_medications
    features['number_diagnoses'] = number_diagnoses
    
    # Set categorical features (one-hot encoded)
    # Age
    age_col = f'age_{age}'
    if age_col in features.columns:
        features[age_col] = 1
    
    # Gender
    if gender == 'Female' and 'gender_Female' in features.columns:
        features['gender_Female'] = 1
    
    # Insulin
    if insulin != 'No':
        insulin_col = f'insulin_{insulin}'
        if insulin_col in features.columns:
            features[insulin_col] = 1
    
    # Diabetes Med
    if diabetesMed == 'Yes' and 'diabetesMed_Yes' in features.columns:
        features['diabetesMed_Yes'] = 1
    
    # Primary diagnosis
    diag_col = f'diag_1_group_{primary_diagnosis}'
    if diag_col in features.columns:
        features[diag_col] = 1
    
    return features

def calculate_simplified_risk(age, gender, time_in_hospital, num_lab_procedures,
                              num_procedures, num_medications, number_diagnoses,
                              insulin, diabetesMed, primary_diagnosis):
    """Fallback simplified risk calculation if models not loaded"""
    
    risk_score = 0
    
    age_risk = {'[0-10)': 0, '[10-20)': 0.5, '[20-30)': 1, '[30-40)': 1.5,
                '[40-50)': 2, '[50-60)': 2.5, '[60-70)': 3, '[70-80)': 4, 
                '[80-90)': 5, '[90-100)': 5}
    risk_score += age_risk.get(age, 2.5)
    risk_score += min(time_in_hospital * 0.3, 3)
    risk_score += min(num_medications * 0.2, 4)
    risk_score += min(num_lab_procedures * 0.05, 2)
    risk_score += min(number_diagnoses * 0.3, 3)
    
    if insulin in ['Down', 'Steady', 'Up']:
        risk_score += 2
    if diabetesMed == 'Yes':
        risk_score += 1.5
    
    diag_risk = {'Circulatory': 3, 'Respiratory': 2.5, 'Diabetes': 3.5,
                 'Digestive': 2, 'Injury': 1.5, 'Other': 2}
    risk_score += diag_risk.get(primary_diagnosis, 2)
    
    return min(max(risk_score * 4, 0), 100)

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_single_patient(age, gender, time_in_hospital, num_lab_procedures,
                          num_procedures, num_medications, number_diagnoses,
                          insulin, diabetesMed, primary_diagnosis,
                          admission_type, discharge_disposition):
    """Single patient prediction with detailed breakdown"""
    
    if MODELS_LOADED:
        # Real model predictions
        features = create_feature_vector(
            age, gender, time_in_hospital, num_lab_procedures,
            num_procedures, num_medications, number_diagnoses,
            insulin, diabetesMed, primary_diagnosis
        )
        
        xgb_prob = xgb_model.predict_proba(features)[0][1] * 100
        rf_prob = rf_model.predict_proba(features)[0][1] * 100
        
        # Use XGBoost as primary
        risk_percentage = xgb_prob
        
    else:
        # Simplified calculation
        risk_percentage = calculate_simplified_risk(
            age, gender, time_in_hospital, num_lab_procedures,
            num_procedures, num_medications, number_diagnoses,
            insulin, diabetesMed, primary_diagnosis
        )
        xgb_prob = risk_percentage
        rf_prob = risk_percentage * 0.9  # Simulate slight difference
    
    # Risk category
    if risk_percentage < 30:
        risk_category = "Low Risk"
        color = "green"
        recommendation = "✅ Standard discharge planning and routine follow-up within 2-3 weeks."
    elif risk_percentage < 60:
        risk_category = "Medium Risk"
        color = "orange"
        recommendation = "⚠️ Enhanced discharge planning with follow-up call within 48 hours. Schedule appointment within 7 days."
    else:
        risk_category = "High Risk"
        color = "red"
        recommendation = "🚨 Intensive discharge planning required. Post-discharge phone call within 24 hours. Schedule appointment within 3-5 days. Consider home health services."
    
    # Create gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{risk_category}<br><span style='font-size:0.8em'>30-Day Readmission Risk</span>", 
                'font': {'size': 24}},
        number = {'suffix': "%", 'font': {'size': 40}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "darkblue"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 60], 'color': '#fff3cd'},
                {'range': [60, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 58
            }
        }
    ))
    
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=20))
    
    # Summary
    summary = f"""
    ## 🏥 Risk Assessment Summary
    
    **Primary Model (XGBoost)**: {xgb_prob:.1f}% risk
    **Secondary Model (Random Forest)**: {rf_prob:.1f}% risk
    **Consensus**: {risk_category}
    
    ---
    
    ### 📋 Patient Profile
    - **Demographics**: {age} year old {gender}
    - **Hospital Stay**: {time_in_hospital} days
    - **Clinical Complexity**: {num_medications} medications, {number_diagnoses} diagnoses
    - **Primary Diagnosis**: {primary_diagnosis}
    - **Diabetes Management**: Insulin {insulin}, Diabetes Med {diabetesMed}
    
    ### 🎯 Clinical Recommendation
    {recommendation}
    
    ---
    
    *This prediction uses XGBoost model (58.1% recall, 18.2% precision)*
    """
    
    return fig, summary, xgb_prob, rf_prob

def predict_batch(csv_file):
    """Batch predictions from CSV upload"""
    
    if csv_file is None:
        return None, "Please upload a CSV file"
    
    try:
        df = pd.read_csv(csv_file.name)
        
        # Validate required columns
        required = ['age', 'gender', 'time_in_hospital', 'num_lab_procedures',
                   'num_procedures', 'num_medications', 'number_diagnoses',
                   'insulin', 'diabetesMed', 'primary_diagnosis']
        
        missing = [col for col in required if col not in df.columns]
        if missing:
            return None, f"Missing columns: {missing}"
        
        # Make predictions
        predictions = []
        for idx, row in df.iterrows():
            if MODELS_LOADED:
                features = create_feature_vector(
                    row['age'], row['gender'], row['time_in_hospital'],
                    row['num_lab_procedures'], row['num_procedures'],
                    row['num_medications'], row['number_diagnoses'],
                    row['insulin'], row['diabetesMed'], row['primary_diagnosis']
                )
                xgb_prob = xgb_model.predict_proba(features)[0][1] * 100
                rf_prob = rf_model.predict_proba(features)[0][1] * 100
            else:
                xgb_prob = calculate_simplified_risk(
                    row['age'], row['gender'], row['time_in_hospital'],
                    row['num_lab_procedures'], row['num_procedures'],
                    row['num_medications'], row['number_diagnoses'],
                    row['insulin'], row['diabetesMed'], row['primary_diagnosis']
                )
                rf_prob = xgb_prob * 0.9
            
            risk_cat = "High" if xgb_prob >= 60 else "Medium" if xgb_prob >= 30 else "Low"
            
            predictions.append({
                'Patient_ID': df.loc[idx, 'patient_id'] if 'patient_id' in df.columns else f'P{idx+1:03d}',
                'XGBoost_Risk_%': round(xgb_prob, 1),
                'RF_Risk_%': round(rf_prob, 1),
                'Risk_Category': risk_cat,
                'Age': row['age'],
                'Primary_Diagnosis': row['primary_diagnosis']
            })
        
        results_df = pd.DataFrame(predictions)
        
        # Create visualization
        fig = go.Figure()
        
        # Scatter plot of predictions
        colors = results_df['Risk_Category'].map({'Low': 'green', 'Medium': 'orange', 'High': 'red'})
        
        fig.add_trace(go.Scatter(
            x=results_df['XGBoost_Risk_%'],
            y=results_df['RF_Risk_%'],
            mode='markers',
            marker=dict(
                size=12,
                color=colors,
                line=dict(width=1, color='white')
            ),
            text=results_df['Patient_ID'],
            hovertemplate='<b>%{text}</b><br>XGBoost: %{x:.1f}%<br>RF: %{y:.1f}%<extra></extra>'
        ))
        
        fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                     line=dict(color="gray", width=1, dash="dash"))
        
        fig.update_layout(
            title="Batch Predictions: XGBoost vs Random Forest",
            xaxis_title="XGBoost Risk %",
            yaxis_title="Random Forest Risk %",
            height=500,
            hovermode='closest'
        )
        
        # Summary text
        high_risk = (results_df['Risk_Category'] == 'High').sum()
        med_risk = (results_df['Risk_Category'] == 'Medium').sum()
        low_risk = (results_df['Risk_Category'] == 'Low').sum()
        
        summary = f"""
        ### 📊 Batch Prediction Results
        
        **Total Patients**: {len(results_df)}
        
        **Risk Distribution**:
        - 🚨 High Risk: {high_risk} patients ({high_risk/len(results_df)*100:.1f}%)
        - ⚠️ Medium Risk: {med_risk} patients ({med_risk/len(results_df)*100:.1f}%)
        - ✅ Low Risk: {low_risk} patients ({low_risk/len(results_df)*100:.1f}%)
        
        **Model Agreement**:
        - Average difference: {abs(results_df['XGBoost_Risk_%'] - results_df['RF_Risk_%']).mean():.1f}%
        - Correlation: {results_df['XGBoost_Risk_%'].corr(results_df['RF_Risk_%']):.3f}
        
        Download the results table below for detailed analysis.
        """
        
        return fig, summary, results_df
        
    except Exception as e:
        return None, f"Error processing CSV: {str(e)}", None

def generate_shap_explanation(age, gender, time_in_hospital, num_lab_procedures,
                              num_procedures, num_medications, number_diagnoses,
                              insulin, diabetesMed, primary_diagnosis):
    """Generate SHAP feature importance explanation"""
    
    if not MODELS_LOADED:
        return None, "SHAP analysis requires models to be loaded. Running in demo mode."
    
    try:
        features = create_feature_vector(
            age, gender, time_in_hospital, num_lab_procedures,
            num_procedures, num_medications, number_diagnoses,
            insulin, diabetesMed, primary_diagnosis
        )
        
        # Calculate SHAP values
        shap_values = shap_explainer.shap_values(features)
        
        # Get feature contributions
        feature_contrib = pd.DataFrame({
            'Feature': features.columns,
            'Value': features.values[0],
            'SHAP': shap_values[0]
        })
        
        # Filter to non-zero and sort by absolute SHAP
        feature_contrib = feature_contrib[feature_contrib['SHAP'].abs() > 0.001]
        feature_contrib = feature_contrib.sort_values('SHAP', key=abs, ascending=False).head(15)
        feature_contrib = feature_contrib.reset_index(drop=True)  # Fix the hierarchical index issue
        
        # Create waterfall-style plot
        fig = go.Figure()
        
        colors = ['red' if x > 0 else 'blue' for x in feature_contrib['SHAP']]
        
        fig.add_trace(go.Bar(
            y=feature_contrib['Feature'],
            x=feature_contrib['SHAP'],
            orientation='h',
            marker=dict(color=colors),
            text=[f"{val:.3f}" for val in feature_contrib['SHAP']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="SHAP Feature Importance (Top 15 Factors)",
            xaxis_title="SHAP Value (Impact on Prediction)",
            yaxis_title="Feature",
            height=500,
            showlegend=False
        )
        
        # Create explanation text
        top_increasing = feature_contrib[feature_contrib['SHAP'] > 0].head(3)
        top_decreasing = feature_contrib[feature_contrib['SHAP'] < 0].head(3)
        
        explanation = f"""
        ### 🔍 SHAP Explanation
        
        SHAP (SHapley Additive exPlanations) shows how each feature contributes to the prediction.
        
        **Top Factors INCREASING Risk**:
        """
        
        for idx, row in top_increasing.iterrows():
            explanation += f"\n- **{row['Feature']}**: +{row['SHAP']:.3f} (value: {row['Value']})"
        
        explanation += "\n\n**Top Factors DECREASING Risk**:"
        
        for idx, row in top_decreasing.iterrows():
            explanation += f"\n- **{row['Feature']}**: {row['SHAP']:.3f} (value: {row['Value']})"
        
        explanation += f"""
        
        ---
        
        **Base Value (Average Prediction)**: {shap_explainer.expected_value:.3f}
        **Final Prediction**: {shap_explainer.expected_value + shap_values[0].sum():.3f}
        
        Red bars push risk UP, blue bars push risk DOWN.
        """
        
        return fig, explanation
        
    except Exception as e:
        return None, f"Error generating SHAP explanation: {str(e)}"

def compare_models_detailed(age, gender, time_in_hospital, num_lab_procedures,
                           num_procedures, num_medications, number_diagnoses,
                           insulin, diabetesMed, primary_diagnosis):
    """Detailed comparison between XGBoost and Random Forest"""
    
    if MODELS_LOADED:
        features = create_feature_vector(
            age, gender, time_in_hospital, num_lab_procedures,
            num_procedures, num_medications, number_diagnoses,
            insulin, diabetesMed, primary_diagnosis
        )
        
        xgb_prob = xgb_model.predict_proba(features)[0][1] * 100
        rf_prob = rf_model.predict_proba(features)[0][1] * 100
    else:
        risk = calculate_simplified_risk(
            age, gender, time_in_hospital, num_lab_procedures,
            num_procedures, num_medications, number_diagnoses,
            insulin, diabetesMed, primary_diagnosis
        )
        xgb_prob = risk
        rf_prob = risk * 0.9
    
    # Create comparison visualization
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('XGBoost Prediction', 'Random Forest Prediction'),
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}]]
    )
    
    # XGBoost gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=xgb_prob,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "#d4edda"},
                {'range': [30, 60], 'color': "#fff3cd"},
                {'range': [60, 100], 'color': "#f8d7da"}
            ]
        }
    ), row=1, col=1)
    
    # Random Forest gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=rf_prob,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 30], 'color': "#d4edda"},
                {'range': [30, 60], 'color': "#fff3cd"},
                {'range': [60, 100], 'color': "#f8d7da"}
            ]
        }
    ), row=1, col=2)
    
    fig.update_layout(height=350)
    
    # Comparison analysis
    diff = abs(xgb_prob - rf_prob)
    agreement = "Strong" if diff < 10 else "Moderate" if diff < 20 else "Weak"
    
    comparison = f"""
    ### 🔬 Model Comparison Analysis
    
    **XGBoost (Primary)**: {xgb_prob:.1f}%
    - Recall: 58.1% | Precision: 18.2%
    - Best overall performance
    - Uses gradient boosting
    
    **Random Forest (Secondary)**: {rf_prob:.1f}%
    - Recall: 46.4% | Precision: 20.4%
    - Good baseline performance
    - Uses ensemble bagging
    
    **Agreement Level**: {agreement} ({diff:.1f}% difference)
    
    ---
    
    ### 📊 Model Selection Reasoning
    
    **Why XGBoost is Primary**:
    - ✅ Higher recall (58% vs 46%) - catches more readmissions
    - ✅ Lower cost ($12.5M vs $14.2M)
    - ✅ Better AUC (0.678 vs 0.674)
    - ✅ Sequential learning handles imbalance better
    
    **Why Keep Random Forest**:
    - ✅ Validation/sanity check
    - ✅ Slightly better precision
    - ✅ Different algorithm = diverse perspective
    - ✅ Ensemble potential
    
    **Clinical Decision**: When models disagree by >20%, recommend manual review.
    """
    
    return fig, comparison

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

custom_css = """
.gradio-container {font-family: 'Arial', sans-serif;}
.risk-header {background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
              color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;}
"""

with gr.Blocks(title="Hospital Readmission Predictor - Advanced", css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    <div class='risk-header'>
    <h1>🏥 Hospital Readmission Risk Predictor</h1>
    <h3>Advanced ML Demo with Real Models, SHAP Explanations, and Batch Processing</h3>
    <p>Developed by <b>Omar Camara</b> | <a href='https://github.com/yourusername' style='color: white;'>GitHub</a> | 
    <a href='https://linkedin.com/in/yourprofile' style='color: white;'>LinkedIn</a></p>
    </div>
    """)
    
    with gr.Tabs():
        # TAB 1: Single Patient Prediction
        with gr.Tab("🔮 Single Patient Prediction"):
            gr.Markdown("### Predict readmission risk for an individual patient")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 👤 Demographics")
                    age = gr.Dropdown(
                        choices=cat_mappings['age'] if MODELS_LOADED else ['[60-70)'],
                        value='[60-70)',
                        label="Age Group"
                    )
                    gender = gr.Radio(choices=['Male', 'Female'], value='Male', label="Gender")
                    
                    gr.Markdown("#### 🏥 Clinical Metrics")
                    time_in_hospital = gr.Slider(1, 14, value=3, step=1, label="Days in Hospital")
                    num_lab_procedures = gr.Slider(1, 132, value=44, step=1, label="Lab Procedures")
                    num_procedures = gr.Slider(0, 6, value=1, step=1, label="Procedures")
                    num_medications = gr.Slider(1, 81, value=16, step=1, label="Medications")
                    number_diagnoses = gr.Slider(1, 16, value=7, step=1, label="Diagnoses")
                    
                with gr.Column(scale=1):
                    gr.Markdown("#### 💊 Diabetes Management")
                    insulin = gr.Dropdown(choices=['No', 'Down', 'Steady', 'Up'], value='Steady', label="Insulin")
                    diabetesMed = gr.Radio(choices=['Yes', 'No'], value='Yes', label="Diabetes Medication")
                    
                    gr.Markdown("#### 🔬 Diagnosis & Admission")
                    primary_diagnosis = gr.Dropdown(
                        choices=cat_mappings['primary_diagnosis'] if MODELS_LOADED else ['Circulatory'],
                        value='Circulatory',
                        label="Primary Diagnosis"
                    )
                    admission_type = gr.Dropdown(
                        choices=cat_mappings['admission_type'] if MODELS_LOADED else ['Emergency'],
                        value='Emergency',
                        label="Admission Type"
                    )
                    discharge_disposition = gr.Dropdown(
                        choices=cat_mappings['discharge_disposition'] if MODELS_LOADED else ['Home'],
                        value='Home',
                        label="Discharge Disposition"
                    )
            
            predict_btn = gr.Button("🔮 Predict Risk", variant="primary", size="lg")
            
            with gr.Row():
                risk_gauge = gr.Plot(label="Risk Assessment")
                risk_summary = gr.Markdown()
            
            xgb_risk_out = gr.Number(label="XGBoost Risk %", visible=False)
            rf_risk_out = gr.Number(label="RF Risk %", visible=False)
            
            predict_btn.click(
                fn=predict_single_patient,
                inputs=[age, gender, time_in_hospital, num_lab_procedures, num_procedures,
                       num_medications, number_diagnoses, insulin, diabetesMed,
                       primary_diagnosis, admission_type, discharge_disposition],
                outputs=[risk_gauge, risk_summary, xgb_risk_out, rf_risk_out]
            )
        
        # TAB 2: Batch Predictions
        with gr.Tab("📊 Batch Predictions"):
            gr.Markdown("""
            ### Upload CSV for Bulk Risk Assessment
            
            **Required Columns**: `age`, `gender`, `time_in_hospital`, `num_lab_procedures`, 
            `num_procedures`, `num_medications`, `number_diagnoses`, `insulin`, `diabetesMed`, 
            `primary_diagnosis`
            
            **Optional**: `patient_id` (will auto-generate if missing)
            """)
            
            with gr.Row():
                csv_upload = gr.File(label="Upload Patient CSV", file_types=['.csv'])
                gr.Examples(
                    examples=[['sample_patients.csv']],
                    inputs=[csv_upload],
                    label="Try Sample CSV"
                )
            
            batch_btn = gr.Button("📊 Analyze Batch", variant="primary", size="lg")
            
            batch_plot = gr.Plot(label="Batch Analysis Visualization")
            batch_summary = gr.Markdown()
            batch_results = gr.Dataframe(label="Detailed Results (Download Below)")
            
            batch_btn.click(
                fn=predict_batch,
                inputs=[csv_upload],
                outputs=[batch_plot, batch_summary, batch_results]
            )
        
        # TAB 3: SHAP Explanation
        with gr.Tab("🔍 SHAP Explanation"):
            gr.Markdown("""
            ### Feature Importance via SHAP Values
            
            Understand which factors contribute most to the prediction for a specific patient.
            SHAP (SHapley Additive exPlanations) provides model-agnostic explanations.
            """)
            
            with gr.Row():
                with gr.Column():
                    shap_age = gr.Dropdown(choices=cat_mappings['age'] if MODELS_LOADED else ['[60-70)'], 
                                          value='[70-80)', label="Age")
                    shap_gender = gr.Radio(choices=['Male', 'Female'], value='Male', label="Gender")
                    shap_time = gr.Slider(1, 14, value=7, step=1, label="Days in Hospital")
                    shap_labs = gr.Slider(1, 132, value=60, step=1, label="Lab Procedures")
                    shap_procs = gr.Slider(0, 6, value=2, step=1, label="Procedures")
                with gr.Column():
                    shap_meds = gr.Slider(1, 81, value=25, step=1, label="Medications")
                    shap_diags = gr.Slider(1, 16, value=9, step=1, label="Diagnoses")
                    shap_insulin = gr.Dropdown(choices=['No', 'Down', 'Steady', 'Up'], value='Up', label="Insulin")
                    shap_diabmed = gr.Radio(choices=['Yes', 'No'], value='Yes', label="Diabetes Med")
                    shap_prim_diag = gr.Dropdown(choices=cat_mappings['primary_diagnosis'] if MODELS_LOADED else ['Circulatory'],
                                                value='Circulatory', label="Primary Diagnosis")
            
            shap_btn = gr.Button("🔍 Generate SHAP Explanation", variant="primary", size="lg")
            
            shap_plot = gr.Plot(label="SHAP Feature Importance")
            shap_explanation = gr.Markdown()
            
            shap_btn.click(
                fn=generate_shap_explanation,
                inputs=[shap_age, shap_gender, shap_time, shap_labs, shap_procs,
                       shap_meds, shap_diags, shap_insulin, shap_diabmed, shap_prim_diag],
                outputs=[shap_plot, shap_explanation]
            )
        
        # TAB 4: Model Comparison
        with gr.Tab("⚖️ Model Comparison"):
            gr.Markdown("""
            ### XGBoost vs Random Forest: Side-by-Side
            
            Compare predictions from both models to understand their agreement and differences.
            """)
            
            with gr.Row():
                with gr.Column():
                    comp_age = gr.Dropdown(choices=cat_mappings['age'] if MODELS_LOADED else ['[60-70)'],
                                          value='[60-70)', label="Age")
                    comp_gender = gr.Radio(choices=['Male', 'Female'], value='Female', label="Gender")
                    comp_time = gr.Slider(1, 14, value=5, step=1, label="Days in Hospital")
                    comp_labs = gr.Slider(1, 132, value=50, step=1, label="Lab Procedures")
                    comp_procs = gr.Slider(0, 6, value=1, step=1, label="Procedures")
                with gr.Column():
                    comp_meds = gr.Slider(1, 81, value=20, step=1, label="Medications")
                    comp_diags = gr.Slider(1, 16, value=8, step=1, label="Diagnoses")
                    comp_insulin = gr.Dropdown(choices=['No', 'Down', 'Steady', 'Up'], value='Steady', label="Insulin")
                    comp_diabmed = gr.Radio(choices=['Yes', 'No'], value='Yes', label="Diabetes Med")
                    comp_prim_diag = gr.Dropdown(choices=cat_mappings['primary_diagnosis'] if MODELS_LOADED else ['Diabetes'],
                                                value='Diabetes', label="Primary Diagnosis")
            
            compare_btn = gr.Button("⚖️ Compare Models", variant="primary", size="lg")
            
            compare_plot = gr.Plot(label="Model Predictions Comparison")
            compare_analysis = gr.Markdown()
            
            compare_btn.click(
                fn=compare_models_detailed,
                inputs=[comp_age, comp_gender, comp_time, comp_labs, comp_procs,
                       comp_meds, comp_diags, comp_insulin, comp_diabmed, comp_prim_diag],
                outputs=[compare_plot, compare_analysis]
            )
    
    # Footer
    gr.Markdown("""
    ---
    ### 📊 Model Performance Summary
    
    | Model | Recall | Precision | F1-Score | AUC | Total Cost |
    |-------|--------|-----------|----------|-----|------------|
    | **XGBoost (Primary)** | **58.1%** | 18.2% | 27.7% | **0.678** | **$12.5M** |
    | Random Forest | 46.4% | 20.4% | 28.3% | 0.674 | $14.2M |
    | Logistic Regression | 54.1% | 17.0% | 25.9% | 0.650 | $13.4M |
    
    **Training Data**: 101,766 diabetic patient records | **Class Imbalance**: 8:1 ratio
    
    ---
    
    ### ⚠️ Disclaimer
    
    This tool is for **educational and research purposes only**. Do not use as the sole basis for 
    clinical decisions. Always consult qualified healthcare professionals and consider institutional 
    protocols, patient-specific factors, and clinical guidelines.
    
    ---
    
    **Built with**: Python, XGBoost, scikit-learn, SHAP, Gradio, Plotly  
    **Author**: Omar Camara | Syracuse University  
    **Contact**: omcamara@syr.edu | [GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourprofile)
    """)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")