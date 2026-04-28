import numpy as np
import gradio as gr
import joblib
from pathlib import Path

# Check for required model files
MODEL_FILES = ["random_forest_model.pkl", "scaler.pkl"]
missing_files = [f for f in MODEL_FILES if not Path(f).exists()]
if missing_files:
    raise FileNotFoundError(
        "Missing model artifacts: "
        + ", ".join(missing_files)
        + ". Run notebook first to generate models."
    )

# Load models
rf = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_risk(age, systolic_bp, diastolic, bs, body_temp,
                 bmi, prev_comp, preex_diabetes, gest_diabetes,
                 mental_health, heart_rate):
    """
    Predict maternal health risk based on patient vitals and history.
    Returns a comprehensive assessment with clinical context.
    """
    inp = np.array([[age, systolic_bp, diastolic, bs, body_temp,
                     bmi, prev_comp, preex_diabetes, gest_diabetes,
                     mental_health, heart_rate]], dtype=float)

    inp_sc = scaler.transform(inp)
    prob_rf = rf.predict_proba(inp_sc)[0][1]
    
    if prob_rf >= 0.65:
        risk_level = "HIGH RISK"
        risk_note = "The model is clearly above the high-risk threshold."
        color = "#b42318"
        banner_bg = "#fef3f2"
    elif prob_rf <= 0.35:
        risk_level = "LOW RISK"
        risk_note = "The model is clearly below the high-risk threshold."
        color = "#027a48"
        banner_bg = "#ecfdf3"
    else:
        risk_level = "BORDERLINE"
        risk_note = "The model is near the decision threshold, so interpret this cautiously."
        color = "#b26a00"
        banner_bg = "#fff7ed"
    
    # Build comprehensive output
    output = f"""
    <div style="padding: 20px; border-radius: 12px; background: #ffffff; border: 1px solid #d1d5db; box-shadow: 0 8px 24px rgba(0,0,0,0.06); color: #111827;">
        <div style="padding: 14px 16px; border-radius: 10px; background: {banner_bg}; border: 1px solid {color}; margin-bottom: 16px;">
            <h2 style="color: {color}; margin: 0; font-size: 28px; letter-spacing: 0.4px;">
                {risk_level}
            </h2>
            <p style="font-size: 14px; color: #374151; margin: 8px 0 0 0;">
                {risk_note}
            </p>
        </div>
        <p style="font-size: 18px; color: #111827; margin: 10px 0; font-weight: 600;">
            Risk Score: {prob_rf*100:.1f}%
        </p>
        
        <hr style="border: none; border-top: 1px solid #d1d5db; margin: 16px 0;">
        
        <div style="font-size: 15px; line-height: 1.6; color: #111827;">
            <p style="font-size: 15px; color: #111827; font-weight: 600; margin: 0 0 8px 0;">Clinical Summary:</p>
    """
    
    # Add context based on inputs
    risk_factors = []
    
    if age >= 35:
        risk_factors.append(f"• Advanced maternal age ({age} years)")
    
    if systolic_bp >= 140 or diastolic >= 90:
        risk_factors.append(f"• Elevated blood pressure ({systolic_bp}/{diastolic} mmHg)")
    
    if bs > 7.0:
        risk_factors.append(f"• Elevated blood sugar ({bs} mmol/L)")
    
    if bmi >= 30:
        risk_factors.append(f"• Obesity (BMI: {bmi})")
    elif bmi < 18.5:
        risk_factors.append(f"• Low BMI (BMI: {bmi})")
    
    if heart_rate > 100:
        risk_factors.append(f"• Elevated resting heart rate ({heart_rate} bpm)")
    
    if prev_comp == 1:
        risk_factors.append("• History of previous complications")
    
    if preex_diabetes == 1:
        risk_factors.append("• Preexisting diabetes")
    
    if gest_diabetes == 1:
        risk_factors.append("• Gestational diabetes")
    
    if mental_health == 1:
        risk_factors.append("• Mental health considerations")
    
    if risk_factors:
        for factor in risk_factors:
            output += f"            {factor}<br/>"
    else:
        output += "            No significant risk factors identified.<br/>"
    
    output += """
        </div>
        
        <hr style="border: none; border-top: 1px solid #d1d5db; margin: 16px 0;">
        
        <div style="font-size: 13px; color: #374151; font-style: italic; background: #f9fafb; padding: 12px 14px; border-radius: 8px; border: 1px solid #e5e7eb;">
            <p style="margin: 0; color: #374151;">
                💡 <b>Disclaimer:</b> This assessment is for educational purposes only 
                and should not replace professional medical consultation. 
                Please consult with a healthcare provider for accurate diagnosis.
            </p>
        </div>
    </div>
    """
    
    return output


# Professional healthcare-themed interface
custom_css = """
#component-0 {
    max-width: 900px;
    margin: 0 auto;
}
.gr-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.gr-box {
    border-radius: 8px;
    border: 1px solid #e0e0e0;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    # Header
    gr.Markdown("""
    # 🤰 Maternal Health Risk Assessment
    
    Enter patient vitals and health information to receive a risk assessment.
    This tool uses machine learning models trained on maternal health data.
    
    ---
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Vital Signs")
            age = gr.Slider(
                minimum=10, maximum=60, value=25, step=1,
                label="Age (years)",
                info="Patient age in years"
            )
            systolic_bp = gr.Slider(
                minimum=70, maximum=200, value=120, step=5,
                label="Systolic BP (mmHg)",
                info="Upper blood pressure reading"
            )
            diastolic = gr.Slider(
                minimum=40, maximum=130, value=80, step=5,
                label="Diastolic BP (mmHg)",
                info="Lower blood pressure reading"
            )
            heart_rate = gr.Slider(
                minimum=50, maximum=120, value=75, step=1,
                label="Heart Rate (bpm)",
                info="Resting heart rate"
            )
            body_temp = gr.Slider(
                minimum=95, maximum=105, value=98, step=0.1,
                label="Body Temperature (°F)",
                info="Core body temperature"
            )
        
        with gr.Column():
            gr.Markdown("### Metabolic Markers")
            bs = gr.Slider(
                minimum=6, maximum=20, value=8.0, step=0.1,
                label="Blood Sugar (mmol/L)",
                info="Fasting glucose level"
            )
            bmi = gr.Slider(
                minimum=10, maximum=50, value=25.0, step=0.1,
                label="BMI",
                info="Body Mass Index"
            )
            gr.Markdown("### Medical History")
            prev_comp = gr.Radio(
                choices=[0, 1], value=0, label="Previous Complications?",
                info="Any complications in previous pregnancies"
            )
            preex_diabetes = gr.Radio(
                choices=[0, 1], value=0, label="Preexisting Diabetes?",
                info="Type 1 or Type 2 diabetes diagnosis"
            )
            gest_diabetes = gr.Radio(
                choices=[0, 1], value=0, label="Gestational Diabetes?",
                info="Diabetes developed during pregnancy"
            )
            mental_health = gr.Radio(
                choices=[0, 1], value=0, label="Mental Health Concerns?",
                info="History of depression, anxiety, or other conditions"
            )
    
    # Assess button
    assess_btn = gr.Button("🔍 Assess Risk", variant="primary", scale=2, size="lg")
    
    # Output
    output = gr.HTML(label="Risk Assessment")
    
    # Connect button to prediction
    assess_btn.click(
        fn=predict_risk,
        inputs=[age, systolic_bp, diastolic, bs, body_temp, bmi, prev_comp, preex_diabetes, gest_diabetes, mental_health, heart_rate],
        outputs=output
    )
    
    # Footer
    gr.Markdown("""
    ---
    
    ### About This Tool
    
    **Model:** Random Forest Classifier trained on maternal health data
    
    **Accuracy:** ~85% on test data | **Key Factors:** Age, Blood Pressure, Blood Sugar, BMI
    
    **Important:** This tool is mostly for educational demonstration purposes. 
    Medical decisions should always involve qualified healthcare professionals.
    """)


if __name__ == "__main__":
    demo.launch(share=False)