import numpy as np
import gradio as gr
import joblib

rf = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_risk(age, systolic_bp, diastolic, bs, body_temp,
                 bmi, prev_comp, preex_diabetes, gest_diabetes,
                 mental_health, heart_rate):

    inp = np.array([[age, systolic_bp, diastolic, bs, body_temp,
                     bmi, prev_comp, preex_diabetes, gest_diabetes,
                     mental_health, heart_rate]], dtype=float)

    inp_sc = scaler.transform(inp)

    prob_rf = rf.predict_proba(inp_sc)[0][1]
    label_rf = "HIGH RISK" if prob_rf >= 0.5 else "LOW RISK"

    return (
        f"### Maternal Health Risk Prediction\n\n"
        f"Prediction: **{label_rf}**\n\n"
        f"High-risk probability: **{prob_rf:.1%}**"
    )

demo = gr.Interface(
    fn=predict_risk,
    inputs=[
        gr.Slider(10, 60, value=25, label="Age"),
        gr.Slider(70, 200, value=120, label="Systolic BP"),
        gr.Slider(40, 130, value=80, label="Diastolic BP"),
        gr.Slider(6, 20, value=8.0, step=0.1, label="Blood Sugar"),
        gr.Slider(95, 105, value=98, label="Body Temperature"),
        gr.Slider(10, 50, value=25.0, step=0.1, label="BMI"),
        gr.Radio([0, 1], value=0, label="Previous Complications"),
        gr.Radio([0, 1], value=0, label="Preexisting Diabetes"),
        gr.Radio([0, 1], value=0, label="Gestational Diabetes"),
        gr.Radio([0, 1], value=0, label="Mental Health Issues"),
        gr.Slider(58, 92, value=75, label="Heart Rate"),
    ],
    outputs=gr.Markdown(),
    title="Maternal Health Risk Predictor",
    description="Enter patient details to predict maternal health risk.",
    flagging_mode="never"
)

demo.launch()