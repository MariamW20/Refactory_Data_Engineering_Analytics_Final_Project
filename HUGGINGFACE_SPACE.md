# Hugging Face Space Quick Start

Use this if you want the fastest deployment path.

## Upload these files

- `app.py`
- `dashboard.py`
- `requirements.txt`
- `random_forest_model.pkl`
- `scaler.pkl`

## Space settings

- SDK: Gradio
- App file: `app.py`
- Python version: 3.11

## What to do

1. Go to Hugging Face and create a new **Gradio Space**.
2. Upload the files listed above.
3. Wait for the Space to build.
4. Open the Space URL and test the dashboard.

## Short description you can paste

Maternal Health Risk Prediction is a Gradio dashboard that predicts maternal health risk from clinical inputs using a trained Random Forest model. It provides a risk level, risk score, and contextual risk factors for demonstration and educational use.

## If the build fails

- Check that `random_forest_model.pkl` and `scaler.pkl` are uploaded.
- Make sure the Space is using `app.py` as the entrypoint.
- Keep the requirements file unchanged unless the build log shows a missing dependency.