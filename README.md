# Maternal Health Risk Prediction Project

This repository is organized to match the project requirements.

## Project Coverage

### Part 1 - GitHub Repo
This project is structured and documented for submission as a complete repository.

### Part 2 - Implementation

#### 1) Data Description
- Implemented in notebook: `maternal_health_analysis.ipynb`
- Includes shape, data types, summary statistics, class counts, missing values, and sample rows.

#### 2) Data Exploration and Visualization
- Implemented in notebook: `maternal_health_analysis.ipynb`
- Includes at least two univariate visualizations and two multivariate visualizations.
- All charts include labels/titles and interpretation notes.

#### 3) Machine Learning
- Implemented in notebook: `maternal_health_analysis.ipynb`
- Data cleaning and preparation steps included.
- Classical model: Random Forest with feature importance visualization.
- Neural network model: Keras MLP.
- Evaluation metrics include Accuracy, Precision, Recall, F1, ROC-AUC, classification report, confusion matrix, and ROC curve comparison.

#### 4) Findings and Presentation
- Findings section included in notebook.
- Interactive demo in `dashboard.py` with professional healthcare UI.

## Dashboard Improvements
The dashboard has been completely redesigned for professional use:
- **Professional Theme:** Healthcare-inspired soft color scheme
- **Organized Layout:** Inputs grouped by category (Vital Signs, Metabolic Markers, Medical History)
- **Color-Coded Output:** Red for HIGH RISK, Green for LOW RISK
- **Contextual Analysis:** Displays specific identified risk factors for each patient
- **Educational Content:** Clear disclaimers and model information
- **Better UX:** Larger inputs, helper text, and natural language

## Files

- `maternal_health_analysis.ipynb`: Main assignment notebook (complete analysis pipeline).
- `dashboard.py`: Standalone Gradio demo with redesigned professional UI.
- `Dataset - Updated.csv`: Maternal health dataset (1205 records).
- `requirements.txt`: Python package dependencies.

## Setup

```bash
cd C:\Users\Admin\Desktop\Maternal_Health

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install jupyter ipykernel
```

## Run Order

### 1. Run the Notebook
1. Open `maternal_health_analysis.ipynb` in VS Code
2. Select the venv kernel
3. Click "Run All"
4. Model artifacts generated:
   - `random_forest_model.pkl`
   - `scaler.pkl`
   - `imputer.pkl`

### 2. Run the Dashboard
```bash
python dashboard.py
```
Open the Gradio interface in your browser (typically `http://localhost:7860`)

## Notes

- Dashboard validates model files exist before launching
- All models are trained during notebook execution
- Educational tool designed for demonstration purposes
