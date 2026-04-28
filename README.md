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
- Classical model: Random Forest.
- Neural network model: Keras MLP.
- Evaluation metrics include Accuracy, Precision, Recall, F1, ROC-AUC, classification report, and confusion matrix.

#### 4) Findings and Presentation
- Findings section included in notebook.
- Interactive demo in `dashboard.py`.

## Files

- `maternal_health_analysis.ipynb`: Main assignment notebook.
- `dashboard.py`: Standalone Gradio demo app.
- `Dataset - Updated.csv`: Dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Run Order

1. Open and run all cells in `maternal_health_analysis.ipynb`.
2. This will save model artifacts:
	- `random_forest_model.pkl`
	- `scaler.pkl`
	- `imputer.pkl`
	- `neural_network_model.keras`
3. Run the dashboard demo:

```bash
python dashboard.py
```

## Notes

- The dashboard checks for required model files and shows a clear error if training artifacts are missing.
