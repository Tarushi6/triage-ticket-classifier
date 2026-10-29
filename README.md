
# Triage — AI Ticket Classification Console

A machine-learning-powered support ticket classification app built with Python and Streamlit.

## Features

- Predicts the support department
- Classifies ticket priority
- Predicts ticket topic
- Interactive dark-themed interface
- Sample tickets for quick testing
- Recent prediction history during the session

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- Joblib

## Run Locally

1. Clone this repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   streamlit run app.py
   ```

## Model Artifacts

The app expects these trained model files:

- `artifacts/model_department.pkl`
- `artifacts/model_priority.pkl`
- `artifacts_data3/model_topic.pkl`

Ensure the model files are available at these paths before running the app.
