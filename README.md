# Customer Churn Prediction

## Project Overview
An interactive machine learning application that estimates customer churn risk using the Telco Customer Churn dataset. The application is built with Python and Streamlit and uses Logistic Regression to predict whether a customer is likely to discontinue their service.

## Features
- Interactive customer information form
- Customer churn risk prediction
- Model evaluation using Accuracy, Precision, Recall, and F1-score
- Confusion Matrix
- Detailed classification report

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Machine Learning — Logistic Regression

## Model Performance
The model was evaluated on a held-out test set.

| Metric | Score |
|---|---:|
| Accuracy | 72.6% |
| Precision (Churn) | 49.0% |
| Recall (Churn) | 79.7% |
| F1-score (Churn) | 60.7% |

These results indicate that the model identifies many actual churn cases, while some predicted churn cases are false positives. Further model tuning could improve performance.

## Dataset
The project uses the Telco Customer Churn dataset, originally obtained from Kaggle. Please refer to the original dataset page and its license before redistributing the data.

## How to Run Locally

1. Clone or download this repository.
2. Install the required packages:

   `pip install -r requirements.txt`

3. Run the Streamlit application:

   `python -m streamlit run app.py`

4. Open the local URL displayed in the terminal.

## Disclaimer
This is an educational project. Predictions are estimates and should not be treated as certain outcomes or used as the sole basis for real business decisions.
