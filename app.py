import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Predictor")
st.caption("Interactive demo: estimate churn risk and review model evaluation metrics.")
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    model = Pipeline([("preprocess", preprocessor), ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "report": classification_report(y_test, y_pred, labels=[0, 1], target_names=["No churn", "Churn"], output_dict=True, zero_division=0),
        "rows": len(df), "test_rows": len(y_test)
    }
    return model, metrics

try:
    model, metrics = train_model()
except FileNotFoundError:
    st.error(f"Dataset not found: {DATA_PATH}. Place the CSV file beside app.py.")
    st.stop()
except Exception as exc:
    st.error(f"Could not train the model: {exc}")
    st.stop()

st.subheader("Customer details")
with st.form("customer_form"):
    left, right = st.columns(2)
    with left:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    with right:
        tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=840.0, step=10.0)
    submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

if submitted:
    customer = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection, "TechSupport": tech,
        "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total
    }])
    probability = float(model.predict_proba(customer)[0, 1])
    if probability >= 0.5:
        st.warning(f"Potential churn risk: {probability:.1%}")
        st.write("This profile is classified as more likely to churn by the model. Consider reviewing service experience and retention options.")
    else:
        st.success(f"Lower predicted churn risk: {probability:.1%}")
        st.write(f"Predicted churn probability: {probability:.1%}")
    st.caption("This is a model estimate, not a certainty.")

st.divider()
st.header("Model Evaluation")
st.write("Metrics below are calculated on a held-out test set. Precision, recall, and F1-score focus on customers who churned.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
c2.metric("Precision (Churn)", f"{metrics['precision']:.1%}")
c3.metric("Recall (Churn)", f"{metrics['recall']:.1%}")
c4.metric("F1-score (Churn)", f"{metrics['f1']:.1%}")

st.subheader("Confusion Matrix")
cm = metrics["confusion"]
cm_df = pd.DataFrame(cm, index=["Actual: No churn", "Actual: Churn"], columns=["Predicted: No churn", "Predicted: Churn"])
st.dataframe(cm_df, use_container_width=True)
st.caption("Rows are actual outcomes; columns are model predictions. Diagonal cells are correct predictions.")
with st.expander("Detailed classification report"):
    st.dataframe(pd.DataFrame(metrics["report"]).transpose().round(3), use_container_width=True)
with st.expander("About this demo"):
    st.write("This app trains a Logistic Regression model using the Telco Customer Churn dataset. Customer ID is excluded from model inputs.")
    st.write(f"Rows after cleaning: {metrics['rows']:,}")
    st.write(f"Rows in held-out test set: {metrics['test_rows']:,}")
    st.warning("This is a learning/demo project. Review model quality and context before using predictions in real decisions.")
