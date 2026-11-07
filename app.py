
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 14px !important;
}

h1 {
    font-size: 26px !important;
}
h2 {
    font-size: 22px !important;
}
h3 {
    font-size: 18px !important;
}

[data-testid="stMetricValue"] {
    font-size: 25px !important;
}

[data-testid="stMetricLabel"] {
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)



st.title("🌱 Dry Bean Classification Application")

st.write(
    "Upload a test dataset, select a trained machine learning model "
    "and evaluate its classification performance."
)

st.info(
    "This application uses previously trained models. "
    "No model retraining takes place."
)

# --------------------------------------------------
# 2. MODEL AND ARTIFACT PATHS
# --------------------------------------------------

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.pkl",
    "Support Vector Machine (SVM)": "svm.pkl"
}

# --------------------------------------------------
# 3. LOAD SAVED MODELS AND PREPROCESSING ARTIFACTS
# --------------------------------------------------

@st.cache_resource
def load_artifacts():

    models = {}

    for model_name, filename in MODEL_FILES.items():

        file_path = os.path.join(MODEL_DIR, filename)

        models[model_name] = joblib.load(file_path)

    scaler = joblib.load(
        os.path.join(MODEL_DIR, "scaler.pkl")
    )

    label_encoder = joblib.load(
        os.path.join(MODEL_DIR, "label_encoder.pkl")
    )

    return models, scaler, label_encoder


try:
    models, scaler, label_encoder = load_artifacts()

except Exception as e:
    st.error(
        f"Unable to load saved models or preprocessing files: {e}"
    )
    st.stop()


# --------------------------------------------------
# 4. GET EXPECTED FEATURE NAMES
# --------------------------------------------------

# The scaler was fitted using the training feature DataFrame.
# Its feature_names_in_ attribute preserves the original order.

if hasattr(scaler, "feature_names_in_"):

    feature_names = list(scaler.feature_names_in_)

else:

    feature_names = [
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "AspectRation",
        "Eccentricity",
        "ConvexArea",
        "EquivDiameter",
        "Extent",
        "Solidity",
        "roundness",
        "Compactness",
        "ShapeFactor1",
        "ShapeFactor2",
        "ShapeFactor3",
        "ShapeFactor4"
    ]


# --------------------------------------------------
# 5. DATASET UPLOAD
# --------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        test_data = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")

        st.dataframe(test_data.head())

        st.write("Dataset Shape:", test_data.shape)

    except Exception as e:

        st.error(f"Unable to read the uploaded CSV: {e}")
        st.stop()

    # --------------------------------------------------
    # 6. VALIDATE DATASET
    # --------------------------------------------------

    st.header("2. Dataset Validation")

    required_columns = feature_names + ["Class"]

    missing_columns = [
        col for col in required_columns
        if col not in test_data.columns
    ]

    if missing_columns:

        st.error(
            f"Missing required columns: {missing_columns}"
        )
        st.stop()

    # Select features in the exact training order.
    X_test = test_data[feature_names].copy()

    y_test = test_data["Class"].copy()

    # Convert input features to numeric values.
    try:

        X_test = X_test.apply(
            pd.to_numeric,
            errors="raise"
        )

    except Exception:

        st.error("All 16 input features must contain numeric values.")
        st.stop()

    # Check for missing and infinite feature values.
    if X_test.isnull().any().any():

        st.error("The uploaded dataset contains missing feature values.")
        st.stop()

    if not np.isfinite(X_test.to_numpy()).all():

        st.error("The uploaded dataset contains infinite feature values.")
        st.stop()

    # Encode actual target labels.
    try:

        y_test_encoded = label_encoder.transform(y_test)

    except ValueError:

        st.error(
            "The target column contains class labels "
            "that were not present during model training."
        )
        st.stop()

    st.success("Dataset validation completed successfully.")

    # --------------------------------------------------
    # 7. MODEL SELECTION
    # --------------------------------------------------

    st.header("3. Select Classification Model")

    selected_model_name = st.selectbox(
        "Choose a machine learning model",
        list(MODEL_FILES.keys())
    )

    # --------------------------------------------------
    # 8. PREPROCESS TEST FEATURES
    # --------------------------------------------------

    # Logistic Regression, KNN and SVM require scaled features.
    # Decision Tree and Gaussian Naive Bayes use original features.

    if selected_model_name in [
        "Logistic Regression",
        "KNN",
        "Support Vector Machine (SVM)"
    ]:

        X_processed = scaler.transform(X_test)

    else:

        X_processed = X_test

    # --------------------------------------------------
    # 9. MODEL PREDICTION
    # --------------------------------------------------

    selected_model = models[selected_model_name]

    try:

        y_pred = selected_model.predict(X_processed)

        y_prob = selected_model.predict_proba(X_processed)

    except Exception as e:

        st.error(f"Model prediction failed: {e}")
        st.stop()

    # --------------------------------------------------
    # 10. CALCULATE EVALUATION METRICS
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test_encoded,
        y_pred
    )

    precision = precision_score(
        y_test_encoded,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test_encoded,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test_encoded,
        y_pred,
        average="weighted",
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test_encoded,
        y_pred
    )

    # Calculate multiclass AUC.
    try:

        auc = roc_auc_score(
            y_test_encoded,
            y_prob,
            labels=np.arange(len(label_encoder.classes_)),
            multi_class="ovr",
            average="weighted"
        )

    except ValueError:

        auc = None

    # --------------------------------------------------
    # 11. DISPLAY EVALUATION METRICS
    # --------------------------------------------------

    st.header("4. Model Evaluation Results")

    st.subheader(
        f"Selected Model: {selected_model_name}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    col2.metric(
        "AUC Score",
        f"{auc:.4f}" if auc is not None else "N/A"
    )

    col3.metric(
        "Precision",
        f"{precision:.4f}"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Recall",
        f"{recall:.4f}"
    )

    col5.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    col6.metric(
        "MCC Score",
        f"{mcc:.4f}"
    )

    # --------------------------------------------------
    # 12. CONFUSION MATRIX
    # --------------------------------------------------

    st.header("5. Confusion Matrix")

    cm = confusion_matrix(
        y_test_encoded,
        y_pred,
        labels=np.arange(len(label_encoder.classes_))
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        annot_kws={"size": 9},
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=ax
    )

    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("Actual Class", fontsize=10)
    ax.set_title(
        f"Confusion Matrix - {selected_model_name}",
        fontsize=13
    )

    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    plt.close(fig)

    # --------------------------------------------------
    # 13. CLASSIFICATION REPORT
    # --------------------------------------------------

    st.header("6. Classification Report")

    report = classification_report(
        y_test_encoded,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.style.format("{:.4f}"),
        use_container_width=True
    )

    # --------------------------------------------------
    # 14. FINAL STATUS
    # --------------------------------------------------

    st.success(
        "Model evaluation completed successfully. "
        "The selected saved model was used without retraining."
    )

else:

    st.info(
        "Please upload a CSV file containing the 16 input features "
        "and the target column named 'Class' to begin evaluation."
    )

# --------------------------------------------------
# END OF APPLICATION
# --------------------------------------------------
