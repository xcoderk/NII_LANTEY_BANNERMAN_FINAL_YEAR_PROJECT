import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from fpdf import FPDF  # type: ignore
import tempfile
from joblib import load
from sklearn.base import BaseEstimator

# Page configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Paths to the pre-trained models
model_paths = {
    "Logistic Regression": r"C:\Users\Dell\Desktop\finalproject\f1\Logistic_Regression_model.pkl",
    "MLP Classifier": r"C:\Users\Dell\Desktop\finalproject\f1\MLP_Classifier_model.pkl",
    "Random Forest": r"C:\Users\Dell\Desktop\finalproject\f1\Random_Forest_model.pkl"
}

# Function to load a model safely using joblib
def load_model(filepath):
    try:
        model = load(filepath)
        if not isinstance(model, BaseEstimator):
            st.error(f"The object at {filepath} is not a valid scikit-learn model.")
            return None
        return model
    except FileNotFoundError:
        st.error(f"The file at {filepath} was not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Function to generate a PDF report
def generate_pdf_report(title, report_content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    for line in report_content:
        pdf.cell(0, 10, line, ln=True)

    return pdf

# Custom CSS for styling
st.markdown(
    """
    <style>
    .header-style {
        font-size: 35px;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
        margin-bottom: 10px;
    }
    .description-style {
        font-size: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main header
st.markdown('<div class="header-style">Credit Card Fraud Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="description-style">A comprehensive solution to analyze and detect credit card fraud effectively using machine learning.</div>', unsafe_allow_html=True)

# Sidebar - Select Analysis Mode
st.sidebar.header("Select Analysis Mode")
analysis_mode = st.sidebar.selectbox("Choose Analysis Mode:", ["Bulk Transaction Analysis", "Individual Transaction Analysis", "Exploratory Data Analysis"])

# Sidebar - Select Model
st.sidebar.header("Select Model")
selected_model = st.sidebar.selectbox("Choose a Model:", list(model_paths.keys()))

# Load the selected model
model_path = model_paths[selected_model]
model = load_model(model_path)

# Bulk Transaction Analysis Mode
features = ['distance_from_home', 'online_order', 'ratio_to_median_purchase_price']  # Adjust this list accordingly

if analysis_mode == "Bulk Transaction Analysis" and model is not None:
    st.sidebar.header("Upload Dataset for Bulk Analysis")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Upload a CSV file containing transaction data.")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.write("### Dataset Preview", data.head())
        st.write(f"**Shape:** {data.shape}")

        st.write("### Exploratory Data Analysis")
        if st.checkbox("Show data info", help="Display detailed information about the dataset."):
            st.write(data.info())

        if st.checkbox("Show class balance", help="Show the balance of fraud and non-fraud classes."):
            st.write(data['Class'].value_counts())

        st.write("#### Correlation Heatmap")
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, fmt=".2f")
        st.pyplot(plt)

        st.write("### Model Evaluation")
        target = st.selectbox("Select Target Column:", list(data.columns), help="Select the target column (label) for the model.")
        X = data[features]
        y = data[target]

        # Prediction using the selected model
        with st.spinner('Performing predictions...'):
            y_pred = model.predict(X)

        classification_report_str = classification_report(y, y_pred)
        st.write("**Classification Report**")
        st.text(classification_report_str)

        st.write("**Confusion Matrix**")
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        st.pyplot(plt)

        # Assuming your model pipeline doesn't have feature importance if it includes MLPClassifier, which doesn't support it
        if hasattr(model, 'feature_importances_'):
            st.write("### Feature Importance")
            feature_importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
            st.write(feature_importance.sort_values('Importance', ascending=False))

        # Suspicious Transactions Display
        st.write("### Transactions Based on Predictions")

        # Add the predictions to the dataset with "Fraud"/"Not Fraud" labels
        data['Prediction'] = y_pred
        data['Prediction Label'] = data['Prediction'].apply(lambda x: 'Fraud' if x == 1 else 'Not Fraud')

        # Separate into DataFrames for fraud and non-fraud
        fraud_transactions = data[data['Prediction'] == 1]
        non_fraud_transactions = data[data['Prediction'] == 0]

        # Display all fraudulent and non-fraudulent transactions
        st.write("**All Fraudulent Transactions:**")
        st.write(fraud_transactions)

        st.write("**All Non-Fraudulent Transactions:**")
        st.write(non_fraud_transactions)

        # Prepare the PDF report, summarizing all findings
        report_content = [
            f"Analysis Mode: {analysis_mode}",
            f"Model: {selected_model}",
            "",
            "Classification Report:",
            classification_report_str,
            "",
            "All Fraudulent Transactions:",
            fraud_transactions.to_string(index=False),
            "",
            "All Non-Fraudulent Transactions:",
            non_fraud_transactions.to_string(index=False)
        ]

        # Generate the PDF report
        pdf = generate_pdf_report("Credit Card Fraud Analysis Report", report_content)

        # Save to a temporary file and provide a download link
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf_file_path = tmp_file.name
            pdf.output(pdf_file_path)

        st.download_button(
            label="Download Analysis Report as PDF",
            data=open(pdf_file_path, "rb").read(),
            file_name="fraud_analysis_report.pdf",
            mime="application/pdf"
        )
# Individual Transaction Analysis Mode
elif analysis_mode == "Individual Transaction Analysis" and model is not None:
    st.write("### Individual Transaction Analysis")

    # Collecting individual feature inputs
    distance_from_home = st.number_input("Distance from Home (km)", min_value=0.0, max_value=10000.0, value=0.0, help="Distance of the transaction location from home.")
    online_order = st.slider("Is it an Online Order? (No: 0, Yes: 1)", min_value=0, max_value=1, value=0, help="Slider: No (0), Yes (1)")
    ratio_to_median_purchase = st.number_input("Ratio to Median Purchase Price", min_value=0.0, value=10.0, help="Ratio of the transaction amount to the median purchase price (minimum value of 10).")

    # Prepare input data for prediction
    input_data = {
        "distance_from_home": distance_from_home,
        "online_order": online_order,
        "ratio_to_median_purchase_price": ratio_to_median_purchase
    }

    input_df = pd.DataFrame([input_data])

    # Predicting the result for the individual transaction
    if st.button("Predict"):
        with st.spinner('Performing prediction...'):
            prediction_proba = model.predict_proba(input_df)[0]
            prediction = int(prediction_proba[1] > 0.5)
            prediction_text = 'Fraudulent' if prediction else 'Non-Fraudulent'
            fraud_probability = round(prediction_proba[1] * 100, 2)

        st.write(f"**Prediction:** {prediction_text}")
        st.write(f"**Fraud Probability:** {fraud_probability}%")

        # Generate a PDF report for the individual transaction
        report_content = [
            f"Analysis Mode: {analysis_mode}",
            f"Model: {selected_model}",
            "",
            "Transaction Details:",
            f"Distance from Home: {distance_from_home}",
            f"Online Order: {'Yes' if input_data['online_order'] else 'No'}",
            f"Ratio to Median Purchase Price: {ratio_to_median_purchase}",
            "",
            f"Prediction: {prediction_text}",
            f"Fraud Probability: {fraud_probability}%"
        ]

        pdf = generate_pdf_report("Individual Fraud Detection Report", report_content)

        # Save to a temporary file and provide a download link
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf_file_path = tmp_file.name
            pdf.output(pdf_file_path)

        st.download_button(
            label="Download Individual Analysis Report as PDF",
            data=open(pdf_file_path, "rb").read(),
            file_name="individual_analysis_report.pdf",
            mime="application/pdf"
        )

# Exploratory Data Analysis Mode
elif analysis_mode == "Exploratory Data Analysis":
    st.sidebar.header("Upload Dataset for EDA")
    uploaded_file = st.sidebar.file_uploader("Upload CSV for EDA", type=["csv"], help="Upload a CSV file for EDA.")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.write("### Exploratory Data Analysis (EDA) Overview")
        st.write("**Dataset Preview**", data.head())
        st.write(f"**Shape:** {data.shape}")
        st.write("**Summary Statistics**", data.describe())

        # Class Distribution
        if "Class" in data.columns:
            st.write("### Class Distribution")
            class_counts = data['Class'].value_counts()
            st.bar_chart(class_counts)

        # Correlation Heatmap
        st.write("### Correlation Matrix")
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, fmt=".2f")
        st.pyplot(plt)

        # Feature Distributions
        st.write("### Feature Distributions")
        selected_features = st.multiselect("Select Features for Distribution Plot:", list(data.columns), default=list(data.columns[:5]))
        plot_type = st.radio("Select Plot Type", ["Box Plot", "Violin Plot"])

        if plot_type == "Box Plot":
            plt.figure(figsize=(15, 8))
            sns.boxplot(data=data[selected_features], orient="h")
            st.pyplot(plt)
        else:
            plt.figure(figsize=(15, 8))
            sns.violinplot(data=data[selected_features], orient="h")
            st.pyplot(plt)

        # Pairplot
        if st.checkbox("Show Pairplot", help="Visualize relationships between features"):
            st.write("### Pairplot of Selected Features")
            sns.pairplot(data[selected_features])
            st.pyplot(plt)
