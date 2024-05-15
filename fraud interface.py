import streamlit as st
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import os
import joblib
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

# Load your model here (using joblib or another method)
mlp_model_path = r"C:\Users\Dell\Desktop\finalproject\f1\MLP_classifier_model.pkl"
rf_model_path = r"C:\Users\Dell\Desktop\finalproject\f1\Random_Forest_model.pkl"
lr_model_path = r"C:\Users\Dell\Desktop\finalproject\f1\Logistic_Regression_model.pkl"
scaler_path = r"C:\Users\Dell\Desktop\finalproject\scaler.pkl"

# Loading the models
mlp_model = joblib.load(mlp_model_path)
rf_model = joblib.load(rf_model_path)
lr_model = joblib.load(lr_model_path)
scaler = joblib.load(scaler_path) 

if 'prediction_log' not in st.session_state:
    st.session_state.prediction_log = pd.DataFrame(columns=['Probability', 'Prediction'])

# Exploratory Data Analysis
def eda():
    st.subheader('Exploratory Data Analysis')
    data_file = st.file_uploader("Upload your dataset for EDA", type=["csv"])
    if data_file:
        data = pd.read_csv(data_file)
        if not data.empty:
            target = st.selectbox("Select the target feature for analysis", data.columns)
            perform_eda(data, target)

# Perform actual EDA
def perform_eda(data, target):
    st.write("Data Head:", data.head())
    display_summary_statistics(data)
    visualize_data_imbalance(data, target)
    display_correlations(data)
    visualize_missing_values(data)

# Train and Predict Function
def train_and_predict():
    st.subheader('Train Your Model and Predict Fraud')
    data_file = st.file_uploader("Upload your dataset for training", type=["csv"])
    if data_file:
        data = pd.read_csv(data_file)
        if not data.empty:
            prepare_and_train_model(data)

# Live Prediction Function
def live_prediction():
    st.subheader('Real-Time Fraud Prediction')
    distance_from_home = st.number_input('Distance From Home', min_value=0.0, format='%f')
    online_order = st.slider('Online Order', min_value=0, max_value=1, value=0)
    ratio_to_median_purchase_price = st.number_input('Ratio To Median Purchase Price', min_value=0.0, format='%f')
    if st.button("Predict"):
        predict_real_time(distance_from_home, online_order, ratio_to_median_purchase_price)

# Bulk Prediction Function
def bulk_prediction():
    st.subheader('Bulk Prediction Using MLP Model')
    uploaded_file = st.file_uploader("Upload your CSV file for prediction", type=["csv"])
    if uploaded_file:
        predict_bulk(uploaded_file)

# Helper Functions for data processing, model training and prediction
def display_summary_statistics(data):
    st.write("Summary Statistics:")
    st.dataframe(data.describe())

def visualize_data_imbalance(data, target):
    st.write("Class Distribution in Target:")
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target, data=data)
    plt.title('Data Class Distribution')
    st.pyplot(plt)

def display_correlations(data):
    st.write("Correlation Matrix:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)

def visualize_missing_values(data):
    st.write("Missing Values Heatmap:")
    plt.figure(figsize=(12, 6))
    sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    st.pyplot(plt)

def predict_real_time(distance_from_home, online_order, ratio_to_median_purchase_price):
    input_data = pd.DataFrame({
        'distance_from_home': [distance_from_home],
        'online_order': [online_order],
        'ratio_to_median_purchase_price': [ratio_to_median_purchase_price]
    })
    input_data_scaled = scaler.transform(input_data)
    prediction = mlp_model.predict(input_data_scaled)
    prediction_proba = mlp_model.predict_proba(input_data_scaled)
    st.write('Fraud' if prediction[0] == 1 else 'Not Fraud')
    st.write('Probability of Fraud: {:.2f}%'.format(prediction_proba[0][1] * 100))

def predict_bulk(file):
    data = pd.read_csv(file)
    st.write("Data Overview:")
    st.dataframe(data.head())
    if 'distance_from_home' in data.columns and 'online_order' in data.columns and 'ratio_to_median_purchase_price' in data.columns:
        data_scaled = scaler.transform(data[['distance_from_home', 'online_order', 'ratio_to_median_purchase_price']])
        predictions = mlp_model.predict(data_scaled)
        prediction_proba = mlp_model.predict_proba(data_scaled)
        data['Prediction'] = predictions
        data['Prediction Label'] = ['Fraud' if x == 1 else 'Not Fraud' for x in predictions]
        st.write("Prediction Results:")
        st.dataframe(data[['Prediction', 'Prediction Label']])
    else:
        st.error("Required features are missing from the dataset.")

def visualize_data_overview(data):
    st.write("Data Overview:")
    st.dataframe(data.head())
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    st.pyplot(plt)

def select_features(data):
    st.write("Feature Selection:")
    all_features = data.columns.tolist()
    selected_features = st.multiselect('Choose your features for the model:', all_features, default=all_features)
    return selected_features

def preprocess_data(data, features):
    st.write("Preprocessing Data:")
    if st.checkbox('Remove Missing Values'):
        data = data.dropna(subset=features)
    st.write("Updated Data Preview:")
    st.dataframe(data.head())

    return data[features]

def balance_data(features, labels):
    st.write("Balancing Data:")
    balance_method = st.radio("Select balancing technique:", ["SMOTE", "UnderSampling"])
    if balance_method == "SMOTE":
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(features, labels)
    elif balance_method == "UnderSampling":
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(features, labels)
    st.write("Balanced Class Distribution:")
    class_distribution = sorted(Counter(y_res).items())
    st.bar_chart(dict(class_distribution))
    return X_res, y_res

def train_model(features, labels, model_choice):
    st.write("Model Training:")
    if model_choice == "MLP Model":
        model = mlp_model  # Assuming mlp_model is preloaded
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(random_state=42)

    # Split data for training and validation
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    st.write(f"Model Trained. Accuracy: {accuracy:.2f}")
    st.write("Confusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    st.write(cm)

    return model


def prepare_and_train_model(data):
    target = st.selectbox("Select the target variable:", data.columns)
    features = data.drop(columns=[target])
    labels = data[target]
    st.write("Features and Labels ready for training.")
    # Model training logic goes here

def main():
    st.title('Credit Card Fraud Detection System')
    options = st.sidebar.radio('Select an option:', ['Home', 'Exploratory Data Analysis', 'Live Prediction', 'Bulk Prediction'])

    if options == 'Home':
        st.subheader('Welcome to the Credit Card Fraud Detection System')
        st.write('This application helps in detecting fraudulent credit card transactions.')

    elif options == 'Exploratory Data Analysis':
        st.subheader('Exploratory Data Analysis')
        data_file = st.file_uploader("Upload your dataset for EDA", type=["csv"])
        if data_file:
            data = pd.read_csv(data_file)
            if not data.empty:
                target = st.selectbox("Select the target feature for analysis", data.columns)
                perform_eda(data, target)

    elif options == 'Live Prediction':
        st.subheader('Real-Time Fraud Prediction')
        with st.form("live_prediction_form"):
            distance_from_home = st.number_input('Distance From Home', min_value=0.0, format='%f')
            online_order = st.number_input('Online Order', min_value=0, max_value=1)
            ratio_to_median_purchase_price = st.number_input('Ratio To Median Purchase Price', min_value=0.0, format='%f')
            submitted = st.form_submit_button("Predict")
            if submitted:
                predict_live(distance_from_home, online_order, ratio_to_median_purchase_price)

    elif options == 'Bulk Prediction':
        st.subheader('Bulk Prediction Using MLP Model')
        bulk_prediction()

def perform_eda(data, target):
    """ Perform exploratory data analysis on the dataset. """
    st.write("Data Head:", data.head())
    st.write("Summary Statistics:", data.describe())
    visualize_data_imbalance(data, target)
    display_correlations(data)
    visualize_missing_values(data)

def bulk_prediction():
    uploaded_file = st.file_uploader("Upload your CSV file for prediction", type=["csv"])
    submit_button = st.button("Predict")
    if submit_button and uploaded_file:
        data = pd.read_csv(uploaded_file)
        expected_features = ['distance_from_home', 'online_order', 'ratio_to_median_purchase_price']
        if not all(feature in data.columns for feature in expected_features):
            missing_features = [feature for feature in expected_features if feature not in data.columns]
            st.error(f"Uploaded file is missing the following required columns: {', '.join(missing_features)}")
        else:
            data_for_prediction = data[expected_features]
            data_scaled = scaler.transform(data_for_prediction)
            predictions = mlp_model.predict(data_scaled)
            prediction_proba = mlp_model.predict_proba(data_scaled)
            data['Prediction'] = predictions
            data['Prediction Label'] = data['Prediction'].apply(lambda x: 'Fraud' if x == 1 else 'Not Fraud')
            st.write("Prediction Results:")
            st.dataframe(data[['Prediction', 'Prediction Label']])
            visualize_prediction_distribution(data)

def visualize_prediction_distribution(data):
    """Visualize the distribution of predictions."""
    st.write("Visualization of Prediction Distribution:")
    plt.figure(figsize=(8, 4))
    sns.countplot(x='Prediction Label', data=data)
    plt.title('Distribution of Predictions')
    plt.xlabel('Prediction Outcome')
    plt.ylabel('Count')
    st.pyplot()

def predict_live(distance_from_home, online_order, ratio_to_median_purchase_price):
    """ Make live predictions based on user input."""
    input_data = pd.DataFrame({
        'distance_from_home': [distance_from_home],
        'online_order': [online_order],
        'ratio_to_median_purchase_price': [ratio_to_median_purchase_price]
    })
    input_data_scaled = scaler.transform(input_data)
    prediction = mlp_model.predict(input_data_scaled)
    prediction_proba = mlp_model.predict_proba(input_data_scaled)
    st.subheader('Prediction:')
    st.write('Fraud' if prediction[0] == 1 else 'Not Fraud')
    st.write('Probability of Fraud: {:.2f}%'.format(prediction_proba[0][1] * 100))

if __name__ == '__main__':
    main()
