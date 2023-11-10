import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Define a function to load the model and scaler with error handling
def load_model_scaler(model_path, scaler_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        return None, None

# Initialize Streamlit layout
st.title("Yash's Classification Models")

# Selectbox for choosing the dataset
dataset_name = st.selectbox('Choose the dataset:', ['Iris', 'Wine'])

# Feature names used during model training
iris_feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
wine_feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
    'pH', 'sulphates', 'alcohol', 'type_white'
]

# Load the saved model and scaler for Iris
iris_classifier, iris_scaler = load_model_scaler('KNN_Iris.pkl', 'Scaler_Iris.pkl')
if iris_classifier is None or iris_scaler is None:
    st.warning("Unable to load Iris model and scaler.")

# Load the saved model and scaler for Wine
wine_classifier, wine_scaler = load_model_scaler('wine_fraud.pkl', 'Scaler_Wine.pkl')
if wine_classifier is None or wine_scaler is None:
    st.warning("Unable to load Wine model and scaler.")

# Based on the dataset, set up the input fields and predict
if dataset_name == 'Iris' and iris_classifier is not None and iris_scaler is not None:
    st.subheader('Iris Species Prediction')
    sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, value=5.8, step=0.1)
    sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, value=3.0, step=0.1)
    petal_length = st.number_input('Petal Length (cm)', min_value=0.0, value=3.7, step=0.1)
    petal_width = st.number_input('Petal Width (cm)', min_value=0.0, value=1.2, step=0.1)
    
    # Predict button for Iris
    if st.button('Predict Iris Species', key='predict_iris'):
        iris_features = [sepal_length, sepal_width, petal_length, petal_width]
        iris_features_df = pd.DataFrame([iris_features], columns=iris_feature_names)
        iris_features_scaled = iris_scaler.transform(iris_features_df)
        iris_prediction = iris_classifier.predict(iris_features_scaled)
        iris_classes = ['Setosa', 'Versicolour', 'Virginica']
        iris_flower_name = iris_classes[iris_prediction[0]]
        st.success(f'Prediction: The Iris is a {iris_flower_name}')

elif dataset_name == 'Wine' and wine_classifier is not None and wine_scaler is not None:
    st.subheader('Wine Fraud Detection')
    fixed_acidity = st.slider('Fixed Acidity', 3.8, 15.9, 7.0)  
    volatile_acidity = st.slider('Volatile Acidity', 0.08, 1.58, 0.5)
    citric_acid = st.slider('Citric Acid', 0.0, 1.66, 0.25)
    residual_sugar = st.slider('Residual Sugar', 0.6, 65.8, 2.5)
    chlorides = st.slider('Chlorides', 0.009, 0.611, 0.05)
    free_sulfur_dioxide = st.slider('Free Sulfur Dioxide', 1, 289, 30)
    total_sulfur_dioxide = st.slider('Total Sulfur Dioxide', 6, 440, 100)
    density = st.slider('Density', 0.98711, 1.03898, 0.995)
    pH = st.slider('pH', 2.72, 4.01, 3.0)
    sulphates = st.slider('Sulphates', 0.22, 2.0, 0.5)
    alcohol = st.slider('Alcohol', 8.0, 14.9, 10.0)
    wine_type = st.selectbox('Wine Type', ['Red', 'White'])
    type_white = 1 if wine_type == 'White' else 0

    # Predict button for Wine
    if st.button('Predict Wine Authenticity', key='predict_wine'):
        wine_features = [
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
            pH, sulphates, alcohol, type_white
        ]
        wine_features_df = pd.DataFrame([wine_features], columns=wine_feature_names)
        wine_features_scaled = wine_scaler.transform(wine_features_df)
        wine_prediction = wine_classifier.predict(wine_features_scaled)
        wine_prediction_label = 'Legit' if wine_prediction[0] == 0 else 'Fraud'
        st.success(f'Prediction: The wine is {wine_prediction_label}')
