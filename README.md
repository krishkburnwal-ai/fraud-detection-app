# Credit Card Fraud Detection

## Overview

This project uses Machine Learning and Deep Learning techniques to detect fraudulent credit card transactions. The model is trained on historical transaction data and predicts whether a transaction is legitimate or fraudulent.

## Features

* Credit card fraud prediction
* Data preprocessing and feature scaling
* Trained Deep Learning model (.h5)
* Interactive Streamlit web application
* Model evaluation using Confusion Matrix and ROC-AUC Score

## Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Streamlit
* Matplotlib

## Project Structure

├── app.py
├── credit_card_fraud_detection.ipynb
├── fraud_model.h5
├── time_scaler.pkl,amount_scaler.pkl
├── requirements.txt
└── README.md

## Model Performance

* ROC-AUC Score: 0.97
* Optimized for handling highly imbalanced fraud datasets

## Installation

1. Clone the repository

git clone <repository-url>

2. Install dependencies

pip install -r requirements.txt

3. Run the application

streamlit run app.py

## Future Improvements

* Hyperparameter tuning
* Real-time fraud detection API


