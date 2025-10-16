# DiabetesPredictor-API
School project
A Flask-based API that predicts the likelihood of diabetes using a trained machine learning model. Features include median imputation, feature scaling, and JWT-based authentication.

Features

Predict diabetes risk from 8 clinical features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

Handles missing values and scales input data

JWT authentication for secure endpoints

Returns both predictions and probabilities

Usage

Run the API:
1. python app.py
2. Authenticate: POST /auth/login
Body: { "username": "Caspar", "password": "gud123" }
3. Make predictions:
 POST /predict
Headers: { "Authorization": "Bearer <token>" }
Body: JSON with the 8 required features

   Project Structure
app.py
best_diabetes_model.pkl
diabetes_scaler.pkl
imputer_median_values.pkl
requirements.txt
README.md


