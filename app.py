import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, create_access_token, jwt_required

model_filename = 'best_diabetes_model.pkl'
scaler_filename = 'diabetes_scaler.pkl'
imputer_filename = 'imputer_median_values.pkl'

REQUIRED_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

COLS_TO_IMPUTE = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

USERS = {"Caspar": "gud123"}


app = Flask(__name__)

app.config["JWT_SECRET_KEY"] = "super-secret-key" 
jwt = JWTManager(app)


try:
    scaler = joblib.load(scaler_filename)
    print("Scaler loaded successfully.")
except FileNotFoundError:
    print(f"Error: The scaler file '{scaler_filename}' was not found.")
    scaler = None

try:
    median_values = joblib.load(imputer_filename)
    print("Median imputer loaded successfully.")
except FileNotFoundError:
    print(f"Error: The imputer file '{imputer_filename}' was not found.")
    median_values = None


try:
    model = joblib.load(model_filename)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: The model file '{model_filename}' was not found.")

@app.route('/predict', methods=['POST'])
@jwt_required()
def predict():

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid input: No JSON data provided"}), 400
    
    if not isinstance(data, list):
            data = [data]
    
    try:
            input_df = pd.DataFrame(data)
            input_df = input_df[REQUIRED_FEATURES]

    except KeyError:
        return jsonify({
            "error": f"Invalid input: Missing required features", 
        "Required features are": {REQUIRED_FEATURES}
        }), 400
    input_df[COLS_TO_IMPUTE] = input_df[COLS_TO_IMPUTE].replace(0, np.nan)
    
    input_df = input_df.fillna(median_values)
    
    input_scaled = scaler.transform(input_df)
    
    try:
        predictions = model.predict(input_scaled).tolist()
        probabilities = model.predict_proba(input_scaled).tolist()

        results = []
        for pred,prob in zip(predictions, probabilities):
            results.append({
                "prediction": int(pred),
                "probability": {
                    "no_diabetes": prob[0],
                    "diabetes": prob[1]
                }
                
            })

        return jsonify({"results": results}), 200
    
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


@app.route('/status', methods=['GET'])
@jwt_required()
def status():
    return jsonify({"status": "API is running"}), 200


@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json(silent=True)

    username = data.get("username", None)
    password = data.get("password", None)
    
    if data is None or not username or not password:
        return jsonify({"error": "Missing username or password"}), 400
    
    if username not in USERS or USERS[username] != password:
        return jsonify({"error": "Bad username or password"}), 401
    
    if username in USERS and USERS[username] == password:
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token), 200

if __name__ == '__main__':
    app.run(debug=True, port=8000)
