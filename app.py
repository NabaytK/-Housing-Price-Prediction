# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("best_housing_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = preprocess_input(data)  # Create your preprocessing function
    prediction_log = model.predict([features])
    return jsonify({
        'prediction': float(np.expm1(prediction_log[0]) )
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
