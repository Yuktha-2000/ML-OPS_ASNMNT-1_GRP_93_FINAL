from flask import Flask, request, jsonify
import joblib
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = joblib.load('models/best_model.joblib')
        # Ensure valid JSON input
        if not request.json or 'features' not in request.json:
            return jsonify({"error": "Invalid input. Provide a JSON with 'features' key."}), 400
        
        # Extract features
        data = request.json['features']
        data = np.array(data).reshape(1, -1)  # Reshape to match scikit-learn input requirements
        
        # Make prediction
        prediction = model.predict(data)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run_app():
    app.run(debug=True, host='0.0.0.0', port=5000)