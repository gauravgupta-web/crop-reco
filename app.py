from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print("Warning: Model not found. Run train.py first.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded. Please train the model.'}), 500
        
    try:
        data = request.json
        # Extract features
        N = float(data.get('N', 0))
        P = float(data.get('P', 0))
        K = float(data.get('K', 0))
        temp = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        ph = float(data.get('ph', 0))
        rainfall = float(data.get('rainfall', 0))
        
        # Format the features array using numpy (since model expects a 2D array)
        features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        
        # Predict the crop
        prediction = model.predict(features)[0]
        
        return jsonify({
            'success': True,
            'prediction': prediction.capitalize()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
