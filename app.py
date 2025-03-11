import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("classes2.h5")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.json['sequence']
        
        # Convert to numpy array
        sequence = np.array(data, dtype=np.float32)
        
        # Ensure correct shape (1, 30, 1662)
        if sequence.shape != (30, 1662):
            return jsonify({"error": "Invalid input shape, expected (30, 1662)"}), 400
        
        # Reshape for model prediction
        sequence = np.expand_dims(sequence, axis=0)
        
        # Make prediction
        res = model.predict(sequence)[0]
        
        return jsonify({"prediction": res.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
