import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("models/fraud_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

def preprocess_and_predict(data):
    """
    Preprocess input transaction data and predict fraudulence.
    :param data: List of input features (e.g., [Amount])
    :return: Prediction result (0 = Legitimate, 1 = Fraud)
    """
    # Convert to numpy array and reshape
    data = np.array(data).reshape(1, -1)

    # Ensure feature names are not an issue
    data_scaled = scaler.transform(data)

    # Predict fraudulence
    prediction = model.predict(data_scaled)[0]
    return prediction
