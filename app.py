from flask import Flask, render_template, request
from preprocess import preprocess_and_predict
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
import io

# Initialize Flask app
app = Flask(__name__)

# 🔹 Load the dataset (Ensure the correct path)
df = pd.read_csv("E:\\CreditCardFraudDetection\\dataset\\medium_fraud_transactions.csv")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    img_data = None  # Variable to store graph image

    if request.method == "POST":
        try:
            # Get transaction amount from form
            amount = float(request.form["Amount"])

            # Predict fraudulence using only amount
            prediction = preprocess_and_predict([amount])
            result = "Fraudulent" if prediction == 1 else "Legitimate"

            # 🔹 Generate histogram and encode as base64 image
            img = io.BytesIO()
            plt.figure(figsize=(8, 4))
            plt.hist(df[df['Class'] == 0]['Amount'], bins=50, alpha=0.5, label='Legitimate', color='blue')
            plt.hist(df[df['Class'] == 1]['Amount'], bins=50, alpha=0.5, label='Fraudulent', color='red')
            plt.axvline(amount, color='black', linestyle='dashed', linewidth=2, label="Input Amount")
            plt.legend()
            plt.xlabel('Transaction Amount')
            plt.ylabel('Frequency')
            plt.title('Transaction Amount Distribution')
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            img_data = base64.b64encode(img.getvalue()).decode()

        except Exception as e:
            error = str(e)

    return render_template("index.html", result=result, error=error, img_data=img_data)

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Change port if needed
