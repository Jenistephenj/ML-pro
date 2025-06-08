import pandas as pd

# Load dataset
df = pd.read_csv("dataset/medium_fraud_transactions.csv")  

# Find the min & max transaction amount
min_amount = df["Amount"].min()
max_amount = df["Amount"].max()

print(f"Transaction Amount Range: ${min_amount} - ${max_amount}")

