import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv("dataset/medium_fraud_transactions.csv")

# Add synthetic extreme values
extra_data = pd.DataFrame({"Amount": [0.01, 2000, 5000], "Class": [0, 1, 1]})
df = pd.concat([df, extra_data], ignore_index=True)

# Define features and target
X = df[["Amount"]]  # Using only Amount
y = df["Class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
pickle.dump(model, open("models/fraud_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
