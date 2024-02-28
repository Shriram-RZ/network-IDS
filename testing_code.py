import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model
rf_classifier = joblib.load('trained_anomaly_model.pkl')

# Load the cleaned dataset
cleaned_data = pd.read_csv("cleaned_data.csv")

# Split features and target variable
X = cleaned_data.drop('flag', axis=1)  # Exclude 'flag' column as the target variable
y = cleaned_data['flag']

# Make predictions
y_pred = rf_classifier.predict(X)

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

# Generate a classification report
report = classification_report(y, y_pred)
print("Classification Report:")
print(report)
