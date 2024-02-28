import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the cleaned dataset
cleaned_data = pd.read_csv("cleaned_data.csv")

# Split features and target variable
X = cleaned_data.drop('flag', axis=1)  # Exclude 'flag' column as the target variable
y = cleaned_data['flag']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Save the trained model
import joblib
joblib.dump(rf_classifier, 'trained_anomaly_model.pkl')
