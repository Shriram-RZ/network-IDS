import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("ids_dataset.csv")

# Drop duplicates
data.drop_duplicates(inplace=True)

# Encode categorical features
encoder = LabelEncoder()
data['protocol_type'] = encoder.fit_transform(data['protocol_type'])
data['service'] = encoder.fit_transform(data['service'])
data['flag'] = encoder.fit_transform(data['flag'])

# Save cleaned data
data.to_csv("cleaned_data.csv", index=False)
