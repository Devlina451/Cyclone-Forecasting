# Cyclone-Forecasting
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load data from CSV
data = pd.read_csv("storms.csv")  # Replace with your file name if different

# Select only the columns you want to use for prediction
data = data[['long', 'lat', 'hour', 'wind', 'pressure', 'status']]

# Separate numeric and categorical columns
numeric_columns = ['long', 'lat', 'hour', 'wind', 'pressure']
categorical_columns = ['status']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Convert categorical features to numerical values (e.g., 'status')
data['status'] = label_encoder.fit_transform(data['status'])  # Encode 'status' column

# Handle missing values for numeric columns using imputation
imputer = SimpleImputer(strategy='mean')  # You can also try 'median' or 'most_frequent'
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

# Define features (X) and target (y)
X = data[['long', 'lat', 'hour', 'wind', 'pressure']]  # Select specific columns for features
y = data['status']  # Target column
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize and train the Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Example data for prediction
exp_data = [[0, 1, 1, 0, 1]]  # This represents one new data point with values for 'long', 'lat', 'hour', 'wind', and 'pressure'

# Make prediction (still in encoded form)
predicted_label = model.predict(exp_data)
# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Example data for prediction
exp_data = [[0, 1, 1, 0, 1]]  # This represents one new data point with values for 'long', 'lat', 'hour', 'wind', and 'pressure'

# Make prediction (still in encoded form)
predicted_label = model.predict(exp_data)

# Convert the encoded prediction back to the original label
predicted_status = label_encoder.inverse_transform(predicted_label)

# Output the predicted status in its original form
print(f"Predicted status: {predicted_status[0]}
