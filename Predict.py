import csv
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)

# Load data from CSV file
csv_file_path = 'path/to/your/data.csv'
with open(csv_file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)
    data = [list(map(int, row)) for row in reader]

# Transpose the data to have hours as rows and days as columns
transposed_data = np.array(data).T

# Extract features (hour) and target (number of people) from the transposed data
hours = np.array(range(9, 21)).reshape(-1, 1)  # Assuming 09:00 to 20:00
counts = transposed_data.flatten()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hours, counts, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a file using pickle
model_filename = 'linear_regression_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)

# Load the trained model from the file
with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)
