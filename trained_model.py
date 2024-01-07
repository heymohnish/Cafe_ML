import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load and prepare data from a CSV file (replace 'your_data.csv' with your actual file)
df = pd.read_csv('C:\\Users\\mohnish.varatharajan\\source\\hello_flask\\Untitled spreadsheet - Sheet1.csv')

# Check for missing values in the target variable
if df['Number_of_People'].isnull().any():
    print("Handling missing values in the target variable.")
    
    # Remove rows with missing values in the target variable
    df = df.dropna(subset=['Number_of_People'])
    print(f"Number of rows after handling missing values: {len(df)}")

# Choose a Model
model = LinearRegression()

# Extract hour from the 'Date' column
df['Hour'] = pd.to_datetime(df['Date']).dt.hour

# Prepare the input features (X) and target variable (y)
X = df[['Hour']]
y = df['Number_of_People']

# Print some information about the dataset
print(f"Number of samples in X: {X.shape[0]}")
print(f"Number of samples in y: {len(y)}")

# Train the Model
model.fit(X, y)

# Save the trained model using pickle
with open('trained_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved successfully.")
