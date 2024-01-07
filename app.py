import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model from the file
model_filename = 'linear_regression_model.pkl'
with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get the requested hour from the query parameters
        hour = int(request.args.get('hour'))
        
        # Ensure the hour is within a valid range (9-20)
        if 9 <= hour <= 21:
            # Make a prediction using the loaded model
            prediction = loaded_model.predict([[hour]])[0]
            return jsonify({'hour': hour, 'predicted_count': prediction})
        else:
            return jsonify({'error': 'Invalid hour. Please provide a value between 9 and 20.'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid input. Please provide a valid integer hour.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
