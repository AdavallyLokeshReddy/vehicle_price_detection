import os
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model (make sure the file is in the same folder or provide relative path)
model = pickle.load(open('vehicle_price_prediction.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract inputs from the form
        year = int(request.form['year'])
        mileage = float(request.form['mileage'])
        # Add other inputs below if you have more features:
        # example: engine = float(request.form['engine'])
        # example: horsepower = float(request.form['horsepower'])

        # Prepare the feature array (update based on your actual features)
        features = np.array([[year, mileage]])

        # Make prediction
        prediction = model.predict(features)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Predicted price is ₹ {output}')
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
