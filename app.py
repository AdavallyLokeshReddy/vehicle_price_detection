import os
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model pipeline (make sure it includes preprocessing inside the pipeline)
model = pickle.load(open("vehicle_price_prediction.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form data
        input_data = {
            "make": request.form["make"],
            "model": request.form["model"],
            "year": int(request.form["year"]),
            "engine": request.form["engine"],
            "cylinders": int(request.form["cylinders"]),
            "fuel": request.form["fuel"],
            "mileage": float(request.form["mileage"]),
            "transmission": request.form["transmission"],
            "body": request.form["body"],
            "drivetrain": request.form["drivetrain"],
            "doors": int(request.form["doors"]),
            "vehicle_age": int(request.form["vehicle_age"]),
        }

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Model should include preprocessing steps (like LabelEncoding, OneHotEncoding, scaling, etc.)
        # If not, you need to replicate preprocessing steps here

        # Predict
        prediction = model.predict(df)[0]
        output = round(prediction, 2)

        return render_template("index.html", prediction_text=f"Estimated Vehicle Price: ₹ {output}")
    
    except ValueError as ve:
        return render_template("index.html", prediction_text=f"Input Error: {ve}")
    
    except KeyError as ke:
        return render_template("index.html", prediction_text=f"Missing Input Field: {ke}")
    
    except Exception as e:
        return render_template("index.html", prediction_text=f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
