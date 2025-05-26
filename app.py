from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load your saved model pipeline
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Extract form data
    data = {
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
    
    # Convert to DataFrame (single row)
    input_df = pd.DataFrame([data])
    
    # Predict price
    predicted_price = best_model.predict(input_df)[0]
    
    # Format price with commas and 2 decimals
    formatted_price = f"{predicted_price:,.2f}"
    
    return render_template("result.html", price=formatted_price)

if __name__ == "__main__":
    app.run(debug=True)
