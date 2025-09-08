from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load saved model and scaler
model = joblib.load("best_random_forest.pkl")
scaler = joblib.load("scaler.pkl")

# Home page (no old results)
@app.route("/")
def home():
    return render_template("index.html", prediction=None, probability=None, risk=None)

# Prediction route with risk level
@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    features = [float(request.form[field]) for field in [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]]
    
    # Convert to numpy array and scale
    final_features = np.array([features])
    final_scaled = scaler.transform(final_features)
    
    # Make prediction
    pred = model.predict(final_scaled)[0]
    prob = model.predict_proba(final_scaled)[0][1]
    
    # Risk level
    if prob < 0.25:
        risk = "Low Risk"
    elif prob < 0.65:
        risk = "Medium Risk"
    else:
        risk = "High Risk"
    
    result = "Disease" if pred == 1 else "No Disease"
    probability = round(prob * 100, 2)  # Convert to percentage
    
    return render_template(
        "index.html",
        prediction=result,
        probability=f"{probability}%",  # Pass as string with %
        risk=risk
    )

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
