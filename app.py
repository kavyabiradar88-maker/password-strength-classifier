from flask import Flask, render_template, request
import joblib
import numpy as np
import re
from scipy.sparse import hstack

app = Flask(__name__)

# Load model & vectorizer
model = joblib.load("../models/high_lr.pkl")
vectorizer = joblib.load("../models/high_lr_vectorizer.pkl")

# Feature extraction
def extract_features(password):
    length = len(password)
    digits = len(re.findall(r"\d", password))
    upper = len(re.findall(r"[A-Z]", password))
    lower = len(re.findall(r"[a-z]", password))
    symbols = len(re.findall(r"[^a-zA-Z0-9]", password))
    return np.array([[length, digits, upper, lower, symbols]])

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    
    if request.method == "POST":
        pwd = request.form["password"]

        # TF-IDF + numeric features
        vectorized = vectorizer.transform([pwd])
        numeric = extract_features(pwd)
        X_input = hstack([vectorized, numeric])

        pred = model.predict(X_input)[0]

        if pred == 0:
            result = "WEAK"
        elif pred == 1:
            result = "MEDIUM"
        else:
            result = "STRONG"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
