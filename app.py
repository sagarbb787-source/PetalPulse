from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
saved_data = joblib.load("model.pkl")
model = saved_data["model"]
target_names = saved_data["target_names"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "message": "PetalPulse is running successfully"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        features = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        prediction = model.predict(features)[0]
        flower = target_names[int(prediction)]

        return f"""
        <h1>🌸 Prediction Result</h1>
        <h2>Flower: {flower}</h2>
        <a href="/">Go Back</a>
        """

    except (KeyError, ValueError):
        return "Please enter all four valid measurements.", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)