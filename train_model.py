from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
saved_data = joblib.load("model.pkl")
model = saved_data["model"]
target_names = saved_data["target_names"]


@app.route("/")
def home():
    return """
    <h1>Welcome to PetalPulse</h1>
    <p>Iris Flower Prediction System</p>
    <p>Use the /health route to check application status.</p>
    """


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "message": "PetalPulse is running successfully"
    })


@app.route("/prediction", methods=["POST"])
def prediction():
    try:
        data = request.get_json()

        features = [
            data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"]
        ]

        if not all(np.isfinite(float(value)) for value in features):
            return jsonify({"error": "Inputs must be finite numbers"}), 400

        result = model.predict([features])[0]
        flower = target_names[result]

        return jsonify({
            "prediction": flower,
            "message": "Flower predicted successfully"
        })

    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Please provide all four valid numeric measurements"
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)