from flask import Flask, request, jsonify
import pickle
import datetime
import logging
import os

base_path = os.path.dirname(__file__)  # directory of the script
model_path = os.path.join(base_path, "models", "model_v1.pkl")

with open(model_path, "rb") as f:
    vectorizer, model = pickle.load(f)

logging.basicConfig(filename='monitoring.log', level=logging.INFO)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    msg = data.get("message", "")

    if not msg:
        return jsonify({"error": "No message provided"}), 400

    vec = vectorizer.transform([msg])
    pred = model.predict(vec)[0]

    timestamp = datetime.datetime.now().isoformat()
    logging.info(f"{timestamp} | input: {msg} | prediction: {pred}")

    return jsonify({"category": pred})

if __name__ == "__main__":
    app.run(debug=True)
