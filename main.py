from math import expm1
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model("price_prediction_model.h5", compile = False)
# model = keras.models.load_model("datamodel/price_prediction_model.h5")
transformer = joblib.load("data_transformer.joblib")

@app.route("/liveness_check")
def liveness():
    return "ok", 200

@app.route("/predict", methods=["POST"])
def index():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(transformer.transform(df))
    predicted_price = expm1(prediction.flatten()[0])
    return jsonify({"price": str(predicted_price)})

if __name__ == "__main__":
    app.run(debug=True)
