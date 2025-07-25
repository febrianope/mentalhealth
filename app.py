from flask import Flask, jsonify
import requests
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# --- KONFIGURASI ---
THINGER_USERNAME = 'Febrianope'
DEVICE_ID = 'esp32bri'
ACCESS_TOKEN = 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJoZWFsdGhfZGF0YSIsInN2ciI6ImFwLXNvdXRoZWFzdC5hd3MudGhpbmdlci5pbyIsInVzciI6IkZlYnJpYW5vcGUifQ.QUMBN9Q7RYXN3eXbhwOQcUrXgSV11Stkqb6nE366rmw'  # Ganti token kamu lengkap di sini!

# Load model & scaler sekali saat start
scaler = joblib.load("scaler.pkl")
model = joblib.load("model_svm.pkl")

@app.route("/")
def home():
    return "🎯 API Prediksi Kelelahan Mental aktif"

@app.route("/predict", methods=["GET"])
def predict():
    url = f"https://api.thinger.io/v3/users/{THINGER_USERNAME}/devices/{DEVICE_ID}/resources/mental_monitoring"
    headers = {"Authorization": ACCESS_TOKEN}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return jsonify({"error": "❌ Gagal ambil data", "status": response.status_code}), response.status_code

    data = response.json()

    try:
        df = pd.DataFrame([data], columns=[
            "obj_temp", "spo2", "gsr_percent", "systolic", "diastolic", "heart_rate"
        ])

        scaled = scaler.transform(df)
        pred = int(model.predict(scaled)[0])

        # Kirim hasil prediksi ke Thinger
        post_url = f"https://api.thinger.io/v3/users/{THINGER_USERNAME}/devices/{DEVICE_ID}/resources/predict_output"
        post = requests.post(post_url, headers=headers, json=pred)

        if post.status_code == 200:
            return jsonify({"success": True, "prediksi": pred})
        else:
            return jsonify({"error": "❌ Gagal kirim ke Thinger", "status": post.status_code}), 500

    except Exception as e:
        return jsonify({"error": "❌ Gagal proses prediksi", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
