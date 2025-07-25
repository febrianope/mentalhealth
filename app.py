from flask import Flask, jsonify
import requests
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# --- KONFIGURASI ---
THINGER_USERNAME = 'Febrianope'
DEVICE_ID = 'esp32bri'
BUCKET_NAME = 'mental_monitoring'
ACCESS_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJoZWFsdGhfZGF0YSIsInN2ciI6ImFwLXNvdXRoZWFzdC5hd3MudGhpbmdlci5pbyIsInVzciI6IkZlYnJpYW5vcGUifQ.QUMBN9Q7RYXN3eXbhwOQcUrXgSV11Stkqb6nE366rmw'

# Load model & scaler sekali saat start
scaler = joblib.load("scaler.pkl")
model = joblib.load("model_svm.pkl")

@app.route("/")
def home():
    return "🎯 API Prediksi Kelelahan Mental aktif"

@app.route("/predict", methods=["GET"])
def predict():
    # Ambil data dari bucket (bukan device resource)
    bucket_url = f"https://api.thinger.io/v3/users/{THINGER_USERNAME}/buckets/{BUCKET_NAME}/data?limit=1"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    
    response = requests.get(bucket_url, headers=headers)
    
    if response.status_code != 200:
        return jsonify({"error": "❌ Gagal ambil data", "status": response.status_code}), response.status_code
    
    data = response.json()
    if not data:
        return jsonify({"error": "❌ Tidak ada data di bucket"}), 404
    
    latest_data = data[0]
    
    try:
        # Siapkan features untuk prediksi
        features = np.array([[
            latest_data['obj_temp'],
            latest_data['spo2'],
            latest_data['gsr_percent'],
            latest_data['systolic'],
            latest_data['diastolic'],
            latest_data['heart_rate']
        ]])
        
        # Normalisasi dan prediksi
        scaled = scaler.transform(features)
        pred = int(model.predict(scaled)[0])
        
        # Kirim hasil prediksi ke Thinger device resource
        device_url = f"https://api.thinger.io/v3/users/{THINGER_USERNAME}/devices/{DEVICE_ID}/resources/predict_output"
        result_data = {"in": pred}
        
        post_response = requests.post(device_url, headers=headers, json=result_data)
        
        if post_response.status_code == 200:
            return jsonify({"success": True, "prediksi": pred, "data": latest_data})
        else:
            return jsonify({"error": "❌ Gagal kirim ke Thinger", "status": post_response.status_code}), 500
            
    except Exception as e:
        return jsonify({"error": "❌ Gagal proses prediksi", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
