from flask import Flask, jsonify
import requests
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# --- KONFIGURASI ---
THINGER_USERNAME = 'Febrianope'
DEVICE_ID = 'esp32bri'
BUCKET_NAME = 'mental_monitoring'
ACCESS_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJoZWFsdGhfZGF0YSIsInN2ciI6ImFwLXNvdXRoZWFzdC5hd3MudGhpbmdlci5pbyIsInVzciI6IkZlYnJpYW5vcGUifQ.QUMBN9Q7RYXN3eXbhwOQcUrXgSV11Stkqb6nE366rmw'

# Load model & scaler dengan error handling
try:
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model_svm.pkl")
    print("✅ Model dan scaler berhasil dimuat")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    scaler = None
    model = None

@app.route("/")
def home():
    return jsonify({
        "status": "🎯 API Prediksi Kelelahan Mental aktif",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    })

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "timestamp": "2024"})

@app.route("/predict", methods=["GET"])
def predict():
    print("📞 Predict endpoint dipanggil")
    
    if model is None or scaler is None:
        return jsonify({"error": "❌ Model belum dimuat"}), 500
    
    # Ambil data dari bucket
    bucket_url = f"https://api.thinger.io/v3/users/{THINGER_USERNAME}/buckets/{BUCKET_NAME}/data?limit=1"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    
    print(f"🔗 Mengambil data dari: {bucket_url}")
    
    try:
        response = requests.get(bucket_url, headers=headers, timeout=10)
        print(f"📡 Thinger response: {response.status_code}")
        
        if response.status_code != 200:
            return jsonify({"error": "❌ Gagal ambil data", "status": response.status_code}), response.status_code
        
        data = response.json()
        print(f"📊 Data received: {data}")
        
        if not data:
            return jsonify({"error": "❌ Tidak ada data di bucket"}), 404
        
        latest_data = data[0]
        print(f"📋 Processing data: {latest_data}")
        
        # Siapkan features untuk prediksi
        features = np.array([[
            latest_data['obj_temp'],
            latest_data['spo2'],
            latest_data['gsr_percent'],
            latest_data['systolic'],
            latest_data['diastolic'],
            latest_data['heart_rate']
        ]])
        
        print(f"🔢 Features: {features}")
        
        # Normalisasi dan prediksi
        scaled = scaler.transform(features)
        pred = int(model.predict(scaled)[0])
        
        print(f"🧠 Prediksi: {pred}")
        
        # Kirim hasil prediksi ke Thinger device resource
        device_url = f"https://api.thinger.io/v3/users/{THINGER_USERNAME}/devices/{DEVICE_ID}/resources/predict_output"
        result_data = {"in": pred}
        
        print(f"📤 Kirim ke device: {device_url}")
        print(f"📦 Data: {result_data}")
        
        post_response = requests.post(device_url, headers=headers, json=result_data, timeout=10)
        print(f"📬 Device response: {post_response.status_code}")
        
        if post_response.status_code == 200:
            return jsonify({
                "success": True, 
                "prediksi": pred, 
                "data": latest_data,
                "message": "✅ Prediksi berhasil dikirim"
            })
        else:
            return jsonify({
                "error": "❌ Gagal kirim ke Thinger", 
                "status": post_response.status_code,
                "response": post_response.text
            }), 500
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": "❌ Gagal proses prediksi", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
