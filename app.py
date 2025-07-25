from flask import Flask, jsonify
import requests
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# --- GANTI DENGAN PUNYAMU SENDIRI ---
THINGER_USERNAME = 'Febrianope'
DEVICE_ID = 'esp32bri'
ACCESS_TOKEN = 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJoZWFsdGhfZGF0YSIsInN2ciI6ImFwLXNvdXRoZWFzdC5hd3MudGhpbmdlci5pbyIsInVzciI6IkZlYnJpYW5vcGUifQ.QUMBN9Q7RYXN3eXbhwOQcUrXgSV11Stkqb6nE366rmw'  # include Bearer!

# Load sekali saja saat app start
scaler = joblib.load("scaler.pkl")
model = joblib.load("model_svm.pkl")

@app.route("/")
def home():
    return "API aktif di Heroku 🚀"

@app.route("/predict", methods=["GET"])
def predict():
    url = f"https://api.thinger.io/v3/users/{THINGER_USERNAME}/devices/{DEVICE_ID}/resources/mental_monitoring"
    headers = {"Authorization": ACCESS_TOKEN}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return jsonify({"error": "Gagal ambil data", "status": response.status_code}), response.status_code

    data = response.json()
    try:
        df = pd.DataFrame([data], col
