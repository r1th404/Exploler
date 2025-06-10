from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from scipy.spatial.distance import cdist
from flask import send_from_directory

app = Flask(__name__)

# Load model GLVQ
model = joblib.load("C:/Users/naufa/OneDrive/Desktop/SEMESTER 4/eksoplanet-app/model/glvq_model_balance.pkl")
print("Model loaded:", model)

# Fungsi prediksi GLVQ
def predict_glvq(prototypes, labels, X):
    distances = cdist(X, prototypes)
    nearest = np.argmin(distances, axis=1)
    return labels[nearest]

# Halaman utama
@app.route("/")
def home():
    return render_template("index.html")


# Endpoint prediksi
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        # Pastikan semua fitur ada
        required_fields = ["koi_score", "koi_period", "koi_prad", "koi_teq", "koi_insol",
                           "koi_steff", "koi_srad", "koi_slogg", "koi_kepmag"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Field '{field}' tidak ditemukan."}), 400

        # Ubah ke array NumPy
        features = np.array([
            float(data.get("koi_period")),
            float(data.get("koi_prad")),
            float(data.get("koi_teq")),
            float(data.get("koi_insol")),
            float(data.get("koi_steff")),
            float(data.get("koi_kepmag")),
            float(data.get("koi_slogg")),
            float(data.get("koi_score")),
            float(data.get("koi_srad"))
        ]).reshape(1, -1)

        # Prediksi menggunakan GLVQ
        prototypes = model['prototypes']
        proto_labels = model['labels']
        prediction = predict_glvq(prototypes, proto_labels, features)[0]

        # Mapping label
        if prediction == 0:
            result_label = "🚫 Bukan Exoplanet"
        elif prediction == 2:
            result_label = "🪐 Exoplanet"
        else:
            result_label = f"⚠️ Tidak diketahui (kode: {prediction})"

        return jsonify({"result": result_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Jalankan aplikasi
if __name__ == "__main__":
    app.run(debug=True)