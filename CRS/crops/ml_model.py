import joblib
import os
import numpy as np

BASE_DIR = os.path.dirname(__file__)

def load_artifact(filename, default=None):
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return default

model = load_artifact("crop_model.pkl")
scaler = load_artifact("scaler.pkl")
FEATURES = load_artifact("features.pkl", ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

CROP_INFO = {
    "Rice":      {"emoji": "🌾", "season": "Kharif", "color": "#10b981"},
    "Wheat":     {"emoji": "🌿", "season": "Rabi",   "color": "#d4a843"},
    "Maize":     {"emoji": "🌽", "season": "Kharif", "color": "#f59e0b"},
    "Cotton":    {"emoji": "🌸", "season": "Kharif", "color": "#ec4899"},
    "Sugarcane": {"emoji": "🎋", "season": "Annual", "color": "#8b5cf6"},
    "Banana":    {"emoji": "🍌", "season": "Annual", "color": "#eab308"},
    "Groundnut": {"emoji": "🥜", "season": "Kharif", "color": "#a855f7"},
    "Millets":   {"emoji": "🌾", "season": "Kharif", "color": "#64748b"},
    "Soybean":   {"emoji": "🫘", "season": "Kharif", "color": "#22c55e"},
    "Tomato":    {"emoji": "🍅", "season": "Annual", "color": "#ef4444"},
}

def predict_crop(input_dict):
    if model is None or scaler is None:
        return {"success": False, "error": "Model not trained. Run 'python crops/train_model.py' first."}

    try:
        values = [float(input_dict.get(f, 0)) for f in FEATURES]
        
        if len(values) != len(FEATURES):
            raise ValueError(f"Expected {len(FEATURES)} features, got {len(values)}")

        
        scaled_input = scaler.transform([values])
        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]
        
        
        classes = model.classes_
        top_indices = np.argsort(probabilities)[::-1][:3]
        
        recommendations = []
        for i in top_indices:
            crop_name = classes[i]
            conf = probabilities[i] * 100
            info = CROP_INFO.get(crop_name, {"emoji": "🌱", "season": "Unknown", "color": "#10b981"})
            recommendations.append({
                "crop": crop_name,
                "confidence": round(conf, 1),
                "emoji": info["emoji"],
                "season": info["season"],
                "color": info["color"]
            })

        return {
            "success": True,
            "prediction": prediction,
            "confidence": round(max(probabilities) * 100, 1),
            "recommendations": recommendations,
            "input_values": dict(zip(FEATURES, values))
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# FERTILIZER PREDICTION
# ============================================================

fertilizer_model = load_artifact("fertilizer_model.pkl")
fertilizer_scaler = load_artifact("fertilizer_scaler.pkl")
soil_encoder = load_artifact("soil_encoder.pkl")
crop_type_encoder = load_artifact("crop_type_encoder.pkl")
FERTILIZER_FEATURES = load_artifact(
    "fertilizer_features.pkl",
    ['N', 'P', 'K', 'temperature', 'humidity', 'moisture', 'soil_type', 'crop_type']
)

FERTILIZER_INFO = {
    "Urea":     {"emoji": "🧪", "npk": "46-0-0",   "color": "#10b981"},
    "DAP":      {"emoji": "🌱", "npk": "18-46-0",  "color": "#f59e0b"},
    "14-35-14": {"emoji": "🌿", "npk": "14-35-14", "color": "#8b5cf6"},
    "28-28":    {"emoji": "🌾", "npk": "28-28-0",  "color": "#22c55e"},
    "17-17-17": {"emoji": "⚖️", "npk": "17-17-17", "color": "#3b82f6"},
    "20-20":    {"emoji": "🌻", "npk": "20-20-0",  "color": "#eab308"},
    "10-26-26": {"emoji": "🍅", "npk": "10-26-26", "color": "#ef4444"},
}

def predict_fertilizer(input_dict):
    if fertilizer_model is None or fertilizer_scaler is None:
        return {"success": False, "error": "Fertilizer model not trained. Run 'python crops/train_fertilizer_model.py' first."}

    try:
        values = []
        for f in FERTILIZER_FEATURES:
            if f == "soil_type":
                if soil_encoder is None:
                    raise ValueError("Soil type encoder not found.")
                values.append(soil_encoder.transform([input_dict.get(f)])[0])
            elif f == "crop_type":
                if crop_type_encoder is None:
                    raise ValueError("Crop type encoder not found.")
                values.append(crop_type_encoder.transform([input_dict.get(f)])[0])
            else:
                values.append(float(input_dict.get(f, 0)))

        if len(values) != len(FERTILIZER_FEATURES):
            raise ValueError(f"Expected {len(FERTILIZER_FEATURES)} features, got {len(values)}")

        scaled_input = fertilizer_scaler.transform([values])
        prediction = fertilizer_model.predict(scaled_input)[0]
        probabilities = fertilizer_model.predict_proba(scaled_input)[0]

        classes = fertilizer_model.classes_
        top_indices = np.argsort(probabilities)[::-1][:3]

        recommendations = []
        for i in top_indices:
            fert_name = classes[i]
            conf = probabilities[i] * 100
            info = FERTILIZER_INFO.get(fert_name, {"emoji": "🧴", "npk": "Unknown", "color": "#10b981"})
            recommendations.append({
                "fertilizer": fert_name,
                "confidence": round(conf, 1),
                "emoji": info["emoji"],
                "npk": info["npk"],
                "color": info["color"]
            })

        return {
            "success": True,
            "prediction": prediction,
            "confidence": round(max(probabilities) * 100, 1),
            "recommendations": recommendations,
            "input_values": dict(zip(FERTILIZER_FEATURES, values))
        }

    except Exception as e:
        return {"success": False, "error": str(e)}