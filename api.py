from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from datetime import datetime
import joblib
import firebase_admin
from firebase_admin import credentials, firestore

#Initialize FastAPI and Firebase
app = FastAPI(title="Water Consumption API")

#Request Model
class HardwareData(BaseModel):
    meter_id: str
    plot_id: str = "UNKNOWN"
    totalLiters: float
    remainingUnits: float
    sourceLiters: float
    leak: bool

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

#Constants for prediction calculations
WATER_TARIFF_PER_LITER = 0.5  # KSh per liter (20L = 10 KSh in Nairobi plots)
AVG_COMMUNITY_HOURLY_USE = 500
API_KEY = "demo-secret-key"  

#Load the ML Model
try:
    consumption_model = joblib.load("models/water_model_production_final.pkl")
    print("ML model loaded successfully")
except Exception as e:
    consumption_model = None
    print(f"ML model not loaded: {e}, using rule-based fallback")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "ml_loaded": consumption_model is not None,
        "timestamp": datetime.utcnow()
    }

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

#Utility Functions
def normalize_hardware_data(data: dict) -> dict:
    return {
        "meter_id": data.get("meter_id"),
        "plot_id": data.get("plot_id", "UNKNOWN"),
        "total_liters": float(data.get("totalLiters", 0)),
        "remaining_units": float(data.get("remainingUnits", 0)),
        "source_liters": float(data.get("sourceLiters", 0)),
        "leak_flag": bool(data.get("leak", False)),
        "timestamp": datetime.utcnow()
    }

def classify_consumption_risk(total_liters: float) -> str:
    #replace with ML output if model exists
    if consumption_model:
        try:
            prediction = consumption_model.predict([[total_liters]])[0]
            return prediction
        except Exception:
            pass

    #rule-based fallback based on Kenyan consumption patterns
    if total_liters > 200:  # Above WASREB lifeline (200L/day household)
        return "High"
    elif total_liters > 120:  # Above urban average (120L/person)
        return "Medium"
    return "Low"  # Normal Kenyan usage (40-120L range)

def detect_leak(leak_flag: bool, total_liters: float) -> bool:
    return leak_flag or total_liters > 400  # Leak if usage exceeds reasonable maximum

def calculate_revenue(total_liters: float) -> float:
    return round(total_liters * WATER_TARIFF_PER_LITER, 2)

def estimate_sustainability(remaining_units: float) -> float:
    if AVG_COMMUNITY_HOURLY_USE == 0:
        return 0
    hours_left = remaining_units / AVG_COMMUNITY_HOURLY_USE
    return round(hours_left, 2)

#Main Automation Endpoint
@app.post("/api/hardware-data")
def receive_hardware_data(
    data: HardwareData,
    x_api_key: str = Header(...)
):
    """
    MVP FLOW:
    1. Receive hardware data
    2. Store raw data
    3. Normalize data
    4. Generate predictions
    5. Store predictions
    """
    
    verify_api_key(x_api_key)

    #Store RAW hardware data
    db.collection("raw_hardware_data").add({
        **data.dict(),
        "received_at": datetime.utcnow()
    })

    #Normalize
    normalized = normalize_hardware_data(data.dict())

    db.collection("water_readings").add(normalized)

    #Predictions
    consumption_risk = classify_consumption_risk(normalized["total_liters"])
    leak_detected = detect_leak(
        normalized["leak_flag"],
        normalized["total_liters"]
    )
    revenue = calculate_revenue(normalized["total_liters"])
    sustainability_hours = estimate_sustainability(
        normalized["remaining_units"]
    )

    prediction_result = {
        "meter_id": normalized["meter_id"],
        "plot_id": normalized["plot_id"],
        "consumption_risk": consumption_risk,
        "leak_detected": leak_detected,
        "predicted_revenue": revenue,
        "sustainability_hours_left": sustainability_hours,
        "generated_at": datetime.utcnow()
    }

    #Store predictions
    db.collection("ml_predictions").add(prediction_result)

    #Return response
    return {
        "status": "success",
        "predictions": prediction_result
    }