from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, root_validator
from datetime import datetime
import joblib
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

#Initialize FastAPI and Firebase
app = FastAPI(title="Water Consumption API")
raw_hardware_watch = None
raw_listener_bootstrapped = False

#Request Model
class HardwareData(BaseModel):
    meterNumber: str
    plot_id: str = "UNKNOWN"
    totalLiters: float
    remainingUnits: float
    sourceLiters: float
    leak: bool

    @root_validator(pre=True)
    def support_legacy_meter_id(cls, values):
        # Backward compatibility for legacy payloads still sending meter_id.
        if "meterNumber" not in values and "meter_id" in values:
            values["meterNumber"] = values["meter_id"]
        return values

#Firebase initialization with environment variable fallback
try:
    if os.path.exists("serviceAccountKey.json"):
        cred = credentials.Certificate("serviceAccountKey.json")
    else:
        #Use environment variable for deployment
        firebase_config = os.getenv('FIREBASE_CONFIG')
        if firebase_config:
            config_dict = json.loads(firebase_config)
            cred = credentials.Certificate(config_dict)
        else:
            raise Exception("Firebase credentials not found")
    
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"Firebase initialization failed: {e}")
    db = None

#Constants for prediction calculations
#KSh per liter (20L = 10 KSh)
WATER_TARIFF_PER_LITER = 0.5  
REVENUE_CURRENCY = "KES"
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
        "meterNumber": data.get("meterNumber"),
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

    #rule-based fallback based on consumption patterns
    if total_liters > 200:  
        return "High"
    elif total_liters > 120: 
        return "Medium"
    return "Low"  

def detect_leak(leak_flag: bool, total_liters: float) -> bool:
    return leak_flag or total_liters > 400  

def calculate_revenue(total_liters: float) -> float:
    return round(total_liters * WATER_TARIFF_PER_LITER, 2)

def estimate_sustainability(remaining_units: float) -> float:
    if AVG_COMMUNITY_HOURLY_USE == 0:
        return 0
    hours_left = remaining_units / AVG_COMMUNITY_HOURLY_USE
    return round(hours_left, 2)


def generate_prediction_from_normalized(normalized: dict) -> dict:
    """Generate and persist prediction outputs for normalized hardware data."""
    consumption_risk = classify_consumption_risk(normalized["total_liters"])
    leak_detected = detect_leak(
        normalized["leak_flag"],
        normalized["total_liters"]
    )
    revenue = calculate_revenue(normalized["total_liters"])
    sustainability_hours = estimate_sustainability(
        normalized["remaining_units"]
    )
    person_name, household_id, user_matched = resolve_person_and_household(normalized["meterNumber"])

    prediction_result = {
        "meterNumber": normalized["meterNumber"],
        "plot_id": normalized["plot_id"],
        "personName": person_name,
        "householdId": household_id,
        "userMatched": user_matched,
        "consumption_risk": consumption_risk,
        "leak_detected": leak_detected,
        "predicted_revenue": revenue,
        "predicted_revenue_currency": REVENUE_CURRENCY,
        "sustainability_hours_left": sustainability_hours,
        "generated_at": datetime.utcnow()
    }

    db.collection("water_readings").add(normalized)
    db.collection("ml_predictions").add(prediction_result)
    return prediction_result


def process_raw_hardware_document(doc_snapshot):
    """Process one raw_hardware_data document into normalized data and predictions."""
    raw_data = doc_snapshot.to_dict() or {}

    # Skip docs already processed or intentionally handled by direct API ingestion.
    if raw_data.get("prediction_generated") or raw_data.get("skip_trigger_processing"):
        return

    # Support either meterNumber
    if "meterNumber" not in raw_data and "meter_id" in raw_data:
        raw_data["meterNumber"] = raw_data.get("meter_id")

    if not raw_data.get("meterNumber"):
        doc_snapshot.reference.update({
            "prediction_error": "Missing meterNumber in raw_hardware_data",
            "prediction_failed_at": datetime.utcnow(),
        })
        return

    try:
        normalized = normalize_hardware_data(raw_data)
        prediction_result = generate_prediction_from_normalized(normalized)
        doc_snapshot.reference.update({
            "prediction_generated": True,
            "prediction_generated_at": datetime.utcnow(),
            "prediction_meterNumber": prediction_result["meterNumber"],
        })
    except Exception as exc:
        doc_snapshot.reference.update({
            "prediction_error": str(exc),
            "prediction_failed_at": datetime.utcnow(),
        })


def raw_hardware_on_snapshot(col_snapshot, changes, read_time):
    """Firestore listener callback for new raw hardware documents."""
    global raw_listener_bootstrapped

    # The first snapshot includes existing docs; skip to avoid accidental backfill.
    if not raw_listener_bootstrapped:
        raw_listener_bootstrapped = True
        return

    for change in changes:
        if change.type.name != "ADDED":
            continue
        process_raw_hardware_document(change.document)


@app.on_event("startup")
def start_firestore_trigger_listener():
    """Start Firestore listener so direct raw_hardware_data writes are processed."""
    global raw_hardware_watch
    if db is None:
        return
    if raw_hardware_watch is None:
        raw_hardware_watch = db.collection("raw_hardware_data").on_snapshot(raw_hardware_on_snapshot)


@app.on_event("shutdown")
def stop_firestore_trigger_listener():
    """Stop Firestore listener on application shutdown."""
    global raw_hardware_watch
    if raw_hardware_watch is not None:
        raw_hardware_watch.unsubscribe()
        raw_hardware_watch = None


def resolve_person_and_household(meter_number: str):
    """
    Resolve person name and household ID from Firestore using meter number.
    Returns (person_name, household_id, user_matched).
    """
    default_name = "Unknown Person"
    default_household = None

    if db is None:
        return default_name, default_household, False

    candidate_collections = ["users", "households", "customers", "meter_profiles"]
    meter_fields = ["meterNumber", "meter_id", "meterId"]
    name_fields = ["personName", "name", "fullName", "full_name", "customerName", "residentName"]
    household_fields = ["householdId", "household_id", "houseId", "house_id"]

    for collection_name in candidate_collections:
        for meter_field in meter_fields:
            try:
                docs = (
                    db.collection(collection_name)
                    .where(meter_field, "==", meter_number)
                    .limit(1)
                    .stream()
                )
                for doc in docs:
                    payload = doc.to_dict() or {}

                    person_name = next(
                        (payload.get(field) for field in name_fields if payload.get(field)),
                        default_name,
                    )
                    household_id = next(
                        (payload.get(field) for field in household_fields if payload.get(field)),
                        default_household,
                    )
                    return person_name, household_id, True
            except Exception:
                # Keep API resilient if a collection or query path is unavailable.
                continue

    return default_name, default_household, False

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
        "received_at": datetime.utcnow(),
        "ingest_source": "api_endpoint",
        "skip_trigger_processing": True,
        "prediction_generated": True,
    })

    #Normalize
    normalized = normalize_hardware_data(data.dict())
    prediction_result = generate_prediction_from_normalized(normalized)

    #Return response
    return {
        "status": "success",
        "predictions": prediction_result
    }