from fastapi import FastAPI, HTTPException
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from collections import defaultdict
import joblib
import pickle
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
import logging
import threading

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Firebase ──────────────────────────────────────────────────────────────────
try:
    if os.path.exists("serviceAccountKey.json"):
        cred = credentials.Certificate("serviceAccountKey.json")
    else:
        firebase_config = os.getenv("FIREBASE_CONFIG")
        if firebase_config:
            cred = credentials.Certificate(json.loads(firebase_config))
        else:
            raise Exception("Firebase credentials not found")

    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logger.info("Firebase initialized successfully")
except Exception as e:
    logger.error(f"Firebase initialization failed: {e}")
    db = None

# ── Constants ─────────────────────────────────────────────────────────────────
WATER_TARIFF_PER_LITER = 0.5   # KES per liter (10 KES per 20L)
REVENUE_CURRENCY       = "KES"

# ── The 18 features the model was trained on (order matters for scaling) ──────
MODEL_FEATURE_COLUMNS = [
    "adult_count",
    "child_count",
    "rooms_in_hh",
    "years_in_community",
    "primary_dw_source",
    "daily_hh_water_cost_for_pay_to_fetch",
    "water_storage_drinking_water",
    "coli_mpn",
    "tc_mpn",
    "tap_closure_days_per_week",
    "estimated_storage_capacity_liters",
    "total_people",
    "people_per_room",
    "consumption_risk_score",
    "water_access_score",
    "estimated_non_dw_storage_capacity",
    "estimated_stored_non_dw",
    "avg_price_per_liter_cedis",
]

# ── Load ML model ─────────────────────────────────────────────────────────────
try:
    consumption_model = joblib.load("models/water_model_production_final.pkl")
    logger.info("ML model loaded successfully")
except Exception as e:
    consumption_model = None
    logger.warning(f"ML model not loaded: {e}")

# ── Load scaler ───────────────────────────────────────────────────────────────
try:
    with open("models/model_config_final.pkl", "rb") as f:
        model_config = pickle.load(f)
    model_scaler = model_config.get("scaler")
    logger.info("Model scaler loaded successfully")
except Exception as e:
    model_scaler = None
    logger.warning(f"Model scaler not loaded: {e}")

# ── Listener state ────────────────────────────────────────────────────────────
water_usage_watch     = None
listener_bootstrapped = False
listener_lock         = threading.Lock()

# ── Community average features — computed once at startup ─────────────────────
# Used as fallback when a meter has no matching household_features document.
COMMUNITY_AVG_FEATURES: dict = {col: 0.0 for col in MODEL_FEATURE_COLUMNS}

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: compute community averages then start listener. Shutdown: clean up."""
    compute_community_avg_features()
    start_water_usage_listener()
    yield
    stop_water_usage_listener()

app = FastAPI(title="Water Consumption ML API", lifespan=lifespan)

# ── Community average computation ─────────────────────────────────────────────
def compute_community_avg_features():
    """
    Read all household_features documents once at startup.
    Compute average value for each of the 18 model features.
    Cached in COMMUNITY_AVG_FEATURES — used when no household match is found.
    """
    global COMMUNITY_AVG_FEATURES

    if db is None:
        logger.warning("Cannot compute community averages — db unavailable")
        return

    logger.info("Computing community averages from household_features...")
    sums   = defaultdict(float)
    counts = defaultdict(int)
    total  = 0

    try:
        for doc in db.collection("household_features").stream():
            data     = doc.to_dict() or {}
            features = data.get("features", data)  # support nested features map
            total   += 1
            for col in MODEL_FEATURE_COLUMNS:
                raw = features.get(col, data.get(col))
                if raw is not None:
                    try:
                        sums[col]   += float(raw)
                        counts[col] += 1
                    except (TypeError, ValueError):
                        pass

        if total == 0:
            logger.warning("household_features is empty — model will use zero defaults")
            return

        for col in MODEL_FEATURE_COLUMNS:
            if counts[col] > 0:
                COMMUNITY_AVG_FEATURES[col] = round(sums[col] / counts[col], 4)

        logger.info(
            f"Community averages computed from {total} documents — "
            f"avg total_people: {COMMUNITY_AVG_FEATURES.get('total_people')}, "
            f"avg coli_mpn: {COMMUNITY_AVG_FEATURES.get('coli_mpn')}"
        )
    except Exception as e:
        logger.error(f"Failed to compute community averages: {e}")

# ── Firestore helpers ─────────────────────────────────────────────────────────
def fetch_household_features(meter_number: str) -> dict:
    """
    Look up household survey features by meterNumber in household_features.
    Falls back to community averages if no match found.
    This ensures the model always receives real numbers, never zeros.
    """
    if db is None:
        return dict(COMMUNITY_AVG_FEATURES)

    try:
        docs = (
            db.collection("household_features")
            .where("meterNumber", "==", meter_number)
            .limit(1)
            .stream()
        )
        for doc in docs:
            payload      = doc.to_dict() or {}
            features_map = payload.get("features", payload)
            # Start from community averages, override with this household's values
            result = dict(COMMUNITY_AVG_FEATURES)
            for col in MODEL_FEATURE_COLUMNS:
                raw = features_map.get(col, payload.get(col))
                if raw is not None:
                    try:
                        result[col] = float(raw)
                    except (TypeError, ValueError):
                        pass
            logger.info(f"Household features found for meter {meter_number}")
            return result
    except Exception as e:
        logger.warning(f"Could not fetch household_features for {meter_number}: {e}")

    logger.info(f"No household_features for {meter_number} — using community averages")
    return dict(COMMUNITY_AVG_FEATURES)

def resolve_person_and_household(meter_number: str):
    """
    Look up person name and household ID from users collection by meterNumber.
    Returns (person_name, household_id, user_matched).
    """
    if db is None:
        return "Unknown Person", None, False

    try:
        docs = (
            db.collection("users")
            .where("meterNumber", "==", meter_number)
            .limit(1)
            .stream()
        )
        for doc in docs:
            payload      = doc.to_dict() or {}
            name         = payload.get("name") or payload.get("personName") or "Unknown Person"
            household_id = payload.get("householdId") or payload.get("household_id") or doc.id
            return name, household_id, True
    except Exception as e:
        logger.warning(f"Could not query users for {meter_number}: {e}")

    return "Unknown Person", None, False

# ── Normalize waterUsage document ─────────────────────────────────────────────
def normalize_water_usage(data: dict) -> dict:
    """
    Map waterUsage field names to our internal normalized format.

    waterUsage field   →  normalized field
    ──────────────────────────────────────
    currentReading     →  total_liters       (liters consumed so far)
    remainingUnits     →  remaining_units    (liters left)
    leakageDetected    →  leak_flag          (hardware leak signal)
    dailyUsage         →  avg_daily_usage    (already calculated by hardware)
    meterNumber        →  meterNumber
    accountNumber      →  account_number
    userId             →  user_id
    """
    return {
        "meterNumber":     data.get("meterNumber"),
        "account_number":  data.get("accountNumber"),
        "user_id":         data.get("userId"),
        "total_liters":    float(data.get("currentReading") or data.get("unitConsumed") or 0),
        "remaining_units": float(data.get("remainingUnits", 0)),
        "leak_flag":       bool(data.get("leakageDetected", False)),
        "avg_daily_usage": float(data.get("dailyUsage", 0)),
        "hourly_usage":    float(data.get("hourlyUsage", 0)),
        "weekly_usage":    float(data.get("weeklyUsage", 0)),
        "monthly_usage":   float(data.get("monthlyUsage", 0)),
        "valve_status":    data.get("valveStatus", "unknown"),
        "battery_level":   data.get("batteryLevel"),
        "signal_strength": data.get("signalStrength"),
        "timestamp":       datetime.now(timezone.utc),
    }

# ── ML prediction functions ───────────────────────────────────────────────────
def build_feature_vector(household_features: dict) -> list:
    """Build the ordered 18-feature vector the model expects."""
    return [household_features.get(col, 0.0) for col in MODEL_FEATURE_COLUMNS]

def get_leak_probability(feature_vector: list) -> float:
    """
    Run the ML model and return leak probability between 0.0 and 1.0.
    Falls back to consumption_risk_score normalization if model unavailable.
    """
    if consumption_model and model_scaler:
        try:
            scaled = model_scaler.transform([feature_vector])
            proba  = consumption_model.predict_proba(scaled)[0][1]  # probability of class 1
            return round(float(proba), 4)
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}")

    # Fallback — normalize consumption_risk_score (range 0–6) to 0–1
    risk_score = feature_vector[MODEL_FEATURE_COLUMNS.index("consumption_risk_score")]
    return round(min(max(risk_score / 6.0, 0.0), 1.0), 4)

def classify_consumption_risk(leak_probability: float) -> str:
    """
    Map leak probability to Low / Medium / High.
    Thresholds from notebook: 60th percentile = Medium, 85th percentile = High.
    """
    if leak_probability > 0.85:
        return "High"
    elif leak_probability > 0.60:
        return "Medium"
    return "Low"

def detect_leak(leak_flag: bool, leak_probability: float) -> bool:
    """
    Hardware flag OR model probability above 0.85 triggers a leak alert.
    Either signal alone is sufficient.
    """
    return leak_flag or leak_probability > 0.85

def calculate_revenue(total_liters: float) -> float:
    """Revenue in KES: liters consumed × tariff rate."""
    return round(total_liters * WATER_TARIFF_PER_LITER, 2)

def estimate_sustainability(remaining_units: float, avg_daily_usage: float) -> dict:
    """
    How long will remaining water last at current daily usage rate?
    Uses dailyUsage directly from waterUsage — no history query needed.
    Returns both days and hours for dashboard flexibility.
    """
    if avg_daily_usage <= 0:
        return {"days": None, "hours": None}

    days  = round(remaining_units / avg_daily_usage, 2)
    hours = round(days * 24, 2)
    return {"days": days, "hours": hours}

# ── Core ML pipeline ──────────────────────────────────────────────────────────
def generate_predictions(normalized: dict) -> dict:
    """
    Full ML pipeline:
    1. Fetch household features (community averages as fallback)
    2. Build and scale 18-feature vector
    3. Run model → leak_probability
    4. Derive all 4 predictions
    5. Resolve person info from users collection
    6. Persist results to ml_predictions
    """
    meter_number = normalized["meterNumber"]

    # Step 1 — household features
    household_features = fetch_household_features(meter_number)

    # Step 2 — feature vector
    feature_vector = build_feature_vector(household_features)

    # Step 3 — ML
    leak_probability = get_leak_probability(feature_vector)

    # Step 4 — 4 predictions
    consumption_risk  = classify_consumption_risk(leak_probability)
    leak_detected     = detect_leak(normalized["leak_flag"], leak_probability)
    predicted_revenue = calculate_revenue(normalized["total_liters"])
    sustainability    = estimate_sustainability(
        normalized["remaining_units"],
        normalized["avg_daily_usage"],  # comes directly from waterUsage.dailyUsage
    )

    # Step 5 — who owns this meter
    person_name, household_id, user_matched = resolve_person_and_household(meter_number)

    prediction_result = {
        # Identity
        "meterNumber":                meter_number,
        "accountNumber":              normalized.get("account_number"),
        "userId":                     normalized.get("user_id"),
        "personName":                 person_name,
        "householdId":                household_id,
        "userMatched":                user_matched,

        # The 4 core predictions
        "consumption_risk":           consumption_risk,     # Low / Medium / High
        "leak_detected":              leak_detected,        # True / False
        "leak_probability":           leak_probability,     # 0.0 → 1.0 for dashboard gauge
        "predicted_revenue":          predicted_revenue,    # KES
        "predicted_revenue_currency": REVENUE_CURRENCY,
        "sustainability_days":        sustainability["days"],
        "sustainability_hours":       sustainability["hours"],

        # Context passed through from waterUsage for dashboard use
        "total_liters":               normalized["total_liters"],
        "remaining_units":            normalized["remaining_units"],
        "avg_daily_usage_liters":     normalized["avg_daily_usage"],
        "hourly_usage":               normalized["hourly_usage"],
        "weekly_usage":               normalized["weekly_usage"],
        "monthly_usage":              normalized["monthly_usage"],
        "valve_status":               normalized["valve_status"],
        "battery_level":              normalized["battery_level"],
        "signal_strength":            normalized["signal_strength"],

        "generated_at":               datetime.now(timezone.utc),
    }

    # Write predictions to Firestore
    db.collection("ml_predictions").add(prediction_result)

    return prediction_result

# ── Firestore listener ────────────────────────────────────────────────────────
def process_water_usage_document(doc_snapshot):
    """Process a single waterUsage document through the full ML pipeline."""
    data         = doc_snapshot.to_dict() or {}
    meter_number = data.get("meterNumber")

    if not meter_number:
        logger.warning("waterUsage document missing meterNumber — skipping")
        return

    try:
        normalized        = normalize_water_usage(data)
        prediction_result = generate_predictions(normalized)
        logger.info(
            f"Meter {meter_number} → "
            f"risk: {prediction_result['consumption_risk']} | "
            f"leak: {prediction_result['leak_detected']} | "
            f"revenue: {prediction_result['predicted_revenue']} KES | "
            f"sustainability: {prediction_result['sustainability_days']} days"
        )
    except Exception as e:
        logger.error(f"Prediction failed for meter {meter_number}: {e}")

def water_usage_on_snapshot(col_snapshot, changes, read_time):
    """
    Firestore listener callback for waterUsage collection.
    Handles ADDED (new meter registered) and MODIFIED (meter reading updated).
    Skips the initial bootstrap snapshot to avoid reprocessing existing data.
    """
    global listener_bootstrapped

    if not listener_bootstrapped:
        listener_bootstrapped = True
        logger.info("waterUsage listener bootstrapped — ready for new readings")
        return

    for change in changes:
        if change.type.name not in ("ADDED", "MODIFIED"):
            continue
        try:
            process_water_usage_document(change.document)
        except Exception as e:
            logger.error(f"Unhandled error in listener: {e}")

def start_water_usage_listener():
    """Start the Firestore real-time listener on waterUsage."""
    global water_usage_watch
    if db is None:
        logger.warning("Listener not started — db unavailable")
        return
    with listener_lock:
        if water_usage_watch is None:
            water_usage_watch = db.collection("waterUsage").on_snapshot(water_usage_on_snapshot)
            logger.info("Firestore listener started on waterUsage")

def stop_water_usage_listener():
    """Stop the Firestore listener cleanly on shutdown."""
    global water_usage_watch
    with listener_lock:
        if water_usage_watch is not None:
            water_usage_watch.unsubscribe()
            water_usage_watch = None
            logger.info("Firestore listener stopped")

# ── Health check endpoint ─────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """
    Confirms all system components are running correctly.
    community_avg_loaded: True means model has real features to work with.
    listener_active: True means real-time processing is running.
    """
    return {
        "status":                     "ok",
        "ml_model_loaded":            consumption_model is not None,
        "scaler_loaded":              model_scaler is not None,
        "db_connected":               db is not None,
        "listener_active":            water_usage_watch is not None,
        "community_avg_loaded":       any(v > 0 for v in COMMUNITY_AVG_FEATURES.values()),
        "community_avg_total_people": COMMUNITY_AVG_FEATURES.get("total_people"),
        "timestamp":                  datetime.now(timezone.utc),
    }