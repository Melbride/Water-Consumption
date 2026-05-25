from fastapi import FastAPI
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from collections import defaultdict
import joblib
import pickle
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
import logging
import threading

#Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Firebase
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

#Constants
WATER_TARIFF_PER_LITER = 0.5   # KES per liter (10 KES per 20L)
REVENUE_CURRENCY       = "KES"

#The 18 features the model was trained on (order matters for scaling)
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

#Load ML model
try:
    consumption_model = joblib.load("models/water_model_production_final.pkl")
    logger.info("ML model loaded successfully")
except Exception as e:
    consumption_model = None
    logger.warning(f"ML model not loaded: {e}")

#Load scaler
try:
    with open("models/model_config_final.pkl", "rb") as f:
        model_config = pickle.load(f)
    model_scaler = model_config.get("scaler")
    logger.info("Model scaler loaded successfully")
except Exception as e:
    model_scaler = None
    logger.warning(f"Model scaler not loaded: {e}")

#Listener state
water_usage_watch     = None
listener_bootstrapped = False
listener_lock         = threading.Lock()

#Community average features — computed once at startup
# Used as fallback when a meter has no matching household_features document.
COMMUNITY_AVG_FEATURES: dict = {col: 0.0 for col in MODEL_FEATURE_COLUMNS}

# In-memory caches — reduce repeated Firestore reads per meter
# household features cache: meterNumber → features dict
HOUSEHOLD_FEATURES_CACHE: dict = {}
# user info cache: meterNumber → (name, household_id, matched)
USER_INFO_CACHE: dict = {}

#Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: start listener immediately so Render wakes up fast.
    Compute community averages in background — doesn't block startup.
    Shutdown: clean up listener.
    """
    start_water_usage_listener()
    thread = threading.Thread(target=compute_community_avg_features, daemon=True)
    thread.start()
    yield
    stop_water_usage_listener()

app = FastAPI(title="Water Consumption ML API", lifespan=lifespan)

#Community average computation
def compute_community_avg_features():
    """
    Try to load community averages from a single cached Firestore document first.
    Cache is valid for 24 hours — only 1 read on most restarts.
    If cache is missing or stale, read all household_features docs and recompute.
    Saves recomputed averages back to cache for next restart.

    Firestore reads per restart:
        Normal restart  → 1 read (cache hit)
        First run / daily refresh → 6055 reads + 1 write (cache miss)
    """
    global COMMUNITY_AVG_FEATURES

    if db is None:
        logger.warning("Cannot compute community averages — db unavailable")
        return

    # Try cache first — only 1 Firestore read
    try:
        cache_doc = db.collection("community_cache").document("avg_features").get()
        if cache_doc.exists:
            cache_data = cache_doc.to_dict() or {}
            cached_at  = cache_data.get("cached_at")
            if cached_at:
                age_hours = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600
                if age_hours < 24:
                    for col in MODEL_FEATURE_COLUMNS:
                        if col in cache_data:
                            COMMUNITY_AVG_FEATURES[col] = cache_data[col]
                    logger.info(
                        f"Community averages loaded from cache — "
                        f"avg total_people: {COMMUNITY_AVG_FEATURES.get('total_people')}, "
                        f"cache age: {age_hours:.1f} hours"
                    )
                    return
    except Exception as e:
        logger.warning(f"Cache read failed: {e} — recomputing from source")

    # Cache missing or stale — recompute from all household_features docs
    logger.info("Computing community averages from household_features...")
    sums   = defaultdict(float)
    counts = defaultdict(int)
    total  = 0

    try:
        for doc in db.collection("household_features").stream():
            data     = doc.to_dict() or {}
            features = data.get("features", data)
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
            logger.warning("household_features is empty — using zero defaults")
            return

        for col in MODEL_FEATURE_COLUMNS:
            if counts[col] > 0:
                COMMUNITY_AVG_FEATURES[col] = round(sums[col] / counts[col], 4)

        # Save averages to cache — next restart reads 1 doc instead of 6055
        cache_payload = dict(COMMUNITY_AVG_FEATURES)
        cache_payload["cached_at"]  = datetime.now(timezone.utc)
        cache_payload["total_docs"] = total
        db.collection("community_cache").document("avg_features").set(cache_payload)

        logger.info(
            f"Community averages computed from {total} documents and cached — "
            f"avg total_people: {COMMUNITY_AVG_FEATURES.get('total_people')}, "
            f"avg coli_mpn: {COMMUNITY_AVG_FEATURES.get('coli_mpn')}"
        )
    except Exception as e:
        logger.error(f"Failed to compute community averages: {e}")

#Firestore helpers
def fetch_household_features(meter_number: str) -> dict:
    """
    Look up household features by meterNumber.
    Checks in-memory cache first — zero Firestore reads on repeat predictions.
    Falls back to community averages if no match found.

    Firestore reads per meter:
        First prediction → 1 read (then cached in memory)
        All subsequent   → 0 reads (served from memory cache)
    """
    #Check memory cache first, no Firestore read needed
    if meter_number in HOUSEHOLD_FEATURES_CACHE:
        return HOUSEHOLD_FEATURES_CACHE[meter_number]

    if db is None:
        return dict(COMMUNITY_AVG_FEATURES)

    try:
        docs = (
            db.collection("household_features")
            .where(filter=firestore.FieldFilter("meterNumber", "==", meter_number))
            .limit(1)
            .stream()
        )
        for doc in docs:
            payload      = doc.to_dict() or {}
            features_map = payload.get("features", payload)
            result       = dict(COMMUNITY_AVG_FEATURES)
            for col in MODEL_FEATURE_COLUMNS:
                raw = features_map.get(col, payload.get(col))
                if raw is not None:
                    try:
                        result[col] = float(raw)
                    except (TypeError, ValueError):
                        pass
            #Save to memory cache, no Firestore read next time
            HOUSEHOLD_FEATURES_CACHE[meter_number] = result
            logger.info(f"Household features found and cached for meter {meter_number}")
            return result
    except Exception as e:
        logger.warning(f"Could not fetch household_features for {meter_number}: {e}")

    #No match, use community averages and cache that too
    result = dict(COMMUNITY_AVG_FEATURES)
    HOUSEHOLD_FEATURES_CACHE[meter_number] = result
    logger.info(f"No household_features for {meter_number} — using community averages")
    return result

def resolve_person_and_household(meter_number: str):
    """
    Look up person name and household ID from users collection by meterNumber.
    Checks in-memory cache first — zero Firestore reads on repeat predictions.
    Returns (person_name, household_id, user_matched).

    Firestore reads per meter:
        First prediction → 1 read (then cached in memory)
        All subsequent   → 0 reads (served from memory cache)
    """
    #Check memory cache first
    if meter_number in USER_INFO_CACHE:
        return USER_INFO_CACHE[meter_number]

    if db is None:
        return "Unknown Person", None, False

    try:
        docs = (
            db.collection("users")
            .where(filter=firestore.FieldFilter("meterNumber", "==", meter_number))
            .limit(1)
            .stream()
        )
        for doc in docs:
            payload      = doc.to_dict() or {}
            name         = payload.get("name") or payload.get("personName") or "Unknown Person"
            household_id = payload.get("householdId") or payload.get("household_id") or doc.id
            result       = (name, household_id, True)
            USER_INFO_CACHE[meter_number] = result
            return result
    except Exception as e:
        logger.warning(f"Could not query users for {meter_number}: {e}")

    result = ("Unknown Person", None, False)
    USER_INFO_CACHE[meter_number] = result
    return result

#Normalize waterUsage document
def normalize_water_usage(data: dict) -> dict:
    """
    Map waterUsage field names to our internal normalized format.

    waterUsage field   →  normalized field
    currentReading     →  total_liters
    remainingUnits     →  remaining_units
    leakageDetected    →  leak_flag
    dailyUsage         →  avg_daily_usage
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

#ML prediction functions
def build_feature_vector(household_features: dict) -> list:
    """Build the ordered 18-feature vector the model expects."""
    return [household_features.get(col, 0.0) for col in MODEL_FEATURE_COLUMNS]

def get_leak_probability(feature_vector: list, leak_flag: bool = False) -> float:
    """
    Run the ML model and return leak probability between 0.0 and 1.0.
    Hardware leak flag is the only way to get 1.0 (100% certainty).
    Model alone is capped at 0.75 — prevents false alarms from community averages.
    Falls back to consumption_risk_score normalization if model unavailable.
    """
    if leak_flag:
        return 1.0

    if consumption_model and model_scaler:
        try:
            df     = pd.DataFrame([feature_vector], columns=MODEL_FEATURE_COLUMNS)
            scaled = model_scaler.transform(df)
            proba  = consumption_model.predict_proba(scaled)[0][1]
            return round(min(float(proba), 0.75), 4)
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}")

    # Fallback, normalize consumption_risk_score (range 0–6) to 0–1, capped at 0.75
    risk_score = feature_vector[MODEL_FEATURE_COLUMNS.index("consumption_risk_score")]
    return round(min(max(risk_score / 6.0, 0.0), 0.75), 4)

def classify_consumption_risk(leak_probability: float) -> str:
    """
    Map leak probability to Low / Medium / High.
    High:   above 0.70 — genuinely concerning
    Medium: above 0.45 — worth monitoring
    Low:    below 0.45 — normal usage
    """
    if leak_probability > 0.70:
        return "High"
    elif leak_probability > 0.45:
        return "Medium"
    return "Low"

def detect_leak(leak_flag: bool, leak_probability: float) -> bool:
    """
    Hardware flag is definitive — always triggers leak alert.
    Model alone needs probability above 0.75 given Ghana training data uncertainty.
    """
    return leak_flag or leak_probability >= 0.75

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

#Core ML pipeline
def generate_predictions(normalized: dict) -> dict:
    """
    Full ML pipeline:
    1. Fetch household features (memory cache → Firestore → community averages)
    2. Build and scale 18-feature vector
    3. Run model → leak_probability
    4. Derive all 4 predictions
    5. Resolve person info (memory cache → Firestore)
    6. Persist results to ml_predictions
    """
    meter_number = normalized["meterNumber"]

    #household features
    household_features = fetch_household_features(meter_number)

    #feature vector
    feature_vector = build_feature_vector(household_features)

    #ML
    leak_probability = get_leak_probability(feature_vector, normalized["leak_flag"])

    #4 predictions
    consumption_risk  = classify_consumption_risk(leak_probability)
    leak_detected     = detect_leak(normalized["leak_flag"], leak_probability)
    predicted_revenue = calculate_revenue(normalized["total_liters"])
    sustainability    = estimate_sustainability(
        normalized["remaining_units"],
        normalized["avg_daily_usage"],
    )

    #who owns this meter
    person_name, household_id, user_matched = resolve_person_and_household(meter_number)

    prediction_result = {
        #Identity
        "meterNumber":                meter_number,
        "accountNumber":              normalized.get("account_number"),
        "userId":                     normalized.get("user_id"),
        "personName":                 person_name,
        "householdId":                household_id,
        "userMatched":                user_matched,

        #The 4 core predictions
        "consumption_risk":           consumption_risk,
        "leak_detected":              leak_detected,
        "leak_probability":           leak_probability,
        "predicted_revenue":          predicted_revenue,
        "predicted_revenue_currency": REVENUE_CURRENCY,
        "sustainability_days":        sustainability["days"],
        "sustainability_hours":       sustainability["hours"],

        #Context passed through from waterUsage for dashboard UI
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

    #Write predictions to Firestore — 1 write per prediction
    db.collection("ml_predictions").add(prediction_result)
    return prediction_result

#Firestore listener
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

#Root endpoint
@app.get("/")
def root():
    return {"message": "Water Consumption ML API is running"}

#Health check endpoint
@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    """
    Confirms all system components are running correctly.
    Accepts both GET and HEAD — HEAD is used by UptimeRobot keep-alive pings.
    """
    return {
        "status":                     "ok",
        "ml_model_loaded":            consumption_model is not None,
        "scaler_loaded":              model_scaler is not None,
        "db_connected":               db is not None,
        "listener_active":            water_usage_watch is not None,
        "community_avg_loaded":       any(v > 0 for v in COMMUNITY_AVG_FEATURES.values()),
        "community_avg_total_people": COMMUNITY_AVG_FEATURES.get("total_people"),
        "cached_meters":              len(HOUSEHOLD_FEATURES_CACHE),
        "timestamp":                  datetime.now(timezone.utc),
    }
