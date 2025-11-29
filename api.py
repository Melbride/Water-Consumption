from typing import Collection
from fastapi import FastAPI, Response
# Add safe conversion functions at the top of the file
def safe_int_convert(value, default=0):
    """Safely convert any value to int, handling NaN and other issues"""
    try:
        if pd.isna(value) or value == float('nan') or value == float('inf') or value == float('-inf') or value is None:
            return default
        return int(float(value))
    except (ValueError, TypeError, OverflowError):
        return default

def safe_str_convert(value, default="unknown"):
    """Safely convert any value to string, handling NaN and other issues"""
    try:
        if pd.isna(value) or value == float('nan') or value == float('inf') or value == float('-inf') or value is None:
            return default
        return str(value)
    except (ValueError, TypeError):
        return default

def safe_bool_convert(value, default=False):
    """Safely convert any value to bool, handling NaN and other issues"""
    try:
        if pd.isna(value) or value == float('nan') or value == float('inf') or value == float('-inf') or value is None:
            return default
        return bool(value)
    except (ValueError, TypeError):
        return default
import csv
import io
import pickle
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Get absolute paths for model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'datasets')

print(f"Base directory: {BASE_DIR}")
print(f"Models directory: {MODEL_PATH}")
print(f"Data directory: {DATA_PATH}")

cred = credentials.Certificate(os.path.join(BASE_DIR, "serviceAccountKey.json"))
firebase_admin.initialize_app(cred, {"projectId": "smartpay-9558e"})

db = firestore.client()
# Load final retrained household model and config
print("Loading household models...")
try:
    with open(os.path.join(MODEL_PATH, 'water_model_production_final.pkl'), 'rb') as f:
        household_model = pickle.load(f)
    print("Household model loaded")
    
    with open(os.path.join(MODEL_PATH, 'model_config_final.pkl'), 'rb') as f:
        model_config = pickle.load(f)
    print("Model config loaded")
    
    household_feature_columns = model_config['feature_columns']
    scaler = model_config['scaler']
    print(f"Final household model loaded successfully!")
    print(f"   Model type: {model_config.get('model_type', 'Unknown')}")
    print(f"   Features: {len(household_feature_columns)}")
    print(f"   Accuracy: {model_config.get('accuracy', 'Unknown')}")
    print(f"   AUC: {model_config.get('auc', 'Unknown')}")
        
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
    print("Final household model not found - using hardware models only")
    household_model = None
    household_feature_columns = []
    scaler = None
except Exception as e:
    print(f"Error loading final household model: {e}")
    print("Using hardware models only")
    household_model = None
    household_feature_columns = []
    scaler = None

print(f"Final status - household_feature_columns: {len(household_feature_columns) if household_feature_columns else 0}")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Water Consumption ML API is running!"}


@app.get("/api/explore")
def explore_firebase():
    """Explore all Firebase collections and sample data"""
    collections = {}
    try:
        for collection in db.collections():
            Collection_name = collection.id
            docs = collection.limit(3).stream()
            collections[Collection_name] = [{"id": doc.id, **doc.to_dict()} for doc in docs]
        return {
            "status": "success",
            "collections_found": len(collections),
            "data": collections
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/collections")
def list_collections():
    """List all collection names in Firebase"""
    try:
        collections = [collection.id for collection in db.collections()]
        return {
            "status": "success",
            "collections": collections
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

#Hardware Data Endpoint
@app.post("/api/hardware-data")
def receive_hardware_data(data: dict):
    """Bridge endpoint to receive hardware data and make predictions"""
    try:
        print(f"Received hardware data: {data}")
        
        # Store raw hardware data for analysis
        raw_data_ref = db.collection("raw_hardware_data")
        doc_ref = raw_data_ref.add({
            "raw_data": data,
            "received_at": firestore.SERVER_TIMESTAMP,
            "processed": False
        })
        
        # Process hardware format
        processed_data = process_your_hardware_format(data)
        
        if processed_data:
            # Store in ML-ready format
            water_readings_ref = db.collection("water_readings")
            water_readings_ref.add(processed_data)
            
            # MAKE PREDICTIONS AUTOMATICALLY
            predictions = make_ml_prediction(processed_data)
            
            # Store predictions in Firebase
            prediction_record = {
                "hardware_input": data,
                "ml_input": processed_data,
                "predictions": predictions,
                "timestamp": firestore.SERVER_TIMESTAMP
            }
            db.collection("ml_predictions").add(prediction_record)
            
            return {
                "status": "success",
                "message": "Hardware data received, processed, and predictions made",
                "raw_id": doc_ref[1].id,
                "extracted_fields": list(processed_data.keys()),
                "predictions": predictions
            }
        else:
            return {
                "status": "received",
                "message": "Data stored but no water reading detected",
                "raw_id": doc_ref[1].id
            }
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

def process_your_hardware_format(raw_data: dict):
    """Process YOUR specific hardware format"""
    try:
        #Calculate consumption from hardware data
        total_liters = raw_data.get('totalLiters', 0)
        remaining_units = raw_data.get('remainingUnits', 0)
        
        #Handle NaN values
        if pd.isna(total_liters) or total_liters == float('nan'):
            total_liters = 0
        if pd.isna(remaining_units) or remaining_units == float('nan'):
            remaining_units = 0
            
        consumption = total_liters - remaining_units
        
        #Convert to ML format
        ml_data = {
            "reading_litres": float(total_liters),
            "consumption_litres": float(consumption),
            "remaining_litres": float(remaining_units),
            "flow_rate_litres_per_sec": float(consumption / 30 if consumption > 0 else 0),
            "source_liters": float(raw_data.get('sourceLiters', 0)),
            "total_liters": float(total_liters),
            "device_id": str(raw_data.get('meter_id', 'UNKNOWN')),
            "timestamp": raw_data.get('timestamp', firestore.SERVER_TIMESTAMP),
            # Keep original hardware data for reference
            "hardware_leak": bool(raw_data.get('leak', False))
        }
        
        print(f"Converted to ML format: {ml_data}")
        return ml_data
        
    except Exception as e:
        print(f"Error processing hardware data: {e}")
        return None

def make_ml_prediction(data: dict):
    """Make ML predictions and add household context"""
    try:
        #Load models
        with open(os.path.join(MODEL_PATH, 'water_ml_models.pkl'), 'rb') as f:
            models = pickle.load(f)
        
        #Extract features with comprehensive NaN handling
        features = models['features']
        feature_values = []
        for feature in features:
            value = data.get(feature, 0)
            try:
                #Handle all types of invalid values
                if pd.isna(value) or value == float('nan') or value == float('inf') or value == float('-inf') or value is None:
                    feature_values.append(0.0)
                else:
                    feature_values.append(float(value))
            except (ValueError, TypeError):
                feature_values.append(0.0)
        
        X = np.array([feature_values])
        
        #Make predictions
        predictions = {}
        
        #Consumption Risk
        try:
            risk_pred = models['risk_model'].predict(X)[0]
            risk_label = models['risk_encoder'].inverse_transform([risk_pred])[0]
            predictions['consumption_risk'] = {"prediction": risk_label}
        except Exception as e:
            predictions['consumption_risk'] = {"prediction": "medium", "error": str(e)}
        
        #Leak Detection
        try:
            leak_pred = models['leak_model'].predict(X)[0]
            leak_prob = models['leak_model'].predict_proba(X)[0][1]
            predictions['leak_detection'] = {
                "prediction": bool(leak_pred),
                "probability": float(leak_prob)
            }
        except Exception as e:
            predictions['leak_detection'] = {"prediction": False, "probability": 0.0, "error": str(e)}
        
        #Revenue Prediction
        try:
            revenue_pred = models['revenue_model'].predict(X)[0]
            predictions['revenue_prediction'] = {
                "predicted_revenue": float(revenue_pred),
                "currency": "USD"
            }
        except Exception as e:
            predictions['revenue_prediction'] = {"predicted_revenue": 0.0, "currency": "USD", "error": str(e)}
        
        #Sustainability Prediction
        try:
            sustainability_pred = models['sustain_model'].predict(X)[0]
            predictions['sustainability_prediction'] = {
                "hours_remaining": float(sustainability_pred),
                "days_remaining": float(sustainability_pred / 24)
            }
        except Exception as e:
            predictions['sustainability_prediction'] = {"hours_remaining": 0.0, "days_remaining": 0.0, "error": str(e)}
        
        #Add household enhancement
        household_enhancement = {}
        if household_model and 'device_id' in data:
            try:
                from meter_mapping import get_household_id
                household_id = get_household_id(data['device_id'])
                if household_id:
                    household_enhancement = {
                        "household_id": household_id,
                        "enhancement_available": True
                    }
            except:
                household_enhancement = {"enhancement_available": False}
        else:
            household_enhancement = {"enhancement_available": False}
        
        predictions['household_enhancement'] = household_enhancement
        return predictions
        
    except Exception as e:
        return {"error": str(e)}

def get_household_features_from_firebase(household_id):
    """Get household features from Firebase for Gradient Boosting model"""
    try:
        household_ref = db.collection("household_features")
        docs = household_ref.where("household_id", "==", household_id).stream()
        
        for doc in docs:
            data = doc.to_dict()
            return data['features']  # Return 197 features
        
        return None
    except Exception as e:
        print(f"Error getting household features: {e}")
        return None

def use_gradient_boosting_model(household_id):
    """Use GradientBoosting model for comprehensive household leak prediction"""
    try:
        if not household_model or not scaler:
            return {"error": "Model or scaler not loaded"}
        
        #Get household features
        household_features = get_household_features_from_firebase(household_id)
        
        if household_features is None:
            return {"error": "Household features not found"}
        
        #Convert to array in correct order (match training features)
        feature_values = []
        for feature in household_feature_columns:
            if feature in household_features:
                value = household_features[feature]
                # Handle NaN values - replace with 0
                if pd.isna(value) or value == float('nan') or value == float('inf') or value == float('-inf'):
                    feature_values.append(0)
                else:
                    feature_values.append(float(value))
            else:
                feature_values.append(0)  # Default for missing features
        
        # Scale features (same as training)
        features_scaled = scaler.transform([feature_values])
        
        # Make prediction with GradientBoosting
        leak_prediction = household_model.predict(features_scaled)[0]
        leak_probability = household_model.predict_proba(features_scaled)[0][1]
        
        return {
            "household_leak_prediction": bool(leak_prediction),
            "household_leak_probability": float(leak_probability),
            "household_id": household_id,
            "model_type": "GradientBoosting",
            "model_accuracy": model_accuracy,
            "model_auc": model_auc,
            "features_used": len(feature_values),
            "data_source": "final_water_monitoring_dataset",
            "prediction_confidence": "high" if leak_probability > 0.8 else "medium" if leak_probability > 0.5 else "low"
        }
        
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

@app.get("/api/household-predict")
def predict_household_leak(household_id: str):
    """Predict leak for specific household using Gradient Boosting model"""
    try:
        if not household_model or not scaler:
            return {"error": "Household model not loaded"}
        
        # Get household features from Firebase
        household_features = get_household_features_from_firebase(household_id)
        
        if household_features is None:
            return {"error": f"Household {household_id} not found in database"}
        
        # Convert to array in correct order
        feature_values = []
        for feature in household_feature_columns:
            if feature in household_features:
                value = household_features[feature]
                # Handle NaN values
                if pd.isna(value) or value == float('nan') or value == float('inf') or value == float('-inf'):
                    feature_values.append(0)
                else:
                    feature_values.append(float(value))
            else:
                feature_values.append(0)  # Default for missing features
        
        # Scale features
        features_scaled = scaler.transform([feature_values])
        
        # Make prediction
        leak_prediction = household_model.predict(features_scaled)[0]
        leak_probability = household_model.predict_proba(features_scaled)[0][1]
        
        return {
            "household_leak_prediction": bool(leak_prediction),
            "household_leak_probability": float(leak_probability),
            "household_id": household_id,
            "model_type": "GradientBoosting",
            "model_accuracy": model_config.get('accuracy', 'Unknown'),
            "model_auc": model_config.get('auc', 'Unknown'),
            "features_used": len(feature_values),
            "prediction_confidence": "high" if leak_probability > 0.8 else "medium" if leak_probability > 0.5 else "low"
        }
        
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

@app.get("/api/hardware-data/process-unprocessed")
def process_unprocessed_hardware_data():
    """Process all unprocessed hardware data from database"""
    try:
        print("Processing unprocessed hardware data...")
        
        # Get all unprocessed hardware data
        raw_data_ref = db.collection("raw_hardware_data")
        unprocessed_docs = raw_data_ref.where("processed", "==", False).get()
        
        processed_count = 0
        results = []
        
        for doc in unprocessed_docs:
            try:
                raw_data = doc.to_dict()
                hardware_input = raw_data.get("raw_data", {})
                
                print(f"Processing hardware data: {hardware_input}")
                
                # Process the hardware data (same as Postman endpoint
                processed_data = process_your_hardware_format(hardware_input)
                
                if processed_data:
                    # Store in ML-ready format
                    water_readings_ref = db.collection("water_readings")
                    water_readings_ref.add(processed_data)
                    
                    # Make ML predictions
                    predictions = make_ml_prediction(processed_data)
                    
                    # Store predictions
                    ml_predictions_ref = db.collection("ml_predictions")
                    ml_predictions_ref.add({
                        "ml_input": processed_data,
                        "hardware_input": hardware_input,
                        "predictions": predictions,
                        "timestamp": firestore.SERVER_TIMESTAMP
                    })
                    
                    # Mark as processed
                    doc.reference.update({"processed": True})
                    
                    processed_count += 1
                    results.append({
                        "doc_id": doc.id,
                        "status": "success",
                        "predictions": predictions
                    })
                    
                else:
                    results.append({
                        "doc_id": doc.id,
                        "status": "error",
                        "message": "Failed to process hardware data"
                    })
                    
            except Exception as e:
                results.append({
                    "doc_id": doc.id,
                    "status": "error",
                    "message": str(e)
                })
        
        return {
            "status": "success",
            "processed_count": processed_count,
            "total_unprocessed": len(unprocessed_docs),
            "results": results
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Processing error: {str(e)}"}

@app.get("/api/household-data/load")
def load_household_features():
    """Load household features from Excel to Firebase"""
    try:
        # Load household dataset (Final_water_dataset.csv )
        df = pd.read_csv(os.path.join(DATA_PATH, 'Final_water_dataset.csv'))
        print(f"Loaded production dataset with shape: {df.shape}")
        
        # Get preprocessing objects
        if not household_feature_columns:
            return {"status": "error", "message": "Model config not loaded - check model files"}
        
        print(f"Using {len(household_feature_columns)} features")
        
        # Clear existing data with timeout handling
        household_ref = db.collection('household_features')
        try:
            docs = household_ref.stream(timeout=10.0)  # 10 second timeout
            deleted_count = 0
            for doc in docs:
                household_ref.document(doc.id).delete()
                deleted_count += 1
                if deleted_count >= 1000:  # Limit deletion to prevent timeout
                    print(f"Deleted {deleted_count} documents, stopping to prevent timeout")
                    break
            print(f"Cleared {deleted_count} existing household records")
        except Exception as e:
            print(f"Warning: Could not clear existing data: {e}")
            # Continue with loading even if clearing fails
        
        # Load household features with better NaN handling and batching
        loaded_count = 0
        # Process in batches to prevent timeout
        batch_size = 500  
        batch = db.batch()
        
        #Process each household row with production dataset structure
        for index, row in df.iterrows():
            #Extract features for model prediction
            features = {}
            for col in household_feature_columns:
                if col in df.columns and col != 'Leak_Alert':
                    value = row[col]
                    # Handle NaN values for production dataset
                    try:
                        if pd.isna(value) or value == float('nan') or value == float('inf') or value == float('-inf') or value is None:
                            features[col] = 0.0
                        else:
                            features[col] = float(value)
                    except (ValueError, TypeError):
                        features[col] = 0.0
            
            # Create household data document with production dataset fields
            household_data = {
                "household_id": str(int(row.get('id', index))),
                "features": features,
                "leak_alert": bool(row.get('Leak_Alert', False)),
                "adult_count": int(row.get('adult_count', 0)),
                "child_count": int(row.get('child_count', 0)),
                "rooms_in_hh": int(row.get('rooms_in_hh', 0)),
                "years_in_community": float(row.get('years_in_community', 0)),
                "primary_dw_source": str(row.get('primary_dw_source', 'unknown')),
                "daily_hh_water_cost_for_pay_to_fetch": float(row.get('daily_hh_water_cost_for_pay_to_fetch', 0)),
                "water_storage_drinking_water": bool(row.get('water_storage_drinking_water', False)),
                "coli_mpn": float(row.get('coli_mpn', 0)),
                "tc_mpn": float(row.get('tc_mpn', 0)),
                "tap_closure_days_per_week": float(row.get('tap_closure_days_per_week', 0)),
                "estimated_storage_capacity_liters": float(row.get('estimated_storage_capacity_liters', 0)),
                "total_people": int(row.get('total_people', 0)),
                "people_per_room": float(row.get('people_per_room', 0)),
                "consumption_risk_score": float(row.get('consumption_risk_score', 0)),
                "water_access_score": float(row.get('water_access_score', 0)),
                "estimated_non_dw_storage_capacity": float(row.get('estimated_non_dw_storage_capacity', 0)),
                "estimated_stored_non_dw": float(row.get('estimated_stored_non_dw', 0)),
                "avg_price_per_liter_cedis": float(row.get('avg_price_per_liter_cedis', 0)),
                "loaded_at": firestore.SERVER_TIMESTAMP
            }
                
            #Add to batch
            doc_ref = household_ref.document()
            batch.set(doc_ref, household_data)
            loaded_count += 1
            
            #Commit batch when full
            if loaded_count % batch_size == 0:
                batch.commit()
                batch = db.batch()
                print(f"Loaded {loaded_count} households...")
                    
        #Commit remaining batch
        if loaded_count % batch_size != 0:
            batch.commit()
        
        return {
            "status": "success",
            "message": f"Loaded {loaded_count} households to Firebase",
            "features_per_household": len(household_feature_columns),
            "model_ready": True
        }
        
    except Exception as e:
        print(f"Error in load_household_features: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.get("/api/household-data/load-skip-clear")
def load_household_features_skip_clear():
    """Load household features without clearing existing data (faster)"""
    try:
        #Load household dataset (Final_water_dataset.csv)
        df = pd.read_csv(os.path.join(DATA_PATH, 'Final_water_dataset.csv'))
        print(f"Loaded production dataset with shape: {df.shape}")
        
        #Get preprocessing objects
        if not household_feature_columns:
            return {"status": "error", "message": "Model config not loaded - check model files"}
        
        print(f"Using {len(household_feature_columns)} features")
        
        #Skip clearing existing data - just load new data
        household_ref = db.collection('household_features')
        
        #Load household features with batching
        loaded_count = 0
        batch_size = 500
        batch = db.batch()
        
        #Process each household row with production dataset structure
        for index, row in df.iterrows():
            #Extract features for model prediction
            features = {}
            for col in household_feature_columns:
                if col in df.columns and col != 'Leak_Alert':
                    value = row[col]
                    #Handle NaN values for production dataset
                    try:
                        if pd.isna(value) or value == float('nan') or value == float('inf') or value == float('-inf') or value is None:
                            features[col] = 0.0
                        else:
                            features[col] = float(value)
                    except (ValueError, TypeError):
                        features[col] = 0.0
            
            #Create household data document with production dataset fields
            household_data = {
                "household_id": str(int(row.get('id', index))),
                "features": features,
                "leak_alert": bool(row.get('Leak_Alert', False)),
                "adult_count": int(row.get('adult_count', 0)),
                "child_count": int(row.get('child_count', 0)),
                "rooms_in_hh": int(row.get('rooms_in_hh', 0)),
                "years_in_community": float(row.get('years_in_community', 0)),
                "primary_dw_source": str(row.get('primary_dw_source', 'unknown')),
                "daily_hh_water_cost_for_pay_to_fetch": float(row.get('daily_hh_water_cost_for_pay_to_fetch', 0)),
                "water_storage_drinking_water": bool(row.get('water_storage_drinking_water', False)),
                "coli_mpn": float(row.get('coli_mpn', 0)),
                "tc_mpn": float(row.get('tc_mpn', 0)),
                "tap_closure_days_per_week": float(row.get('tap_closure_days_per_week', 0)),
                "estimated_storage_capacity_liters": float(row.get('estimated_storage_capacity_liters', 0)),
                "total_people": int(row.get('total_people', 0)),
                "people_per_room": float(row.get('people_per_room', 0)),
                "consumption_risk_score": float(row.get('consumption_risk_score', 0)),
                "water_access_score": float(row.get('water_access_score', 0)),
                "estimated_non_dw_storage_capacity": float(row.get('estimated_non_dw_storage_capacity', 0)),
                "estimated_stored_non_dw": float(row.get('estimated_stored_non_dw', 0)),
                "avg_price_per_liter_cedis": float(row.get('avg_price_per_liter_cedis', 0)),
                "loaded_at": firestore.SERVER_TIMESTAMP
            }
                
            #Add to batch
            doc_ref = household_ref.document()
            batch.set(doc_ref, household_data)
            loaded_count += 1
            
            #Commit batch when full
            if loaded_count % batch_size == 0:
                batch.commit()
                batch = db.batch()
                print(f"Loaded {loaded_count} households...")
                    
        #Commit remaining batch
        if loaded_count % batch_size != 0:
            batch.commit()
        
        return {
            "status": "success",
            "message": f"Loaded {loaded_count} households to Firebase (skipped clearing)",
            "features_per_household": len(household_feature_columns),
            "model_ready": True
        }
        
    except Exception as e:
        print(f"Error in load_household_features_skip_clear: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

def extract_water_reading(raw_data: dict):
    """Extract water reading from your specific hardware data format"""
    hardware_fields = {
        'current_units': ['currentunits', 'current_units', 'current'],
        'units_used': ['units_used', 'used_units', 'consumed', 'consumption'],
        'remaining_units': ['remaining_units', 'remaining', 'balance', 'left'],
        'timestamp': ['timestamp', 'time', 'datetime', 'date', 'created_at'],
        'device_id': ['device_id', 'meter_id', 'sensor_id', 'hardware_id', 'id'],
        'flow_rate': ['flow_rate', 'flow', 'rate', 'speed'],
        'pressure': ['pressure', 'psi', 'bar']
    }
    
    extracted = {}
    
    for target_field, possible_names in hardware_fields.items():
        for field_name in possible_names:
            if field_name in raw_data:
                extracted[target_field] = raw_data[field_name]
                break
    
    #Add timestamp if not present
    if 'timestamp' not in extracted:
        extracted['timestamp'] = firestore.SERVER_TIMESTAMP
    
    #Convert to litres and calculate additional metrics
    if 'current_units' in extracted:
        extracted['reading_litres'] = float(extracted['current_units'])
    
    if 'units_used' in extracted:
        extracted['consumption_litres'] = float(extracted['units_used'])
    
    if 'remaining_units' in extracted:
        extracted['remaining_litres'] = float(extracted['remaining_units'])
    
    #Calculate flow rate if we have consumption data
    if 'consumption_litres' in extracted:
        extracted['flow_rate_litres_per_sec'] = extracted['consumption_litres'] / 30  #per 30 seconds
    
    #Only return if we found water-related data
    if any(key in extracted for key in ['current_units', 'units_used', 'remaining_units']):
        return extracted
    
    return None

@app.get("/api/training-data/prepare")
def prepare_training_data():
    """Prepare training data from household_dataset.xlsx"""
    try:
        import pandas as pd
        
        #Load your household dataset
        df = pd.read_excel('household_dataset.xlsx')
        
        #Convert to Firebase-compatible format matching the hardware
        training_data = []
        
        for index, row in df.iterrows():
            #Create targets
            import numpy as np
            import random
            from datetime import datetime
            
            #consumption risk with household and time factors
            def create_realistic_consumption_risk(row):
                base_usage = row['usedUnits']
                #Add household size factor (simulate from house_id)
                household_size = hash(str(row['house_id'])) % 6 + 1  # 1-6 people
                #Larger households use more
                household_factor = 0.8 + (household_size * 0.1)  
                
                #Add time factor (from timestamp)
                hour = datetime.strptime(str(row['timestamp']), "%Y-%m-%d %H:%M:%S").hour
                 #Peak hours
                if 6 <= hour <= 9 or 18 <= hour <= 21: 
                    time_factor = 1.3
                    #Night hours
                elif 22 <= hour <= 5:  
                    time_factor = 0.7
                    #Normal hours
                else:  
                    time_factor = 1.0
                
                #Calculate adjusted usage with factors
                adjusted_usage = base_usage * household_factor * time_factor
                
                #Add realistic noise (±10%)
                noise = np.random.normal(0, 0.1)
                final_usage = adjusted_usage * (1 + noise)
                
                #Risk classification thresholds
                if final_usage > 80:
                    return "critical"
                elif final_usage > 50:
                    return "high"
                elif final_usage > 25:
                    return "medium"
                else:
                    return "low"
            
            #Realistic leak detection with multiple factors
            def create_realistic_leak_detection(row):
                usage = row['usedUnits']
                # Approximate flow rate
                flow_rate = usage / 60  
                
                #Multiple leak indicators (not just usage > 80)
                #High continuous flow
                continuous_flow = flow_rate > 2.0 
                #Used 40% of remaining
                sudden_spike = usage > (row['remainingUnits'] * 0.4)  
                unusual_time = datetime.strptime(str(row['timestamp']), "%Y-%m-%d %H:%M:%S").hour in range(2, 5)
                
                #Calculate leak probability
                leak_probability = 0.0
                if continuous_flow:
                    leak_probability += 0.4
                if sudden_spike:
                    leak_probability += 0.3
                if unusual_time and usage > 30:
                    leak_probability += 0.3
                
                #Add some randomness for realism
                leak_probability += random.uniform(-0.1, 0.1)
                
                return random.random() < max(0, min(leak_probability, 1.0))
            
            # Create targets
            record = {
                #Hardware-compatible features (6 fields)
                "reading_litres": float(row['currentUnits']),
                "consumption_litres": float(row['usedUnits']),
                "remaining_litres": float(row['remainingUnits']),
                "flow_rate_litres_per_sec": float(row['usedUnits']) / 60,  # per minute
                "timestamp": row['timestamp'],
                "device_id": str(row['metre_id']),
                
                # Additional features for better predictions
                "house_id": str(row['house_id']),
                "source_liters": float(row['sourceLiters']),
                "total_liters": float(row['totalLiters']),
                
                #TARGETS
                "consumption_risk": create_realistic_consumption_risk(row),
                "leak_detected": create_realistic_leak_detection(row),
                "efficiency_score": min(1.0, row['usedUnits'] / max(row['sourceLiters'], 1) * random.uniform(8, 12)),
                "sustainability_hours": row['remainingUnits'] / max(row['usedUnits'], 1) * random.uniform(0.8, 1.2),
                "revenue_impact": row['usedUnits'] * random.uniform(2.0, 3.0)  # Variable water rate
            }
            training_data.append(record)
        
        #Store in Firebase for training
        training_ref = db.collection("training_data")
        
        #Clear existing data first
        docs = training_ref.stream()
        for doc in docs:
            training_ref.document(doc.id).delete()
        
        #Add new training data
        for record in training_data:
            training_ref.add(record)
        
        return {
            "status": "success",
            "message": f"Created {len(training_data)} training records from household_dataset",
            "features": list(training_data[0].keys()),
            "sample_record": training_data[0]
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/ml/train")
def train_ml_model():
    """Train ML model using household data"""
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score, mean_squared_error
        
        #Get training data from Firebase
        training_ref = db.collection("training_data")
        docs = training_ref.stream()
        data = [{"id": doc.id, **doc.to_dict()} for doc in docs]
        
        if not data:
            return {"status": "error", "message": "No training data found"}
        
        df = pd.DataFrame(data)
        
        #Define features same as hardware data
        feature_columns = [
            'reading_litres', 'consumption_litres', 'remaining_litres', 
            'flow_rate_litres_per_sec', 'source_liters', 'total_liters'
        ]
        
        X = df[feature_columns]
        
        #Train multiple models
        models = {}
        
        #Consumption Risk Classification
        le_risk = LabelEncoder()
        y_risk = le_risk.fit_transform(df['consumption_risk'])
        X_train, X_test, y_train, y_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)
        risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
        risk_model.fit(X_train, y_train)
        risk_accuracy = accuracy_score(y_test, risk_model.predict(X_test))
        
        #Leak Detection
        y_leak = df['leak_detected'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y_leak, test_size=0.2, random_state=42)
        leak_model = RandomForestClassifier(n_estimators=100, random_state=42)
        leak_model.fit(X_train, y_train)
        leak_accuracy = accuracy_score(y_test, leak_model.predict(X_test))
        
        #Revenue Prediction
        y_revenue = df['revenue_impact']
        X_train, X_test, y_train, y_test = train_test_split(X, y_revenue, test_size=0.2, random_state=42)
        revenue_model = RandomForestRegressor(n_estimators=100, random_state=42)
        revenue_model.fit(X_train, y_train)
        revenue_mse = mean_squared_error(y_test, revenue_model.predict(X_test))
        
        #Sustainability Prediction
        y_sustain = df['sustainability_hours']
        X_train, X_test, y_train, y_test = train_test_split(X, y_sustain, test_size=0.2, random_state=42)
        sustain_model = RandomForestRegressor(n_estimators=100, random_state=42)
        sustain_model.fit(X_train, y_train)
        sustain_mse = mean_squared_error(y_test, sustain_model.predict(X_test))
        
        #Save models 
        models = {
            'risk_model': risk_model,
            'leak_model': leak_model, 
            'revenue_model': revenue_model,
            'sustain_model': sustain_model,
            'risk_encoder': le_risk,
            'features': feature_columns
        }
        
        with open('water_ml_models.pkl', 'wb') as f:
            pickle.dump(models, f)
        
        return {
            "status": "success",
            "message": "ML models trained successfully",
            "training_records": len(data),
            "model_performance": {
                "consumption_risk_accuracy": risk_accuracy,
                "leak_detection_accuracy": leak_accuracy,
                "revenue_mse": revenue_mse,
                "sustainability_mse": sustain_mse
            }
        }      
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/ml/predict")
def make_predictions(data: dict):
    """Enhanced prediction with household context"""
    try:
        #Load real-time models
        try:
            with open('water_ml_models.pkl', 'rb') as f:
                models = pickle.load(f)
        except FileNotFoundError:
            return {"status": "error", "message": "Models not trained yet. Use GET /api/ml/train first"}
        
        #Extract features from hardware data
        features = models['features']
        feature_values = []
        
        for feature in features:
            if feature in data:
                feature_values.append(float(data[feature]))
            else:
                return {"status": "error", "message": f"Missing feature: {feature}"}
        X = np.array([feature_values])
        
        #Make real-time predictions
        predictions = {}
        
        #Consumption Risk
        risk_pred = models['risk_model'].predict(X)[0]
        risk_label = models['risk_encoder'].inverse_transform([risk_pred])[0]
        predictions['consumption_risk'] = {"prediction": risk_label}
        
        #Leak Detection
        leak_pred = models['leak_model'].predict(X)[0]
        leak_prob = models['leak_model'].predict_proba(X)[0][1]
        predictions['leak_detection'] = {"prediction": bool(leak_pred), "probability": float(leak_prob)}
        
        #Revenue
        revenue_pred = models['revenue_model'].predict(X)[0]
        predictions['revenue_prediction'] = {"predicted_revenue": float(revenue_pred), "currency": "USD"}
        
        #Sustainability
        sustain_pred = models['sustain_model'].predict(X)[0]
        predictions['sustainability_prediction'] = {"hours_remaining": float(sustain_pred)}
        
        return predictions
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
