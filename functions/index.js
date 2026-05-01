const functions = require("firebase-functions");
const admin = require("firebase-admin");
admin.initializeApp();

const fetch = require("node-fetch");

exports.predictOnWaterUsage = functions.firestore
    .document("waterUsage/{meterNumber}")
    .onCreate(async (snap, context) => {
      const data = snap.data();

      // Prepare payload for FastAPI
      const payload = {
        meterNumber: data.meterNumber,
        plot_id: data.plot_id || "UNKNOWN",
        totalLiters: data.currentReading || 0,
        remainingUnits: data.remainingUnits || 0,
        sourceLiters: data.sourceLiters || 0,
        leak: data.leakageDetected || false,
      };

      try {
        const response = await fetch("https://water-consumption-vd9i.onrender.com/api/hardware-data", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-api-key": "demo-secret-key",
          },
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }
        const result = await response.json();
        const prediction = result.predictions || {};
        // Add Firestore timestamp
        prediction.generated_at = admin.firestore.FieldValue.serverTimestamp();
        await admin.firestore().collection("ml_predictions").add(prediction);
      } catch (error) {
        console.error("Prediction API call failed:", error);
      }
      return null;
    });
