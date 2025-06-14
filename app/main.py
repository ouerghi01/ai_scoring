from flask import Flask, request, jsonify
import logging  # For better error logging
from flask_cors import CORS
from model import predict_model
from schemas import LoanApplication, LoanPredictionResult
import json
# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
app = Flask(__name__)
CORS(app)


# --- 2. Prediction Endpoint ---
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data: LoanApplication = request.get_json(
            force=True
        )  # Get JSON data from the request
        resulta = predict_model(app, data)
        resulta = resulta.get_data(as_text=True)
        resulta = json.loads(resulta)  # Convert bytes to JSON
        result = LoanPredictionResult(
            prediction=int(resulta["prediction"]),
            probability_not_approved=round(1-resulta["probability"], 4),
            probability_approved=round(resulta["probability"], 4),
            top_features=resulta["top_features"],
            recommendations=resulta.get(
                "recommendations", "No recommendations available at this time."
            ),
        )
        # Convert top_features to list of dicts for JSON serialization
        result_dict = result.__dict__.copy()
        result_dict["top_features"] = [tf.__dict__ for tf in result.top_features]
        app.logger.info(f"Prediction made successfully: {result}")
        print(f"Prediction made successfully: {result}")
        return jsonify(result_dict)

    except Exception as e:
        app.logger.error(
            f"An unexpected error occurred during prediction: {e}", exc_info=True
        )
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500


# --- How to run the Flask app ---
if __name__ == "__main__":
    # BEFORE RUNNING THIS FLASK APP:
    # 1. Ensure you have the updated training script (Step 1 above).
    # 2. Run the updated training script to generate 'random_forest_pipeline.pkl'.
    # 3. Save this Flask code as 'app.py' in the same directory as 'random_forest_pipeline.pkl'.
    # 4. Install necessary libraries: pip install Flask scikit-learn pandas joblib
    # 5. Open your terminal in that directory and run: python app.py
    app.run(debug=True, host="0.0.0.0", port=5000)
    # debug=True allows for auto-reloading and better error messages in development.
    # host='0.0.0.0' makes the server accessible from other machines on your network.
