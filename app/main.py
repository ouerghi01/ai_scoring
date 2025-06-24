from flask import Flask, request, jsonify
import logging  # For better error logging
from flask_cors import CORS
from model import predict_model
from schemas import LoanApplication
import json
# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data: LoanApplication = request.get_json(
            force=True
        )  
        
        resulta = predict_model(app, data)
        result_dict = process_prediction_data(resulta)
        print(result_dict)
        return jsonify(result_dict)

    except Exception as e:
        app.logger.error(
            f"An unexpected error occurred during prediction: {e}", exc_info=True
        )
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

def process_prediction_data(resulta):
    resulta = resulta.get_data(as_text=True)
    resulta = json.loads(resulta)  # Convert bytes to JSON
    
    return resulta


# --- How to run the Flask app ---
if __name__ == "__main__":
    
    app.run(debug=True, host="0.0.0.0", port=5000)
    
