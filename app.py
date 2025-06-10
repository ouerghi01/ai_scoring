import pandas as pd
import numpy as np # Keep for consistency, though not strictly used in this Flask file directly
import joblib # Use joblib for loading models saved with joblib.dump
from flask import Flask, request, jsonify
import logging # For better error logging
from flask_cors import CORS
# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing cross-origin requests
# --- 1. Load the pre-trained model pipeline ---
# This model now includes all the preprocessing steps (scaling, encoding).
model_pipeline = None
pipeline_filename = 'random_forest_pipeline.pkl'
try:
    with open(pipeline_filename, 'rb') as f:
        model_pipeline = joblib.load(f)
    app.logger.info(f"Model pipeline '{pipeline_filename}' loaded successfully!")
except FileNotFoundError:
    app.logger.error(f"Error: Model pipeline '{pipeline_filename}' not found. Make sure it's in the same directory as app.py.")
except Exception as e:
    app.logger.error(f"Error loading model pipeline '{pipeline_filename}': {e}", exc_info=True)


# --- 2. Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({'error': 'Model pipeline not loaded. Cannot make predictions.'}), 500

    try:
        data = request.get_json(force=True) # Get JSON data from the request

        # Define expected input features and their order.
        # This order MUST match the columns X was fed in during training.
        expected_features_order = [
            'Professional_Status', 'Sector', 'Existing_Loan',
            'Total_Acquisition_Price_DT', 'Repayment_Duration_Years',
            'Monthly_Payment_DT', 'Documents_Complete',
            'Number_of_Clicks', 'Time_Spent_Seconds'
        ]

        # Basic validation for required fields
        for field in expected_features_order:
            if field not in data:
                app.logger.warning(f"Missing required field in request: {field}")
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Create a Pandas DataFrame from the incoming JSON data
        # Ensure the column order matches the training data features
        input_df = pd.DataFrame([data], columns=expected_features_order)

        # Convert data types to match those expected by the preprocessor/model
        # This is crucial as JSON might parse numbers as floats or strings.
        try:
            input_df['Total_Acquisition_Price_DT'] = pd.to_numeric(input_df['Total_Acquisition_Price_DT'])
            input_df['Repayment_Duration_Years'] = pd.to_numeric(input_df['Repayment_Duration_Years'])
            input_df['Monthly_Payment_DT'] = pd.to_numeric(input_df['Monthly_Payment_DT'])
            input_df['Number_of_Clicks'] = pd.to_numeric(input_df['Number_of_Clicks'])
            input_df['Time_Spent_Seconds'] = pd.to_numeric(input_df['Time_Spent_Seconds'])
            # Convert 'Documents_Complete' to boolean, it will be one-hot encoded by the pipeline
            input_df['Documents_Complete'] = input_df['Documents_Complete'].astype(bool)
        except ValueError as ve:
            app.logger.error(f"Data type conversion error: {ve}")
            return jsonify({'error': f'Invalid data type for a numerical/boolean field: {ve}'}), 400


        # The loaded `model_pipeline` handles all preprocessing (scaling, one-hot encoding)
        # and then makes the prediction.
        prediction = model_pipeline.predict(input_df)
        prediction_proba = model_pipeline.predict_proba(input_df)

        # Return results
        result = {
            'prediction': int(prediction[0]), # 0 (Not Approved) or 1 (Approved)
            'probability_not_approved': round(prediction_proba[0][0], 4),
            'probability_approved': round(prediction_proba[0][1], 4)
        }
        app.logger.info(f"Prediction made successfully: {result}")
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

# --- How to run the Flask app ---
if __name__ == '__main__':
    # BEFORE RUNNING THIS FLASK APP:
    # 1. Ensure you have the updated training script (Step 1 above).
    # 2. Run the updated training script to generate 'random_forest_pipeline.pkl'.
    # 3. Save this Flask code as 'app.py' in the same directory as 'random_forest_pipeline.pkl'.
    # 4. Install necessary libraries: pip install Flask scikit-learn pandas joblib
    # 5. Open your terminal in that directory and run: python app.py
    app.run(debug=True, host='0.0.0.0', port=5000)
    # debug=True allows for auto-reloading and better error messages in development.
    # host='0.0.0.0' makes the server accessible from other machines on your network.