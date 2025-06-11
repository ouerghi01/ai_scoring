from flask import jsonify
import joblib
import pandas as pd

def load_model(app, pipeline_filename):
    try:
        with open(pipeline_filename, 'rb') as f:
            model_pipeline = joblib.load(f)
        app.logger.info(f"Model pipeline '{pipeline_filename}' loaded successfully!")
    except FileNotFoundError:
        app.logger.error(f"Error: Model pipeline '{pipeline_filename}' not found. Make sure it's in the same directory as app.py.")
    except Exception as e:
        app.logger.error(f"Error loading model pipeline '{pipeline_filename}': {e}", exc_info=True)
    return model_pipeline
def predict_score(app,data):
    
    model_pipeline = None
    pipeline_filename = 'model/model.pkl'


    model_pipeline = load_model(app, pipeline_filename)
    if model_pipeline is None:
        return jsonify({'error': 'Model pipeline not loaded. Cannot make predictions.'}), 500
    input_df = pd.DataFrame([data], columns=data)       
    prediction = model_pipeline.predict(input_df)
    prediction_proba = model_pipeline.predict_proba(input_df)
    return prediction,prediction_proba