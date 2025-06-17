
from flask import jsonify
import joblib
import pandas as pd
import shap
from google import genai


client = genai.Client(api_key="AIzaSyAanSNKSegZw7pvCbzv8JVrgfHUKvwUdcc")

def preprocess_data(df):
    # Normalize Oui/Non columns to boolean
    bool_map = {'Oui': True, 'Non': False}
    for col in df.columns:
        if df[col].dropna().isin(bool_map.keys()).all():
            df[col] = df[col].map(bool_map)
    
    # Ensure proper dtypes
    bool_columns = ['compte_bancaire_actif', 'incident_bancaire', 'historique_credit']
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    return df

def load_model(app, pipeline_filename):
    try:
        with open(pipeline_filename, "rb") as f:
            model_pipeline = joblib.load(f)
        app.logger.info(f"Model pipeline '{pipeline_filename}' loaded successfully!")
    except FileNotFoundError:
        app.logger.error(
            f"Error: Model pipeline '{pipeline_filename}' not found. Make sure it's in the same directory as app.py."
        )
    except Exception as e:
        app.logger.error(
            f"Error loading model pipeline '{pipeline_filename}': {e}", exc_info=True
        )
    return model_pipeline


def predict_model(app, data):
    pipeline_filename = "model/model.pkl"
    model_pipeline = load_model(app, pipeline_filename)

    if model_pipeline is None:
        return (
            jsonify({"error": "Model pipeline not loaded. Cannot make predictions."}),
            500,
        )

    input_df = pd.DataFrame([data], columns=data)
    input_df = preprocess_data(input_df)
    prediction = model_pipeline.predict(input_df)
    prediction_proba = model_pipeline.predict_proba(input_df)
    # Transform input for SHAP
    X_transformed = model_pipeline.named_steps["preprocessor"].transform(input_df)
    # Convert to dense if sparse
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    # Create SHAP explainer on classifier
    explainer = shap.TreeExplainer(model_pipeline.named_steps["classifier"])
    explanation = explainer(X_transformed)
    predicted_class = int(prediction[0])
    feature_names = input_df.columns
    shap_value_array = explanation.values[0, :, predicted_class]

    feature_importance = sorted(
        zip(feature_names, shap_value_array),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    top_features = [
        {"feature": name, "impact": round(val, 3)} for name, val in feature_importance[:5]
    ]
    recommendations = []
    for feature, impact in feature_importance:
        if impact < 0:
            recommendations.append(f"Try increasing {feature}")
    
    xx= ({
        "prediction": int(prediction[0]),
        "probability": round(prediction_proba[0][1], 3),
        "top_features": top_features
    })
    
    prompt = f"""
    You are a financial advisor. Review the user Input data below and provide 3-5 short, clear, and actionable recommendations to improve their credit score. Avoid explanations—just list the steps.
    Focus on practical actions the user can take to improve their chances of credit approval.
    User Input Data:
    {(data)}

    Additional Context:
    {xx}
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    ).text
    app.logger.info(f"Generated recommendations: {response}")
    return jsonify({
        "prediction": int(prediction[0]),
        "probability": round(prediction_proba[0][1], 3),
        "top_features": top_features,
        "recommendations": response
    })
    
