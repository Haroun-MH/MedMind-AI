import flask
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

app = flask.Flask(__name__)
CORS(app)

# --- Load model and data --- 
MODEL_PATH = os.path.join('Models', 'Back_up', 'svc_model.pkl')
DATA_PATH = os.path.join('Datasets', 'Training.csv')

# Load the trained SVC model
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the training data to get feature names (symptoms)
training_data = pd.read_csv('Datasets/Training.csv')
all_symptoms = training_data.columns[:-1].tolist() # Exclude 'prognosis'

# Load additional data files
description_df = pd.read_csv('Datasets/description.csv')
precautions_df = pd.read_csv('Datasets/precautions_df.csv')
medications_df = pd.read_csv('Datasets/medications.csv')
workout_df = pd.read_csv('Datasets/workout_df.csv')
symptom_severity_df = pd.read_csv('Datasets/Symptom-severity.csv')

# Pre-process medications data (convert string list to actual list)
import ast
def parse_medications(med_str):
    try:
        return ast.literal_eval(med_str)
    except (ValueError, SyntaxError):
        return [] # Return empty list if parsing fails
medications_df['Medication'] = medications_df['Medication'].apply(parse_medications)

def preprocess_input(natural_language_input, all_symptoms_list):
    """
    Processes natural language input to identify known symptoms and create a binary feature vector.
    Args:
        natural_language_input (str): The user's input string describing symptoms.
        all_symptoms_list (list): A list of all known symptom names (in snake_case format).
    Returns:
        tuple: (np.array feature_vector, list unknown_symptoms_for_model, list identified_raw_symptoms)
    """
    processed_input = natural_language_input.lower()
    input_vector_model = [0] * len(all_symptoms_list)
    identified_raw_symptoms = [] # Symptoms found in the input, in their original (spaced) form for severity lookup
    unknown_symptoms_for_model = [] # For symptoms that might be mentioned but aren't in our list

    for i, symptom_snake_case in enumerate(all_symptoms_list):
        # Convert snake_case symptom from dataset to space-separated for matching in natural language
        symptom_natural_form = symptom_snake_case.replace('_', ' ')
        if symptom_natural_form in processed_input:
            input_vector_model[i] = 1
            identified_raw_symptoms.append(symptom_natural_form) # Store the version with spaces
    
    # For 'unknown_symptoms_for_model', we could try to be smarter, 
    # but for now, if no known symptoms are found, we can't really identify unknowns easily from free text.
    # A simple approach for unknown: if identified_raw_symptoms is empty, maybe the whole input was unknown?
    # This part is tricky without more advanced NLP. We'll keep it simple.
    # The current logic for unknown_symptoms_for_model in the calling function might need adjustment
    # or we can just return an empty list if we only care about *model-relevant* unknowns.
    # For now, let's assume unknown_symptoms_for_model will be populated if a specific word was *expected* but not found.
    # Given the new approach, this list might be less relevant unless we try to parse out all nouns/adjectives etc.
    # Let's return an empty list for unknown_symptoms_for_model for now, as we are directly checking against known symptoms.

    return np.array([input_vector_model]), [], identified_raw_symptoms


@app.route('/predict', methods=['POST'])
def predict():
    if not model or training_data is None or not all_symptoms:
        return flask.jsonify({'error': 'Model, LabelEncoder, or training data not loaded properly. Check server logs.'}), 500

    symptoms_input = flask.request.form.get('symptoms', '')
    if not symptoms_input:
        return flask.jsonify({"error": "No symptoms provided", "symptoms_entered": ""}), 400

    try:
        # Pass all_symptoms (the global list) to preprocess_input
        input_vector, unknown_symptoms_for_model, symptoms_list_raw = preprocess_input(symptoms_input, all_symptoms)

        # if unknown_symptoms_for_model:
        #     # Optionally, inform about unknown symptoms, but still try to predict with known ones
        #     pass

        if not np.any(input_vector):
             return flask.jsonify({
                "error": "No known symptoms provided or processed for the model. Please check your input.",
                "symptoms_entered": symptoms_input,
                "unknown_symptoms_for_model": unknown_symptoms_for_model
            }), 400

        prediction_encoded = model.predict(input_vector)
        predicted_disease = prediction_encoded[0] if isinstance(prediction_encoded, (list, np.ndarray)) else prediction_encoded

        # --- Fetch additional information ---
        description = "Not available"
        desc_row = description_df[description_df['Disease'].str.lower() == predicted_disease.lower()]
        if not desc_row.empty:
            description = desc_row['Description'].iloc[0]

        precautions = []
        prec_row = precautions_df[precautions_df['Disease'].str.lower() == predicted_disease.lower()]
        if not prec_row.empty:
            for i in range(1, 5):
                p = prec_row[f'Precaution_{i}'].iloc[0]
                if pd.notna(p) and str(p).strip():
                    precautions.append(str(p).strip())
        if not precautions: precautions = ["Not available"]

        medications = ["Not available"]
        med_row = medications_df[medications_df['Disease'].str.lower() == predicted_disease.lower()]
        if not med_row.empty:
            meds = med_row['Medication'].iloc[0]
            if isinstance(meds, list) and meds:
                medications = meds
            elif isinstance(meds, str) and meds.strip(): # Should be pre-parsed, but as fallback
                 try:
                     parsed_meds = ast.literal_eval(meds)
                     if parsed_meds: medications = parsed_meds
                 except: pass # Keep default if parsing fails
        
        recommendations = []
        # workout_df has 'disease' column, ensure case-insensitivity
        workout_entries = workout_df[workout_df['disease'].str.lower() == predicted_disease.lower()]
        if not workout_entries.empty:
            recommendations = workout_entries['workout'].tolist()
        if not recommendations: recommendations = ["Not available"]

        symptom_severities = []
        # symptom_severity_df has 'Symptom' (needs to be lowercased and underscore for matching) and 'weight'
        # symptoms_list_raw contains the user's input symptoms, lowercased but with spaces
        for symp_raw in symptoms_list_raw:
            symp_processed_for_severity = symp_raw.replace(' ', '_') # Match format in Symptom-severity.csv
            sev_row = symptom_severity_df[symptom_severity_df['Symptom'].str.lower() == symp_processed_for_severity]
            if not sev_row.empty:
                symptom_severities.append({"symptom": symp_raw, "severity_weight": int(sev_row['weight'].iloc[0])})
            else:
                symptom_severities.append({"symptom": symp_raw, "severity_weight": "Not available"})
        if not symptom_severities and symptoms_list_raw : # if input was there but no severities found
             symptom_severities = [{"symptom": s, "severity_weight": "Not available"} for s in symptoms_list_raw]
        elif not symptoms_list_raw: # No symptoms input
            symptom_severities = []

        return flask.jsonify({
            "symptoms_entered": symptoms_input,
            "predicted_disease": predicted_disease,
            "description": description,
            "precautions": precautions,
            "medications": medications,
            "recommendations": recommendations,
            "symptom_severity": symptom_severities,
            "unknown_symptoms_for_model": unknown_symptoms_for_model
        })
    except Exception as e:
        flask.current_app.logger.error(f"Error during prediction or data fetching: {e}")
        flask.current_app.logger.error(traceback.format_exc())
        return flask.jsonify({
            "error": f"An error occurred: {str(e)}",
            "symptoms_entered": symptoms_input
        }), 500

if __name__ == '__main__':
    # Ensure model and data are loaded before starting the app
    if model is not None and training_data is not None and all_symptoms:
        print("Model and data loaded successfully.")
        app.run(debug=True)
    else:
        print("Failed to load model or data. Application not starting.")