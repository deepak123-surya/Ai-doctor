# from flask import Flask, request, render_template, jsonify  # Import jsonify
# import numpy as np
# import pandas as pd
# import pickle
#
#
# # flask app
# app = Flask(__name__)
#
#
#
# # load databasedataset===================================
# sym_des = pd.read_csv("dataset/symtoms_df.csv")
# precautions = pd.read_csv("dataset/precautions_df.csv")
# workout = pd.read_csv("dataset/workout_df.csv")
# description = pd.read_csv("dataset/description.csv")
# medications = pd.read_csv('dataset/medications.csv')
# diets = pd.read_csv("dataset/diets.csv")
#
#
# # load model===========================================
# svc = pickle.load(open('models/svc.pkl','rb'))
#
#
# #============================================================
# # custome and helping functions
# #==========================helper funtions================
# def helper(dis):
#     desc = description[description['Disease'] == dis]['Description']
#     desc = " ".join([w for w in desc])
#
#     pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
#     pre = [col for col in pre.values]
#
#     med = medications[medications['Disease'] == dis]['Medication']
#     med = [med for med in med.values]
#
#     die = diets[diets['Disease'] == dis]['Diet']
#     die = [die for die in die.values]
#
#     wrkout = workout[workout['disease'] == dis] ['workout']
#
#
#     return desc,pre,med,die,wrkout
#
# symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
# diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
#
# # Model Prediction function
# def get_predicted_value(patient_symptoms):
#     input_vector = np.zeros(len(symptoms_dict))
#     for item in patient_symptoms:
#         input_vector[symptoms_dict[item]] = 1
#     return diseases_list[svc.predict([input_vector])[0]]
#
#
#
#
# # creating routes========================================
#
#
# @app.route("/")
# def index():
#     return render_template("index.html")
#
# # Define a route for the home page
# @app.route('/predict', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         symptoms = request.form.get('symptoms')
#         # mysysms = request.form.get('mysysms')
#         # print(mysysms)
#         print(symptoms)
#         if symptoms =="Symptoms":
#             message = "Please either write symptoms or you have written misspelled symptoms"
#             return render_template('index.html', message=message)
#         else:
#
#             # Split the user's input into a list of symptoms (assuming they are comma-separated)
#             user_symptoms = [s.strip() for s in symptoms.split(',')]
#             # Remove any extra characters, if any
#             user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
#             predicted_disease = get_predicted_value(user_symptoms)
#             dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
#
#             my_precautions = []
#             for i in precautions[0]:
#                 my_precautions.append(i)
#
#             return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
#                                    my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
#                                    workout=workout)
#
#     return render_template('index.html')
#
#
#
# # about view funtion and path
# @app.route('/about')
# def about():
#     return render_template("about.html")
# # contact view funtion and path
# @app.route('/contact')
# def contact():
#     return render_template("contact.html")
#
# # developer view funtion and path
# @app.route('/developer')
# def developer():
#     return render_template("developer.html")
#
# # about view funtion and path
# @app.route('/blog')
# def blog():
#     return render_template("blog.html")
#
#
# if __name__ == '__main__':
#
#     app.run(debug=True)

from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import re

# flask app
app = Flask(__name__)

# load databasedataset===================================
sym_des = pd.read_csv("dataset/symtoms_df.csv")
precautions = pd.read_csv("dataset/precautions_df.csv")
workout = pd.read_csv("dataset/workout_df.csv")
description = pd.read_csv("dataset/description.csv")
medications = pd.read_csv('dataset/medications.csv')
diets = pd.read_csv("dataset/diets.csv")

# load model===========================================
svc = pickle.load(open('models/svc.pkl', 'rb'))


# ============================================================
# custom and helping functions
# ==========================helper functions================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout


# Your symptoms dictionary
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
                 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
                 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
                 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
                 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
                 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
                 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28,
                 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
                 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36,
                 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40,
                 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44,
                 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47,
                 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51,
                 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
                 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
                 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
                 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
                 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70,
                 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73,
                 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76,
                 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_paint': 79, 'muscle_weakness': 80,
                 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84,
                 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87,
                 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90,
                 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93,
                 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97,
                 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
                 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
                 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
                 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
                 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
                 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
                 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119,
                 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123,
                 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
                 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis',
                 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ',
                 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine',
                 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice',
                 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
                 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E',
                 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia',
                 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins',
                 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis',
                 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne',
                 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}


# Create a reverse lookup dictionary for symptom matching
def create_symptom_variations():
    """Create variations of symptom names for better matching"""
    variations = {}

    for symptom in symptoms_dict.keys():
        # Original with underscores
        key = symptom
        variations[key] = symptom

        # Replace underscores with spaces
        spaced = symptom.replace('_', ' ')
        variations[spaced] = symptom

        # Lowercase version
        variations[symptom.lower()] = symptom

        # Lowercase with spaces
        variations[spaced.lower()] = symptom

        # Remove extra spaces
        clean_spaced = ' '.join(spaced.split())
        variations[clean_spaced] = symptom

        # Title case
        variations[spaced.title()] = symptom

    return variations


symptom_variations = create_symptom_variations()


def normalize_symptom(user_symptom):
    """
    Normalize user symptom to match dictionary keys
    """
    if not user_symptom:
        return None

    # Clean the input
    user_symptom = str(user_symptom).strip()

    # Try exact match in variations first
    if user_symptom in symptom_variations:
        return symptom_variations[user_symptom]

    # Try lowercase match
    if user_symptom.lower() in symptom_variations:
        return symptom_variations[user_symptom.lower()]

    # Replace spaces with underscores and try
    underscored = user_symptom.replace(' ', '_')
    if underscored in symptom_variations:
        return symptom_variations[underscored]

    # Try to find partial match
    for dict_symptom in symptoms_dict.keys():
        # Clean both for comparison
        clean_user = user_symptom.lower().replace(' ', '_')
        clean_dict = dict_symptom.lower()

        # Check if one contains the other
        if (clean_user in clean_dict or clean_dict in clean_user):
            return dict_symptom

    return None


# Enhanced Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    matched_symptoms = []
    unmatched_symptoms = []

    for item in patient_symptoms:
        normalized_symptom = normalize_symptom(item)

        if normalized_symptom and normalized_symptom in symptoms_dict:
            input_vector[symptoms_dict[normalized_symptom]] = 1
            matched_symptoms.append(f"{item} → {normalized_symptom}")
        else:
            unmatched_symptoms.append(item)
            print(f"Warning: Symptom '{item}' not found in dictionary")

    # Debug info
    if matched_symptoms:
        print(f"Matched symptoms: {matched_symptoms}")
    if unmatched_symptoms:
        print(f"Unmatched symptoms: {unmatched_symptoms}")

    # Make prediction
    if np.sum(input_vector) > 0:  # Only predict if at least one symptom matched
        prediction = svc.predict([input_vector])[0]
        return diseases_list[prediction]
    else:
        return "Insufficient data for prediction"


# Alternative: Simple symptom matcher for common symptoms
def simple_symptom_matcher(user_symptom):
    """Simple mapping for common symptom variations"""
    common_mappings = {
        # User input: Dictionary key
        'skin rash': 'skin_rash',
        'skin rashes': 'skin_rash',
        'rash': 'skin_rash',
        'fever': 'high_fever',
        'head ache': 'headache',
        'head pain': 'headache',
        'stomach ache': 'stomach_pain',
        'abdominal pain': 'abdominal_pain',
        'belly pain': 'abdominal_pain',
        'joint pain': 'joint_pain',
        'muscle pain': 'muscle_pain',
        'chest pain': 'chest_pain',
        'back pain': 'back_pain',
        'neck pain': 'neck_pain',
        'knee pain': 'knee_pain',
        'runny nose': 'runny_nose',
        'blocked nose': 'congestion',
        'nasal congestion': 'congestion',
        'sore throat': 'throat_irritation',
        'shortness of breath': 'breathlessness',
        'difficulty breathing': 'breathlessness',
        'weight loss': 'weight_loss',
        'weight gain': 'weight_gain',
        'loss of appetite': 'loss_of_appetite',
        'increased appetite': 'increased_appetite',
        'yellow skin': 'yellowish_skin',
        'yellow eyes': 'yellowing_of_eyes',
        'dark urine': 'dark_urine',
        'yellow urine': 'yellow_urine',
        'blurred vision': 'blurred_and_distorted_vision',
        'vision problems': 'visual_disturbances',
        'mood swings': 'mood_swings',
        'cold hands': 'cold_hands_and_feets',
        'cold feet': 'cold_hands_and_feets',
        'fast heartbeat': 'fast_heart_rate',
        'heart palpitations': 'palpitations',
        'swollen legs': 'swollen_legs',
        'puffy face': 'puffy_face_and_eyes',
        'swollen eyes': 'puffy_face_and_eyes',
    }

    # Clean user input
    user_symptom = user_symptom.lower().strip()

    # Check direct mapping
    if user_symptom in common_mappings:
        return common_mappings[user_symptom]

    # Check if any key is in the user input
    for key, value in common_mappings.items():
        if key in user_symptom:
            return value

    return None


# Alternative prediction function using simple matcher
def get_predicted_value_enhanced(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    matched = []

    for item in patient_symptoms:
        # Try simple matcher first
        matched_symptom = simple_symptom_matcher(item)

        if not matched_symptom:
            # Fall back to normalization
            matched_symptom = normalize_symptom(item)

        if matched_symptom and matched_symptom in symptoms_dict:
            input_vector[symptoms_dict[matched_symptom]] = 1
            matched.append(f"{item} → {matched_symptom}")
        else:
            print(f"Could not match: {item}")

    print(f"Successfully matched: {matched}")

    if np.sum(input_vector) > 0:
        prediction = svc.predict([input_vector])[0]
        return diseases_list[prediction]
    else:
        return "Please provide valid symptoms"


# creating routes========================================
@app.route("/")
def index():
    return render_template("index.html")


# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms', '').strip()

        if not symptoms or symptoms.lower() == 'symptoms':
            message = "Please enter your symptoms (e.g., fever, headache, cough)"
            return render_template('index.html', message=message)

        try:
            # Clean and split symptoms
            user_symptoms = []
            for s in symptoms.split(','):
                s_clean = s.strip()
                if s_clean:
                    user_symptoms.append(s_clean)

            if not user_symptoms:
                message = "Please enter valid symptoms"
                return render_template('index.html', message=message)

            print(f"User symptoms: {user_symptoms}")

            # Use enhanced prediction function
            predicted_disease = get_predicted_value_enhanced(user_symptoms)

            if predicted_disease.startswith("Please") or predicted_disease.startswith("Insufficient"):
                return render_template('index.html',
                                       message=predicted_disease,
                                       symptoms_entered=symptoms)

            # Get additional information
            dis_des, precautions_list, medications_list, rec_diet, workout_list = helper(predicted_disease)

            # Process precautions
            my_precautions = []
            if precautions_list and len(precautions_list) > 0:
                for i in precautions_list[0]:
                    if i and str(i).lower() != 'nan':
                        my_precautions.append(i)

            # Process medications
            medications = []
            if medications_list and len(medications_list) > 0:
                for med in medications_list:
                    if med and str(med).lower() != 'nan':
                        medications.append(med)

            # Process diet
            my_diet = []
            if rec_diet and len(rec_diet) > 0:
                for diet in rec_diet:
                    if diet and str(diet).lower() != 'nan':
                        my_diet.append(diet)

            # Process workout
            workout = []
            if not workout_list.empty:
                workout = workout_list.tolist()

            return render_template('index.html',
                                   predicted_disease=predicted_disease,
                                   dis_des=dis_des,
                                   my_precautions=my_precautions,
                                   medications=medications,
                                   my_diet=my_diet,
                                   workout=workout,
                                   symptoms_entered=symptoms)

        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html',
                                   message=f"An error occurred: {str(e)}")

    return render_template('index.html')


# API endpoint for voice input
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        symptoms_text = data.get('symptoms', '')

        if not symptoms_text:
            return jsonify({'error': 'No symptoms provided'}), 400

        # Split symptoms
        user_symptoms = [s.strip() for s in symptoms_text.split(',') if s.strip()]

        if not user_symptoms:
            return jsonify({'error': 'No valid symptoms provided'}), 400

        # Get prediction
        predicted_disease = get_predicted_value_enhanced(user_symptoms)

        if predicted_disease.startswith("Please") or predicted_disease.startswith("Insufficient"):
            return jsonify({'error': predicted_disease}), 400

        # Get additional info
        dis_des, precautions_list, medications_list, rec_diet, workout_list = helper(predicted_disease)

        # Format response
        response = {
            'disease': predicted_disease,
            'description': dis_des,
            'precautions': precautions_list[0].tolist() if precautions_list and len(precautions_list) > 0 else [],
            'medications': medications_list if medications_list else [],
            'diet': rec_diet if rec_diet else [],
            'workout': workout_list.tolist() if not workout_list.empty else []
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# about view function and path
@app.route('/about')
def about():
    return render_template("about.html")


# contact view function and path
@app.route('/contact')
def contact():
    return render_template("contact.html")


# developer view function and path
@app.route('/developer')
def developer():
    return render_template("developer.html")


# blog view function and path
@app.route('/blog')
def blog():
    return render_template("blog.html")


if __name__ == '__main__':
    app.run(debug=True)