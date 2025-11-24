from transformers import pipeline
import mlflow

# Load GPT text-generation pipeline once
text_generator = pipeline("text-generation", model="gpt2")

def create_patient_from_prescriptions(prescriptions_data):
    common_meds = prescriptions_data['drug'].value_counts().head(3).index.tolist()
    medications_text = ", ".join(common_meds[:2])
    return {
        'age': 65,
        'gender': 'M',
        'past_history': 'Hypertension, Diabetes',
        'medications': medications_text,
        'allergies': 'None documented',
        'blood_pressure': '150/90',
        'heart_rate': 88,
        'temperature': 98.6,
        'respiratory_rate': 18,
        'oxygen_saturation': 98
    }

def generate_discharge_summary_gpt(patient_data, extracted_symptoms):
    symptoms_list = [s['word'] for s in extracted_symptoms]
    symptoms_text = ", ".join(symptoms_list)
    patient_info = create_patient_from_prescriptions(patient_data)

    prompt = f"""
Generate a discharge summary for a patient with the following details:
Age: {patient_info['age']}
Gender: {patient_info['gender']}
Past Medical History: {patient_info['past_history']}
Current Medications: {patient_info['medications']}
Allergies: {patient_info['allergies']}
Symptoms: {symptoms_text}
Discharge Summary:
"""

    generated = text_generator(prompt, max_length=300, num_return_sequences=1)
    summary = generated[0]['generated_text']

    mlflow.log_metric("discharge_summary_length", len(summary))
    mlflow.log_text(summary, "generated_discharge_summary.txt")
    return summary
