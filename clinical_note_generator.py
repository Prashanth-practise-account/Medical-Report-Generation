import mlflow
import discharge_summary as ds_module
from transformers import pipeline
def generate_clinical_note(patient_data, extracted_symptoms):
    text_generator = pipeline("text-generation",model="gpt2")
    symptoms_list = [s['word'] for s in extracted_symptoms]
    symptoms_text = ", ".join(symptoms_list)

    patient_info = ds_module.create_patient_from_prescriptions(patient_data)

    clinical_note = f"""
    SUBJECTIVE:
    {patient_info['age']} year old {patient_info['gender']} presents with {symptoms_text}.
    Past medical history: {patient_info['past_history']}.
    Current medications: {patient_info['medications']}.
    Allergies: {patient_info['allergies']}.
    
    OBJECTIVE:
    Vital Signs: BP {patient_info['blood_pressure']}, HR {patient_info['heart_rate']}, 
    Temp {patient_info['temperature']}°F, RR {patient_info['respiratory_rate']}, 
    SpO2 {patient_info['oxygen_saturation']}%.
    Physical Exam: Appropriate for chief complaint.
    
    ASSESSMENT:
    Based on symptoms: {symptoms_text}
    
    PLAN:
        1. Further evaluation for {symptoms_text}
        2. Review current medications: {patient_info['medications']}
        3. Follow up as needed
        """
    generated = text_generator(clinical_note, max_length=300, num_return_sequences=1)
    Clinical_notes = generated[0]['generated_text']
    mlflow.log_metric("clinical_note_length", len(Clinical_notes))
    mlflow.log_text(Clinical_notes, "generated_clinical_note.txt")
    return Clinical_notes