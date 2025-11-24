from transformers import pipeline
import mlflow

def extract_entity(patient_data):
    nr = pipeline(
        "token-classification",
        model="samrawal/bert-base-uncased_clinical-ner",
        aggregation_strategy="first"
    )
    entities = nr(patient_data)
    medical_symptoms = [e for e in entities if e['entity_group'] == 'problem']

    mlflow.log_param("ner_model", "bert-base-uncased_clinical-ner")
    mlflow.log_param("input_text_length", len(patient_data))
    mlflow.log_metric("symptoms_extracted", len(medical_symptoms))
    mlflow.log_metric("average_confidence",
                      sum([e['score'] for e in medical_symptoms]) / len(medical_symptoms) if medical_symptoms else 0)
    return medical_symptoms