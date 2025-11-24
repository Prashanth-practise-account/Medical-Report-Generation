import mlflow
import data_loader
import ner_entity
import clinical_note_generator as cng
import discharge_summary as ds
from datetime import datetime
import pandas as pd
import streamlit as st
st.title("GenAI Clinical Notes & Discharge Summary")
input_text = st.text_input("Enter patient symptoms")
if st.button("Generate Notes"):
    try:
        mlflow.set_experiment("Clinical_Notes_Discharge_Summary")
        with mlflow.start_run(run_name="Full_Pipeline"):
            data = data_loader.load_data()
            print("PRESCRIPTIONS DATA:")
            print(data.info())
            print(f"\nDataset shape: {data.shape}")
            print(f"\nCommon medications: {data['drug'].value_counts().head(2)}")
            st.write("PRESCRIPTIONS DATA:")
            st.write(data.head())
            # input_text = "chest pain and shortness of breath"
            mlflow.log_param("input_symptoms", input_text)
            mlflow.log_metric("dataset_size", len(data))
            mlflow.log_metric("unique_medications", data['drug'].nunique())

            entities = ner_entity.extract_entity(input_text)
            print(f"\nEXTRACTED SYMPTOMS: {entities}")
            st.write("EXTRACTED SYMPTOMS:", entities)

            clinical_note = cng.generate_clinical_note(data, entities)
            print("\nGENERATED CLINICAL NOTE:\n", clinical_note)
            st.subheader("Generated Clinical Note")
            st.text(clinical_note)

            discharge_sum = ds.generate_discharge_summary_gpt(data, entities)
            print("\nGENERATED DISCHARGE SUMMARY:\n", discharge_sum)
            st.subheader("Generated Discharge Summary")
            st.text(discharge_sum)

            clinical_data = [{
                'patient_id': 'PATIENT_001',
                'age': 65,
                'gender': 'M',
                'symptoms': input_text,
                'clinical_note': clinical_note,
                'discharge_summary': discharge_sum,
                'medications': "Insulin, 0.9% Sodium Chloride",
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }]
            df = pd.DataFrame(clinical_data)
            df.to_csv('dataset/my_patient_records.csv', index=False)
            print("SAVED to: my_patient_records.csv")
            print("CSV Contents:\n", df)
            st.success("Saved to my_patient_records.csv")
            st.write(df)
            # Log CSV as artifact in MLflow
            mlflow.log_metric("records_generated", 1)
            mlflow.log_artifact('dataset/my_patient_records.csv')

    except Exception as e:
        print(f"Error: {e}")
        st.error(f"Error: {e}")
