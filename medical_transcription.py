import whisper
import mlflow

model = whisper.load_model("tiny")


def medical_transcription(audio_file):
    result = model.transcribe(audio_file)
    transcript = f"""
                MEDICAL TRANSCRIPTION
                Doctor said:
                "{result['text']}"
                Interpretation:
                Transcribed doctor dictation for patient record.
            """

    if mlflow.active_run() is None:
        mlflow.start_run(run_name="Medical_Transcription")

    mlflow.log_text(transcript, "generated_medical_transcription.txt")

    return transcript
