import pandas as pd

def load_data():
    return pd.read_csv('hosp/prescriptions.csv.gz')