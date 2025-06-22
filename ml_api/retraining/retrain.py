import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

# --- Configuración ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_CSV = os.path.join(BASE_DIR, 'data', 'log_registros.csv')
MODELO_DIR = os.path.join(BASE_DIR, 'model')
COLUMNA_OBJETIVO = 'Tiene Depresion'
MIN_REGISTROS = 700
UMBRAL_DRIFT = 0.5
LOG_PATH = os.path.join(BASE_DIR, 'retraining', 'retrain_log.txt')
DATASET_ORIGINAL = os.path.join(BASE_DIR, 'dataset_original.csv')

# --- Columnas originales del modelo (orden exacto) ---
columnas_modelo_original = [
    "Age", "Academic Pressure", "Work Pressure", "CGPA", "Study Satisfaction", 
    "Job Satisfaction", "Sleep Duration", "Work/Study Hours", "Financial Stress", 
    "Gender_Male", "Dietary Habits_Moderate", "Dietary Habits_Others", 
    "Dietary Habits_Unhealthy", "Have you ever had suicidal thoughts ?_Yes", 
    "Family History of Mental Illness_Yes"
]

# --- Preprocesamiento idéntico al entrenamiento original ---
def preprocesar(df_raw):
    df = df_raw.copy()

    map_sueno = {
        'Less than 5 hours': 4.5,
        '5-6 hours': 5.5,
        '7-8 hours': 7.5,
        'More than 8 hours': 9.0,
        'Others': 4
    }
    df['Sleep Duration'] = df['Sleep Duration'].map(map_sueno)

    for col in ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']:
        df[col] = df[col].astype(str).str.strip().str.capitalize()

    categorical_cols = ['Gender', 'Profession', 'Dietary Habits',
                        'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    for col in columnas_modelo_original:
        if col not in df.columns:
            df[col] = 0

    df = df[columnas_modelo_original]
    return df

def detectar_drift(df_original, df_nuevo, columnas):
    columnas_con_drift = []
    for col in columnas:
        if col not in df_original.columns or col not in df_nuevo.columns:
            continue
        col1 = pd.to_numeric(df_original[col], errors='coerce').dropna()
        col2 = pd.to_numeric(df_nuevo[col], errors='coerce').dropna()
        if len(col2) < 50:
            continue
        _, p = ks_2samp(col1, col2)
        if p < UMBRAL_DRIFT:
            columnas_con_drift.append(col)
    return columnas_con_drift

def reentrenar_modelo():
    if not os.path.exists(LOG_CSV):
        print("❌ No se encuentra log_registros.csv")
        return

    df_raw = pd.read_csv(LOG_CSV)
    if len(df_raw) < MIN_REGISTROS:
        print("No hay suficientes registros")
        return

    columnas_usables = [
        'Gender', 'Age', 'Profession', 'Academic Pressure', 'Work Pressure', 'CGPA',
        'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration', 'Dietary Habits',
        'Have you ever had suicidal thoughts ?', 'Work/Study Hours',
        'Financial Stress', 'Family History of Mental Illness'
    ]
    df = df_raw[columnas_usables + [COLUMNA_OBJETIVO]].copy()
    df_proc = preprocesar(df)

    # Forzar orden y tipo
    df_proc = df_proc[columnas_modelo_original].copy()
    assert isinstance(df_proc, pd.DataFrame), "❌ df_proc no es DataFrame"

    y = df[COLUMNA_OBJETIVO].apply(lambda x: 1 if str(x).strip().lower() == 'sí' else 0)

    if not os.path.exists(DATASET_ORIGINAL):
        print("❌ No se encuentra dataset_original.csv para drift.")
        return

    df_original = pd.read_csv(DATASET_ORIGINAL)
    columnas_con_drift = detectar_drift(df_original, df_proc, columnas_modelo_original)
    if not columnas_con_drift:
        print("✅ No se detectó drift.")
        return

    print("⚠️ Drift detectado en:")
    for col in columnas_con_drift:
        print(f" - {col}")

    columnas_numericas = [
        'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
        'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration',
        'Work/Study Hours', 'Financial Stress'
    ]

    scaler = StandardScaler()
    df_proc[columnas_numericas] = scaler.fit_transform(df_proc[columnas_numericas])

    model_log = LogisticRegression(
        C=1, max_iter=100, penalty='l2',
        solver='liblinear', class_weight='balanced',
        random_state=42
    )
    model_log.fit(df_proc, y)

    # Guardar artefactos
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    joblib.dump(model_log, os.path.join(MODELO_DIR, 'modelo_TEST.pkl'))
    joblib.dump(scaler, os.path.join(MODELO_DIR, 'scaler_TEST.pkl'))

    with open(os.path.join(MODELO_DIR, 'columnas_modelo_TEST.json'), 'w') as f:
        json.dump(df_proc.columns.tolist(), f)

    with open(LOG_PATH, 'a', encoding='utf-8') as logf:
        logf.write(f"[{timestamp}] Drift: {', '.join(columnas_con_drift)} | Registros usados: {len(df_proc)}\n")

    print("✅ Modelo reentrenado y guardado correctamente.")

if __name__ == '__main__':
    reentrenar_modelo()
