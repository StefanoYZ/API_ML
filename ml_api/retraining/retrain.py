import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REGISTROS_PATH = os.path.join(BASE_DIR, 'data', 'registros.csv')
MODELO_DIR = os.path.join(BASE_DIR, 'model')
COLUMNA_OBJETIVO = 'prediccion'
MIN_REGISTROS = 120
UMBRAL_DRIFT = 0.05
LOG_PATH = os.path.join(BASE_DIR, 'retraining', 'retrain_log.txt')
DATASET_ORIGINAL = os.path.join(BASE_DIR, 'dataset_original.csv')

# -----------------------------
# FUNCIONES
# -----------------------------
def detectar_drift(df_original, df_nuevo, columnas):
    columnas_con_drift = []
    for col in columnas:
        col1 = pd.to_numeric(df_original[col], errors='coerce').dropna()
        col2 = pd.to_numeric(df_nuevo[col], errors='coerce').dropna()
        if len(col2) < 50:
            continue
        stat, p = ks_2samp(col1, col2)
        if p < UMBRAL_DRIFT:
            columnas_con_drift.append(col)
    return columnas_con_drift

def reentrenar_modelo_prueba():
    if not os.path.exists(REGISTROS_PATH):
        print("❌ No existe registros.csv")
        return

    df = pd.read_csv(REGISTROS_PATH)
    if len(df) < MIN_REGISTROS:
        print(f"ℹ️ Solo hay {len(df)} registros. Se requieren al menos {MIN_REGISTROS}.")
        return

    if not os.path.exists(DATASET_ORIGINAL):
        print("❌ No se encontró dataset_original.csv")
        return

    df_original = pd.read_csv(DATASET_ORIGINAL)
    columnas_comunes = [col for col in df.columns if col in df_original.columns and col != COLUMNA_OBJETIVO]
    columnas_con_drift = detectar_drift(df_original, df, columnas_comunes)

    if not columnas_con_drift:
        print("✅ No se detectó drift. No se reentrenará el modelo.")
        return

    print("⚠️ Drift detectado en:")
    for col in columnas_con_drift:
        print(f" - {col}")

    # -----------------------------
    # ENTRENAMIENTO
    # -----------------------------
    X = df[columnas_comunes]
    y = df[COLUMNA_OBJETIVO]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # -----------------------------
    # GUARDAR COMO ARCHIVOS DE PRUEBA
    # -----------------------------
    joblib.dump(model, os.path.join(MODELO_DIR, 'modelo_TEST.pkl'))
    joblib.dump(scaler, os.path.join(MODELO_DIR, 'scaler_TEST.pkl'))

    with open(os.path.join(MODELO_DIR, 'columnas_modelo_TEST.json'), 'w') as f:
        json.dump(columnas_comunes, f)

    with open(LOG_PATH, 'a') as logf:
        logf.write(f"[{timestamp}] Reentrenamiento de prueba con {len(df)} registros. Drift en: {', '.join(columnas_con_drift)}\n")

    print("✅ Reentrenamiento de prueba completado. Modelos guardados como *_TEST.pkl")

# -----------------------------
# EJECUCIÓN
# -----------------------------
if __name__ == '__main__':
    reentrenar_modelo_prueba()
