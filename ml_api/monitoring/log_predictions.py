import pandas as pd
from datetime import datetime
import os

def guardar_prediccion(df_procesado: pd.DataFrame, pred: int, proba: float, ruta_csv='data/registros.csv'):
    """
    Guarda el dato procesado (con columnas consistentes) y su predicción.

    Args:
        df_procesado (pd.DataFrame): Dato ya preprocesado, con columnas correctas.
        pred (int): Resultado de la predicción (0 = No, 1 = Sí).
        proba (float): Probabilidad estimada.
        ruta_csv (str): Ruta del archivo de registros.
    """
    # Agregar columnas de resultado
    df_procesado = df_procesado.copy()
    df_procesado['prediccion'] = pred
    df_procesado['probabilidad'] = round(proba, 4)
    df_procesado['timestamp'] = datetime.now().isoformat()

    # Guardar
    if os.path.exists(ruta_csv):
        df_antiguo = pd.read_csv(ruta_csv)
        df_total = pd.concat([df_antiguo, df_procesado], ignore_index=True)
    else:
        df_total = df_procesado

    df_total.to_csv(ruta_csv, index=False)
