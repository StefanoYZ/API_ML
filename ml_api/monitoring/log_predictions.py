import pandas as pd
from datetime import datetime
import os

def guardar_prediccion(df_procesado: pd.DataFrame, pred: int, proba: float, ruta_csv='data/registros.csv'):
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
