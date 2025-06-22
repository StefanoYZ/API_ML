import os
import csv
from datetime import datetime

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'log_registros.csv')


def guardar_prediccion(dato: dict, pred: int, proba: float):
    """
    Guarda los datos originales ingresados por el usuario en un CSV legible.

    Args:
        dato (dict): Datos originales ingresados por el usuario.
        pred (int): Resultado de la predicción (0 o 1).
        proba (float): Probabilidad asociada a la predicción.
    """
    dato_guardado = dato.copy()
    dato_guardado["Tiene Depresion"] = "Sí" if pred == 1 else "No"
    dato_guardado["Probabilidad"] = round(proba, 4)
    dato_guardado["Fecha"] = datetime.now().isoformat()

    archivo_existe = os.path.exists(DATA_PATH)

    with open(DATA_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=dato_guardado.keys())

        if not archivo_existe:
            writer.writeheader()

        writer.writerow(dato_guardado)
