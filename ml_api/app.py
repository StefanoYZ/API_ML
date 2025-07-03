import sys
import os
import numpy as np
import pandas as pd
from flask_cors import CORS
from flask import send_file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
import pickle
import json
from utils.preprocessing import procesar_dato
from monitoring.log_predictions import guardar_prediccion
from monitoring.log_registros import guardar_registro
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar el modelo entrenado
with open(os.path.join(BASE_DIR, "model", "modelo.pkl"), "rb") as f:
    modelo = pickle.load(f)

# Cargar el escalador
with open(os.path.join(BASE_DIR, "model", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# Cargar las columnas esperadas
with open(os.path.join(BASE_DIR, "model", "columnas_modelo.json"), "r") as f:
    columnas_modelo = json.load(f)

app = Flask(__name__)
CORS(app)
@app.route('/')
def index():
    return "✅ API de predicción de depresión activa."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Leer JSON enviado en el cuerpo de la petición
        nuevo_dato = request.get_json()

        # Validar que se haya recibido algo
        if not nuevo_dato:
            return jsonify({"error": "No se recibió ningún dato"}), 400

        # Preprocesar
        df_procesado = procesar_dato(nuevo_dato, scaler, columnas_modelo)

        # Predicción
        probabilidad = modelo.predict_proba(df_procesado)[0][1]
        prediccion = int(modelo.predict(df_procesado)[0])
        # Guardado de la predicción
        guardar_prediccion(df_procesado, prediccion, probabilidad)
        guardar_registro(nuevo_dato,prediccion,probabilidad)
        # Respuesta JSON
        return jsonify({
            "tiene_depresion": bool(prediccion),
            "probabilidad": round(probabilidad, 4),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/descargar-registros', methods=['GET'])
def descargar_registros():
    ruta_csv = os.path.join(BASE_DIR, 'data', 'log_registros.csv')
    if os.path.exists(ruta_csv):
        return send_file(ruta_csv, as_attachment=True)
    else:
        return jsonify({"error": "El archivo de registros no existe"}), 404

# Solo si ejecutas directamente
if __name__ == "__main__":
    app.run(debug=True)
