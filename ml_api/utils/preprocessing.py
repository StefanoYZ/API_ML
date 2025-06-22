import pandas as pd

def procesar_dato(nuevo_dato, scaler, columnas_modelo):
    """
    Transforma un diccionario de entrada en un DataFrame listo para predicción.

    Args:
        nuevo_dato (dict): Diccionario con los campos originales del usuario.
        scaler (StandardScaler): Escalador entrenado.
        columnas_modelo (list): Lista de columnas ordenadas esperadas por el modelo.

    Returns:
        pd.DataFrame: Fila procesada y lista para pasar al modelo.
    """

    # Convertir a DataFrame
    df = pd.DataFrame([nuevo_dato])

    # Mapear duración del sueño a valores numéricos
    map_sueno = {
        'Less than 5 hours': 4.5,
        '5-6 hours': 5.5,
        '7-8 hours': 7.5,
        'More than 8 hours': 9.0,
        'Others': 4
    }
    df['Sleep Duration'] = df['Sleep Duration'].map(map_sueno)

    # Normalizar valores categóricos tipo Yes/No
    for col in ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']:
        df[col] = df[col].astype(str).str.strip().str.capitalize()

    # Variables categóricas que deben codificarse
    categorical_cols = ['Gender', 'Profession', 'Dietary Habits',
                        'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

    # Codificar dummies sin drop_first para mantener todas las columnas
    dummies = pd.get_dummies(df[categorical_cols])
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, dummies], axis=1)

    # Asegurar que todas las columnas estén presentes en el mismo orden
    for col in columnas_modelo:
        if col not in df.columns:
            df[col] = 0  # Agrega columnas faltantes con 0
    df = df[columnas_modelo]

    # Escalar variables numéricas
    columnas_numericas = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
                        'Job Satisfaction', 'Work/Study Hours', 'Financial Stress', 'Sleep Duration']

    df[columnas_numericas] = scaler.transform(df[columnas_numericas])

    return df
