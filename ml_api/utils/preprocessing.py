import pandas as pd

def procesar_dato(nuevo_dato, scaler, columnas_modelo):
    df = pd.DataFrame([nuevo_dato])

    # Mapeo de sueño
    map_sueno = {
        'Less than 5 hours': 4.5,
        '5-6 hours': 5.5,
        '7-8 hours': 7.5,
        'More than 8 hours': 9.0,
        'Others': 4
    }
    df['Sleep Duration'] = df['Sleep Duration'].map(map_sueno)

    # Normalizar valores binarios
    for col in ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']:
        df[col] = df[col].astype(str).str.strip().str.capitalize()

    # Variables categóricas
    categorical_cols = ['Gender', 'Profession', 'Dietary Habits',
                        'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

    # ✅ Consistente con entrenamiento: drop_first=True
    dummies = pd.get_dummies(df[categorical_cols], drop_first=True)
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, dummies], axis=1)

    # Rellenar columnas faltantes con 0
    for col in columnas_modelo:
        if col not in df.columns:
            df[col] = 0

    # Ordenar exactamente como en el modelo
    df = df[columnas_modelo]

    # Escalado seguro (solo columnas numéricas)
    columnas_numericas = [
        'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
        'Study Satisfaction', 'Job Satisfaction',
        'Work/Study Hours', 'Financial Stress', 'Sleep Duration'
    ]
    df[columnas_numericas] = scaler.transform(df[columnas_numericas])

    return df
