import pandas as pd
from scipy.stats import ks_2samp

def detectar_drift(ruta_original, ruta_nuevo, columnas_a_comparar, umbral=0.05):
    """
    Compara la distribución de columnas entre el dataset original y los nuevos registros.

    Args:
        ruta_original (str): Ruta al archivo CSV original.
        ruta_nuevo (str): Ruta al archivo CSV de registros reales.
        columnas_a_comparar (list): Lista de nombres de columnas a comparar.
        umbral (float): Valor de corte para p-value. Por defecto: 0.05.
    """
    try:
        df_original = pd.read_csv(ruta_original)
        df_nuevo = pd.read_csv(ruta_nuevo)
    except Exception as e:
        print(f"❌ Error cargando archivos: {e}")
        return

    print("\n🔍 Resultados de comparación (Kolmogorov-Smirnov):\n")

    for col in columnas_a_comparar:
        if col in df_original.columns and col in df_nuevo.columns:
            original_col = pd.to_numeric(df_original[col], errors='coerce').dropna()
            nuevo_col = pd.to_numeric(df_nuevo[col], errors='coerce').dropna()
            # Evaluar si hay suficientes datos nuevos
            if len(nuevo_col) < 50:
                print(f"{col:<35} | ⚠️ No se evalúa (solo {len(nuevo_col)} nuevos registros)")
                continue

            if len(original_col) > 0 and len(nuevo_col) > 0:
                stat, p_value = ks_2samp(original_col, nuevo_col)
                alerta = "⚠️ DRIFT detectado" if p_value < umbral else "✅ Sin drift"
                print(f"{col:<35} | p-value: {p_value:.4f} | {alerta}")
            else:
                print(f"{col:<35} | ⚠️ Columnas sin suficientes datos")
        else:
            print(f"{col:<35} | ❌ Columna no encontrada en ambos datasets")

# --- EJECUCIÓN DIRECTA ---

if __name__ == "__main__":
    columnas = [
        "Age", "Academic Pressure", "Work Pressure", "CGPA", "Study Satisfaction",
        "Job Satisfaction", "Sleep Duration", "Work/Study Hours", "Financial Stress",
        "Gender_Male", "Dietary Habits_Moderate", "Dietary Habits_Others",
        "Dietary Habits_Unhealthy", "Have you ever had suicidal thoughts ?_Yes",
        "Family History of Mental Illness_Yes"
    ]

    detectar_drift(
        ruta_original="dataset_original.csv",
        ruta_nuevo="data/registros.csv",
        columnas_a_comparar=columnas
    )
