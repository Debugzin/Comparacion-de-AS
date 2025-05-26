from sklearn.linear_model import LogisticRegression
from utils import (
    cargar_dataset,
    preparar_datos,
    evaluar_modelo,
    visualizar_resultados,
    guardar_resultados
)

def entrenar_regresion_logistica():
    """
    Entrena y evalúa un modelo de regresión logística para la predicción de diabetes
    """
    # Cargar datos
    print("Cargando dataset...")
    df = cargar_dataset('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
    
    # Preparar datos
    print("Preparando datos...")
    X_train, X_test, y_train, y_test = preparar_datos(df)
    
    # Crear y entrenar modelo
    print("Entrenando modelo...")
    modelo = LogisticRegression(random_state=42)
    modelo.fit(X_train, y_train)
    
    # Realizar predicciones
    print("\nRealizando predicciones...")
    y_pred = modelo.predict(X_test)
    
    # Evaluar modelo
    resultados = evaluar_modelo(y_test, y_pred, "Regresión Logística")
    
    # Visualizar resultados
    print("Generando visualizaciones...")
    visualizar_resultados(y_test, y_pred, "Regresion_Logistica")
    
    # Guardar resultados
    guardar_resultados(resultados, "regresion_logistica")
    
    return modelo

if __name__ == "__main__":
    entrenar_regresion_logistica() 