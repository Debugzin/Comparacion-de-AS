import os
from sklearn.neighbors import KNeighborsClassifier
from utils import (
    cargar_dataset,
    preparar_datos,
    evaluar_modelo,
    visualizar_resultados,
    guardar_resultados
)

# Configurar el número de cores a usar
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Usar 4 cores, ajusta este número según tu CPU

def entrenar_knn():
    """
    Entrena y evalúa un modelo KNN para la predicción de diabetes
    """
    # Cargar datos
    print("Cargando dataset...")
    df = cargar_dataset('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
    
    # Preparar datos
    print("Preparando datos...")
    X_train, X_test, y_train, y_test = preparar_datos(df)
    
    # Crear y entrenar modelo
    print("Entrenando modelo...")
    modelo = KNeighborsClassifier()
    modelo.fit(X_train, y_train)
    
    # Realizar predicciones
    print("\nRealizando predicciones...")
    y_pred = modelo.predict(X_test)
    
    # Evaluar modelo
    resultados = evaluar_modelo(y_test, y_pred, "KNN")
    
    # Visualizar resultados
    print("Generando visualizaciones...")
    visualizar_resultados(y_test, y_pred, "KNN")
    
    # Guardar resultados
    guardar_resultados(resultados, "knn")
    
    return modelo

if __name__ == "__main__":
    entrenar_knn() 