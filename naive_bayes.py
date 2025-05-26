from sklearn.naive_bayes import GaussianNB
from utils import (
    cargar_dataset,
    preparar_datos,
    evaluar_modelo,
    visualizar_resultados,
    guardar_resultados
)

def entrenar_naive_bayes():
    """
    Entrena y evalúa un modelo Naive Bayes para la predicción de diabetes
    """
    # Cargar datos
    print("Cargando dataset...")
    df = cargar_dataset('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
    
    # Preparar datos
    print("Preparando datos...")
    X_train, X_test, y_train, y_test = preparar_datos(df)
    
    # Crear y entrenar modelo
    print("Entrenando modelo...")
    modelo = GaussianNB()
    modelo.fit(X_train, y_train)
    
    # Realizar predicciones
    print("\nRealizando predicciones...")
    y_pred = modelo.predict(X_test)
    
    # Evaluar modelo
    resultados = evaluar_modelo(y_test, y_pred, "Naive Bayes")
    
    # Visualizar resultados
    print("Generando visualizaciones...")
    visualizar_resultados(y_test, y_pred, "Naive_Bayes")
    
    # Guardar resultados
    guardar_resultados(resultados, "naive_bayes")
    
    return modelo

if __name__ == "__main__":
    entrenar_naive_bayes() 