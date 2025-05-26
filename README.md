# Comparación de Modelos de Machine Learning para Predicción de Diabetes

Este proyecto compara diferentes modelos de machine learning para predecir la diabetes utilizando indicadores de salud.

## Modelos Implementados

1. Regresión Logística
2. Árbol de Decisión
3. Random Forest
4. SVM (Support Vector Machine)
5. KNN (K-Nearest Neighbors)
6. Naive Bayes
7. Red Neuronal

## Estructura del Proyecto

- `comparar_modelos.py`: Script principal que ejecuta y compara todos los modelos
- `utils.py`: Funciones de utilidad para cargar datos y evaluar modelos
- Archivos individuales para cada modelo:
  - `regresion_logistica.py`
  - `arbol_decision.py`
  - `random_forest.py`
  - `svm.py`
  - `knn.py`
  - `naive_bayes.py`
  - `red_neuronal.py`

## Requisitos

```bash
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
kagglehub>=0.1.0
joblib>=1.2.0
```

## Uso

Para ejecutar la comparación de todos los modelos:
```bash
python comparar_modelos.py
```

Para ejecutar un modelo específico (por ejemplo, regresión logística):
```bash
python regresion_logistica.py
```

## Resultados

Los resultados se guardan en la carpeta `models/` e incluyen:
- Métricas de rendimiento (accuracy, precision, recall, F1-score)
- Matrices de confusión
- Visualizaciones comparativas
- Archivos CSV con resultados detallados

## Dataset

El proyecto utiliza el dataset "Diabetes Health Indicators" de Kaggle, que incluye diversos indicadores de salud para predecir la diabetes. 
