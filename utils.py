import pandas as pd
import numpy as np
import os
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import parallel_backend, Memory

# Configurar recursos
os.environ['LOKY_MAX_CPU_COUNT'] = '6'  # Usar 6 núcleos físicos
os.environ['OMP_NUM_THREADS'] = '12'    # Usar 12 hilos
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
memory = Memory(location='./cachedir', verbose=0)

# Configurar el uso de memoria para scikit-learn
import sklearn
sklearn.set_config(working_memory=8*1024)  # 8GB de RAM

def obtener_dataset():
    """
    Descarga y obtiene el dataset de Kaggle
    """
    print("Descargando dataset desde Kaggle...")
    path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
    print("Path to dataset files:", path)
    return path

def crear_features_adicionales(df):
    """
    Crea características adicionales basadas en el conocimiento del dominio
    """
    # Índice de Salud General (combinación de salud física y mental)
    df['IndicesSaludGeneral'] = (df['GenHlth'] * 2 + 
                                df['PhysHlth'] / 30 + 
                                df['MentHlth'] / 30) / 4
    
    # Índice de Riesgo Cardiovascular
    df['RiesgoCardiovascular'] = df['HighBP'] + df['HighChol'] + df['HeartDiseaseorAttack']
    
    # Índice de Estilo de Vida Saludable
    df['EstiloVidaSaludable'] = (df['PhysActivity'] + 
                                df['Fruits'] + 
                                df['Veggies'] - 
                                df['Smoker'] - 
                                df['HvyAlcoholConsump'])
    
    # Índice de Acceso a Salud
    df['AccesoSalud'] = df['AnyHealthcare'] - df['NoDocbcCost']
    
    # BMI Categories (basado en rangos estándar)
    df['BMI_Category'] = pd.cut(df['BMI'], 
                               bins=[0, 18.5, 24.9, 29.9, float('inf')],
                               labels=[0, 1, 2, 3])
    
    # Interacciones importantes
    df['BMI_X_Age'] = df['BMI'] * df['Age']
    df['HighBP_X_HighChol'] = df['HighBP'] * df['HighChol']
    df['Health_X_Age'] = df['GenHlth'] * df['Age']
    
    return df

@memory.cache
def cargar_dataset(nombre_archivo):
    """
    Carga el dataset desde Kaggle o desde la carpeta local
    """
    path = obtener_dataset()
    archivo = os.path.join(path, nombre_archivo)
    df = pd.read_csv(archivo)
    
    # Aplicar feature engineering
    df = crear_features_adicionales(df)
    
    return df

@memory.cache
def preparar_datos(df, test_size=0.2, random_state=42):
    """
    Prepara los datos para el entrenamiento
    """
    # Separar features y target
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluar_modelo(y_true, y_pred, nombre_modelo):
    """
    Evalúa el modelo usando múltiples métricas
    """
    resultados = {
        'Modelo': nombre_modelo,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }
    
    print(f"\nResultados para {nombre_modelo}:")
    for metrica, valor in resultados.items():
        if metrica != 'Modelo':
            print(f"{metrica}: {valor:.4f}")
    
    return resultados

def visualizar_resultados(y_true, y_pred, nombre_modelo):
    """
    Genera visualizaciones de los resultados
    """
    # Crear directorio models si no existe
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {nombre_modelo}')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    plt.savefig(f'models/{nombre_modelo}_confusion_matrix.png')
    plt.close()

def guardar_resultados(resultados, nombre_archivo):
    """
    Guarda los resultados en un archivo CSV
    """
    # Crear directorio models si no existe
    if not os.path.exists('models'):
        os.makedirs('models')
    
    df_resultados = pd.DataFrame([resultados])
    df_resultados.to_csv(f'models/resultados_{nombre_archivo}.csv', index=False) 