import kagglehub
import pandas as pd
import os

def descargar_dataset():
    """
    Descarga el dataset de Kaggle y lo guarda en la carpeta datasets
    """
    print("Descargando dataset...")
    path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
    
    # Crear directorio datasets si no existe
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    
    # Copiar los archivos necesarios a la carpeta datasets
    print("Copiando archivos...")
    df_binary_5050 = pd.read_csv(os.path.join(path, 'diabetes_binary_5050split_health_indicators_BRFSS2015.csv'))
    df_binary = pd.read_csv(os.path.join(path, 'diabetes_binary_health_indicators_BRFSS2015.csv'))
    df_012 = pd.read_csv(os.path.join(path, 'diabetes_012_health_indicators_BRFSS2015.csv'))
    
    # Guardar en la carpeta datasets
    df_binary_5050.to_csv('datasets/diabetes_binary_5050split_health_indicators_BRFSS2015.csv', index=False)
    df_binary.to_csv('datasets/diabetes_binary_health_indicators_BRFSS2015.csv', index=False)
    df_012.to_csv('datasets/diabetes_012_health_indicators_BRFSS2015.csv', index=False)
    
    print("Dataset descargado y preparado con éxito!")
    print("\nEstadísticas del dataset:")
    print(f"Total de registros (binary_5050): {len(df_binary_5050)}")
    print(f"Total de registros (binary): {len(df_binary)}")
    print(f"Total de registros (012): {len(df_012)}")
    
    # Mostrar distribución de clases
    print("\nDistribución de clases (binary_5050):")
    print(df_binary_5050['Diabetes_binary'].value_counts())

if __name__ == "__main__":
    descargar_dataset() 