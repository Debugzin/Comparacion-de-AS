import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from regresion_logistica import entrenar_regresion_logistica
from arbol_decision import entrenar_arbol_decision
from random_forest import entrenar_random_forest
from svm import entrenar_svm
from knn import entrenar_knn
from naive_bayes import entrenar_naive_bayes
from red_neuronal import entrenar_red_neuronal
import time

def mostrar_progreso(modelo_actual, total_modelos, inicio_tiempo):
    """
    Muestra el progreso de entrenamiento
    """
    tiempo_transcurrido = time.time() - inicio_tiempo
    tiempo_estimado = (tiempo_transcurrido / modelo_actual) * total_modelos if modelo_actual > 0 else 0
    tiempo_restante = tiempo_estimado - tiempo_transcurrido
    
    print("\n" + "="*50)
    print(f"Progreso: {modelo_actual}/{total_modelos} modelos completados")
    print(f"Tiempo transcurrido: {tiempo_transcurrido:.1f} segundos")
    print(f"Tiempo restante estimado: {tiempo_restante:.1f} segundos")
    print("="*50 + "\n")

def crear_visualizaciones(df_resultados):
    """
    Crea visualizaciones comparativas de los resultados
    """
    # 1. Gráfico de barras para cada métrica
    plt.figure(figsize=(15, 10))
    metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for i, metrica in enumerate(metricas):
        plt.subplot(2, 2, i+1)
        ax = sns.barplot(data=df_resultados, x='Modelo', y=metrica)
        plt.xticks(rotation=45)
        plt.title(f'Comparación de {metrica}')
        
        # Añadir valores en las barras
        for j, v in enumerate(df_resultados[metrica]):
            ax.text(j, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('models/comparacion_metricas.png')
    plt.close()
    
    # 2. Gráfico de radar
    plt.figure(figsize=(10, 10))
    
    # Preparar datos para el gráfico de radar
    modelos = df_resultados['Modelo'].values
    valores = df_resultados[metricas].values
    
    # Número de variables
    num_vars = len(metricas)
    
    # Calcular ángulos para el gráfico de radar
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Inicializar el gráfico
    ax = plt.subplot(111, polar=True)
    
    # Dibujar un eje por variable
    plt.xticks(angles[:-1], metricas)
    
    # Graficar datos
    for i, modelo in enumerate(modelos):
        valores_modelo = valores[i].tolist()
        valores_modelo += valores_modelo[:1]
        ax.plot(angles, valores_modelo, linewidth=1, linestyle='solid', label=modelo)
        ax.fill(angles, valores_modelo, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Comparación de Modelos - Gráfico Radar")
    plt.savefig('models/comparacion_radar.png')
    plt.close()

def ejecutar_todos_modelos():
    """
    Ejecuta todos los modelos y compara sus resultados
    """
    print("Ejecutando todos los modelos...")
    
    # Lista de funciones de entrenamiento
    modelos = [
        entrenar_regresion_logistica,
        entrenar_arbol_decision,
        entrenar_random_forest,
        entrenar_svm,
        entrenar_knn,
        entrenar_naive_bayes,
        entrenar_red_neuronal
    ]
    
    total_modelos = len(modelos)
    inicio_tiempo = time.time()
    
    # Ejecutar cada modelo
    for i, modelo in enumerate(modelos, 1):
        print(f"\nEjecutando {modelo.__name__}...")
        modelo()
        mostrar_progreso(i, total_modelos, inicio_tiempo)
    
    tiempo_total = time.time() - inicio_tiempo
    print(f"\nTiempo total de ejecución: {tiempo_total:.1f} segundos")
    
    # Cargar y comparar resultados
    print("\nComparando resultados...")
    resultados = []
    for nombre in ['regresion_logistica', 'arbol_decision', 'random_forest', 
                  'svm', 'knn', 'naive_bayes', 'red_neuronal']:
        try:
            df = pd.read_csv(f'models/resultados_{nombre}.csv')
            resultados.append(df)
        except FileNotFoundError:
            print(f"No se encontraron resultados para {nombre}")
    
    if resultados:
        # Combinar todos los resultados
        df_resultados = pd.concat(resultados, ignore_index=True)
        
        # Crear visualizaciones
        crear_visualizaciones(df_resultados)
        
        # Mostrar tabla de resultados
        print("\nResultados finales:")
        print(df_resultados.to_string(index=False))
        
        # Guardar resultados comparativos
        df_resultados.to_csv('models/comparacion_resultados.csv', index=False)
        
        # Identificar mejor modelo para cada métrica
        print("\nMejores modelos por métrica:")
        metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for metrica in metricas:
            idx = df_resultados[metrica].idxmax()
            mejor = df_resultados.iloc[idx]
            print(f"\nMejor modelo según {metrica}:")
            print(f"Modelo: {mejor['Modelo']}")
            print(f"Valor: {mejor[metrica]:.4f}")
        
        # Calcular ranking general
        df_resultados['Ranking_Promedio'] = df_resultados[metricas].mean(axis=1)
        idx_mejor = df_resultados['Ranking_Promedio'].idxmax()
        mejor_general = df_resultados.iloc[idx_mejor]
        
        print("\nMejor modelo general (promedio de todas las métricas):")
        print(f"Modelo: {mejor_general['Modelo']}")
        print(f"Ranking promedio: {mejor_general['Ranking_Promedio']:.4f}")
        
        # Mostrar ranking completo
        print("\nRanking completo de modelos:")
        ranking = df_resultados.sort_values('Ranking_Promedio', ascending=False)
        for idx, modelo in ranking.iterrows():
            print(f"{modelo['Modelo']}: {modelo['Ranking_Promedio']:.4f}")

if __name__ == "__main__":
    ejecutar_todos_modelos() 