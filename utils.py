import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def cargar_datos(ruta_archivo):
    """
    Carga los datos desde un archivo CSV y limpia los nombres de columnas.
    
    Args:
        ruta_archivo (str): Ruta al archivo CSV.
        
    Returns:
        pandas.DataFrame: DataFrame con los datos cargados.
    """
    df = pd.read_csv(ruta_archivo)
    df.columns = df.columns.str.strip()
    return df

def normalizar_datos(df, columnas, invertir=None):
    """
    Normaliza las columnas especificadas a una escala de 0 a 1.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        columnas (list): Lista de columnas a normalizar.
        invertir (list, optional): Lista de columnas donde un valor menor es mejor.
        
    Returns:
        pandas.DataFrame: DataFrame con las columnas normalizadas.
    """
    if invertir is None:
        invertir = []
    
    df_norm = df.copy()
    scaler = MinMaxScaler()
    
    for col in columnas:
        if col in df.columns:
            df_norm[f"{col}_norm"] = scaler.fit_transform(df[[col]])
            
            # Invertir si es necesario (1 - valor)
            if col in invertir:
                df_norm[f"{col}_norm"] = 1 - df_norm[f"{col}_norm"]
    
    return df_norm

def calcular_puntuacion(df, columnas, pesos, invertir=None):
    """
    Calcula una puntuación ponderada para cada fila basada en las columnas especificadas.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        columnas (list): Lista de columnas a considerar.
        pesos (list): Lista de pesos para cada columna.
        invertir (list, optional): Lista de columnas donde un valor menor es mejor.
        
    Returns:
        pandas.DataFrame: DataFrame con la puntuación calculada.
    """
    # Normalizar los datos
    df_norm = normalizar_datos(df, columnas, invertir)
    
    # Aplicar pesos
    for i, col in enumerate(columnas):
        if col in df.columns:
            df_norm[f"{col}_weighted"] = df_norm[f"{col}_norm"] * pesos[i]
    
    # Calcular puntuación total
    weighted_cols = [f"{col}_weighted" for col in columnas if f"{col}_weighted" in df_norm.columns]
    df_norm["puntuacion_total"] = df_norm[weighted_cols].sum(axis=1)
    
    # Normalizar la puntuación total a escala 0-100
    max_score = df_norm["puntuacion_total"].max()
    df_norm["puntuacion_final"] = (df_norm["puntuacion_total"] / max_score) * 100
    
    return df_norm

def identificar_fortalezas(df, columnas, umbral=0.7):
    """
    Identifica las fortalezas de cada estado basado en sus valores normalizados.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos normalizados.
        columnas (list): Lista de columnas a considerar.
        umbral (float, optional): Umbral para considerar una variable como fortaleza.
        
    Returns:
        dict: Diccionario con las fortalezas de cada estado.
    """
    fortalezas = {}
    
    for _, row in df.iterrows():
        estado = row["Estado"]
        fortalezas[estado] = []
        
        for col in columnas:
            if f"{col}_norm" in df.columns and row[f"{col}_norm"] > umbral:
                fortalezas[estado].append(col)
    
    return fortalezas

def calcular_correlacion(df, columnas):
    """
    Calcula la matriz de correlación entre las columnas especificadas.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        columnas (list): Lista de columnas a considerar.
        
    Returns:
        pandas.DataFrame: Matriz de correlación.
    """
    return df[columnas].corr()

def obtener_top_estados(df, n=5):
    """
    Obtiene los top N estados según la puntuación final.
    
    Args:
        df (pandas.DataFrame): DataFrame con la puntuación calculada.
        n (int, optional): Número de estados a retornar.
        
    Returns:
        pandas.DataFrame: DataFrame con los top N estados.
    """
    return df.sort_values("puntuacion_final", ascending=False).head(n) 