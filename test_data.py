import pandas as pd

# Intentar cargar los datos
try:
    print("Intentando cargar el archivo CSV...")
    df = pd.read_csv('BASE DE DATOS ROMPEX.xlsx - Hoja 3.csv')
    
    # Mostrar información sobre el DataFrame
    print("\nInformación del DataFrame:")
    print(f"Forma: {df.shape}")
    print(f"Columnas: {df.columns.tolist()}")
    
    # Mostrar las primeras filas
    print("\nPrimeras 5 filas:")
    print(df.head())
    
    # Verificar tipos de datos
    print("\nTipos de datos:")
    print(df.dtypes)
    
    # Verificar valores nulos
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    
    print("\nCarga exitosa!")
except Exception as e:
    print(f"Error al cargar los datos: {e}") 