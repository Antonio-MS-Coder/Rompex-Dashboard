# Dashboard para Evaluación de Ubicación de Planta de Distribución

Este dashboard interactivo permite evaluar diferentes estados de México para determinar la ubicación óptima para una nueva planta de distribución.

## Características

- Visualización de indicadores económicos por estado
- Análisis comparativo entre estados
- Ranking de estados según criterios ponderados
- Mapa de calor para visualizar datos geográficamente
- Análisis de correlación entre variables

## Instalación

1. Clonar este repositorio
2. Instalar las dependencias:
   ```
   pip install -r requirements.txt
   ```
3. Ejecutar la aplicación:
   ```
   python app.py
   ```

## Datos

El dashboard utiliza datos de diferentes indicadores por estado, incluyendo:
- PIB Estatal
- Inversión extranjera
- Crecimiento del PIB
- PIB de Construcción
- Terminales punto de venta
- Location Quotient (LQ)
- Índice de transparencia fiscal
- Satisfacción de servicios generales
- Días para trámites de apertura de empresa
- Áreas naturales protegidas

## Uso

Una vez ejecutada la aplicación, abra su navegador web y vaya a `http://127.0.0.1:8050/` para acceder al dashboard. 