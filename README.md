# Dashboard de Análisis de Estados de México

Este dashboard interactivo permite analizar y comparar los estados de México según diferentes criterios económicos y sociales.

![Dashboard Preview](dashboard_preview.png)

## Características

- **Mapa interactivo de México**: Visualiza indicadores económicos y sociales por estado.
- **Ranking de estados**: Compara los estados según una puntuación calculada en base a criterios ponderados.
- **Comparación de estados**: Visualiza el desempeño de múltiples estados en un gráfico de radar.
- **Análisis de indicadores**: Explora cada indicador individualmente para todos los estados.
- **Ponderación personalizable**: Ajusta la importancia de cada criterio según tus necesidades.

## Criterios de análisis

El dashboard incluye los siguientes criterios:

- PIB Estatal (millones de pesos)
- Inversión extranjera (proporción)
- Crecimiento del PIB (%)
- PIB de Construcción (millones de pesos)
- Terminales punto de venta (cantidad)
- Location Quotient
- Índice de transparencia fiscal
- Satisfacción de servicios generales
- Días para trámites de apertura de empresa
- Áreas naturales protegidas (hectáreas)

## Instalación

1. Clona este repositorio:
   ```
   git clone https://github.com/yourusername/mexico-dashboard.git
   cd mexico-dashboard
   ```

2. Crea un entorno virtual e instala las dependencias:
   ```
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Ejecuta la aplicación:
   ```
   python app.py
   ```

4. Abre tu navegador y ve a `http://127.0.0.1:8070/`

## Requisitos

- Python 3.7+
- Dash
- Plotly
- Pandas
- NumPy
- scikit-learn
- Matplotlib

Consulta el archivo `requirements.txt` para ver la lista completa de dependencias.

## Uso

1. Ajusta los pesos de cada criterio usando los deslizadores.
2. Haz clic en "Calcular" para actualizar las visualizaciones.
3. Explora las diferentes visualizaciones:
   - Mapa: Selecciona un indicador para visualizar en el mapa.
   - Ranking: Observa qué estados tienen mejor desempeño según los criterios ponderados.
   - Comparación: Selecciona estados específicos para comparar en un gráfico de radar.
   - Indicadores: Explora cada indicador individualmente.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir los cambios propuestos o envía un pull request.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles. 