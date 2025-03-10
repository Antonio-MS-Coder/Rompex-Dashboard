# Rompex Dashboard

Dashboard interactivo para evaluar la ubicación óptima de una nueva planta de distribución en México, basado en diversos indicadores económicos y sociales por estado.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Rompex+Dashboard+Preview)

## Descripción

Este dashboard permite analizar y comparar diferentes estados de México para determinar la ubicación ideal para una nueva planta de distribución. Utiliza datos de indicadores económicos, sociales y de infraestructura para cada estado, y permite ponderar estos criterios según su importancia para la decisión.

## Características

- **Ponderación de criterios**: Asigna pesos a diferentes factores según su importancia para la decisión.
- **Ranking de estados**: Visualiza qué estados son más adecuados según los criterios ponderados.
- **Mapa interactivo**: Visualiza los datos geográficamente en un mapa de México.
- **Análisis comparativo**: Compara estados específicos mediante gráficos de radar.
- **Análisis de correlación**: Entiende cómo se relacionan las diferentes variables entre sí.
- **Visualización de indicadores**: Analiza cada indicador por separado para todos los estados.

## Datos utilizados

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

## Instalación

1. Clona este repositorio:
   ```
   git clone https://github.com/TU_USUARIO/Rompex-Dashboard.git
   cd Rompex-Dashboard
   ```

2. Crea un entorno virtual:
   ```
   python3 -m venv venv
   ```

3. Activa el entorno virtual:
   - En macOS/Linux:
     ```
     source venv/bin/activate
     ```
   - En Windows:
     ```
     venv\Scripts\activate
     ```

4. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

## Uso

1. Ejecuta la aplicación:
   ```
   python app.py
   ```

2. Abre tu navegador web y ve a:
   ```
   http://127.0.0.1:8050/
   ```

3. Utiliza el dashboard:
   - Ajusta los pesos de los criterios según su importancia
   - Selecciona qué criterios deben invertirse (donde un valor menor es mejor)
   - Haz clic en "Calcular Ranking" para ver los resultados
   - Explora las diferentes visualizaciones en las pestañas

## Estructura del proyecto

- `app.py`: Archivo principal que contiene el código del dashboard
- `utils.py`: Funciones auxiliares para el procesamiento de datos
- `mexico_states.py`: Datos geográficos de los estados de México
- `assets/styles.css`: Estilos CSS para el dashboard
- `requirements.txt`: Lista de dependencias
- `BASE DE DATOS ROMPEX.xlsx - Hoja 3.csv`: Datos utilizados por el dashboard

## Tecnologías utilizadas

- [Dash](https://dash.plotly.com/): Framework para crear aplicaciones web analíticas
- [Plotly](https://plotly.com/python/): Biblioteca para visualizaciones interactivas
- [Pandas](https://pandas.pydata.org/): Manipulación y análisis de datos
- [NumPy](https://numpy.org/): Computación numérica
- [scikit-learn](https://scikit-learn.org/): Herramientas de aprendizaje automático

## Contribuir

Si deseas contribuir a este proyecto, por favor:
1. Haz un fork del repositorio
2. Crea una rama para tu característica (`git checkout -b feature/nueva-caracteristica`)
3. Haz commit de tus cambios (`git commit -m 'Añadir nueva característica'`)
4. Haz push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Contacto

Si tienes preguntas o comentarios sobre este proyecto, por favor contacta a [tu nombre o correo electrónico]. 