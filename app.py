import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mexico_states import mexico_states_coords, mexico_states_iso

# Cargar los datos
df = pd.read_csv('BASE DE DATOS ROMPEX.xlsx - Hoja 3.csv')

# Obtener los nombres de columnas originales para referencia
columnas_originales = df.columns.tolist()

# Crear un diccionario de mapeo para las columnas con espacios
mapeo_columnas = {}
for col in columnas_originales:
    mapeo_columnas[col] = col.strip()

# Renombrar las columnas
df = df.rename(columns=mapeo_columnas)

# Definir los criterios y sus descripciones
criterios = {
    'PIB_Estatal': 'PIB Estatal (millones de pesos)',
    'Inversion_extranjera': 'Inversión extranjera (proporción)',
    'Crecimiento_PIB': 'Crecimiento del PIB (%)',
    'PIB_Construccion': 'PIB de Construcción (millones de pesos)',
    'Terminales_punto_venta': 'Terminales punto de venta (cantidad)',
    'LQ': 'Location Quotient',
    'indice_transp_fiscal': 'Índice de transparencia fiscal',
    'satisfaccion_servicios_generales': 'Satisfacción de servicios generales',
    'Días_tramites_abrir_empresa': 'Días para trámites de apertura de empresa',
    'areas_naturales_protegidas': 'Áreas naturales protegidas (hectáreas)'
}

# Crear la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Definir el layout del dashboard
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Dashboard para Evaluación de Ubicación de Planta de Distribución", 
                    className="text-center my-4"),
            html.P("Seleccione y pondere los criterios para evaluar la ubicación óptima de una nueva planta de distribución.",
                   className="text-center mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Ponderación de Criterios", className="mb-3"),
            html.P("Asigne un peso a cada criterio (0-10) según su importancia para la decisión."),
            
            # Sliders para ponderación de criterios
            *[dbc.Row([
                dbc.Col([
                    html.Label(desc, className="mt-2"),
                    dcc.Slider(
                        id=f'slider-{crit}',
                        min=0,
                        max=10,
                        step=1,
                        value=5,  # Valor predeterminado
                        marks={i: str(i) for i in range(0, 11, 2)},
                        className="mb-4"
                    )
                ])
            ]) for crit, desc in criterios.items()],
            
            # Botón para calcular ranking
            dbc.Button("Calcular Ranking", id="btn-calcular", color="primary", className="mt-3 mb-4 w-100"),
            
            # Selector para invertir criterios (menor es mejor)
            html.H5("Criterios a Invertir", className="mt-4 mb-2"),
            html.P("Seleccione los criterios donde un valor menor es mejor (ej: días para trámites)."),
            dbc.Checklist(
                id="checklist-invertir",
                options=[{"label": desc, "value": crit} for crit, desc in criterios.items()],
                value=["Días_tramites_abrir_empresa"],  # Por defecto, días para trámites (menor es mejor)
                className="mb-4"
            )
        ], width=4),
        
        dbc.Col([
            html.H4("Resultados del Análisis", className="mb-3"),
            
            # Tabs para diferentes visualizaciones
            dbc.Tabs([
                dbc.Tab([
                    dcc.Graph(id="graph-ranking", style={"height": "600px"})
                ], label="Ranking de Estados"),
                
                dbc.Tab([
                    dcc.Dropdown(
                        id="dropdown-indicador",
                        options=[{"label": desc, "value": crit} for crit, desc in criterios.items()],
                        value="PIB_Estatal",
                        className="mb-3"
                    ),
                    dcc.Graph(id="graph-indicador", style={"height": "550px"})
                ], label="Indicadores por Estado"),
                
                dbc.Tab([
                    dcc.Graph(id="graph-correlacion", style={"height": "600px"})
                ], label="Correlación de Variables"),
                
                dbc.Tab([
                    dcc.Dropdown(
                        id="dropdown-comparar",
                        options=[{"label": estado, "value": estado} for estado in df["Estado"].unique()],
                        value=df["Estado"].unique()[:5].tolist(),
                        multi=True,
                        className="mb-3"
                    ),
                    dcc.Graph(id="graph-comparar", style={"height": "550px"})
                ], label="Comparar Estados"),
                
                # Nueva pestaña para el mapa de calor
                dbc.Tab([
                    dcc.Dropdown(
                        id="dropdown-mapa",
                        options=[{"label": desc, "value": crit} for crit, desc in criterios.items()] + 
                                [{"label": "Puntuación Final", "value": "puntuacion_final"}],
                        value="PIB_Estatal",
                        className="mb-3"
                    ),
                    dcc.Graph(id="graph-mapa", style={"height": "550px"})
                ], label="Mapa de México"),
            ]),
        ], width=8)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Top 5 Estados Recomendados", className="mt-4 mb-3"),
            html.Div(id="top-estados", className="mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Footer([
                html.P("Dashboard desarrollado para evaluación de ubicación de planta de distribución. © 2023",
                       className="text-center text-muted")
            ], className="mt-4 pt-3 border-top")
        ])
    ])
], fluid=True)

# Callback para actualizar el ranking de estados
@app.callback(
    [Output("graph-ranking", "figure"),
     Output("top-estados", "children")],
    [Input("btn-calcular", "n_clicks")],
    [State(f"slider-{crit}", "value") for crit in criterios.keys()] + 
    [State("checklist-invertir", "value")]
)
def actualizar_ranking(n_clicks, *args):
    if n_clicks is None:
        # Valores predeterminados para la carga inicial
        pesos = [5] * len(criterios)
        invertir = ["Días_tramites_abrir_empresa"]
    else:
        pesos = args[:-1]  # Todos los argumentos excepto el último (checklist)
        invertir = args[-1]  # Último argumento (checklist)
    
    # Crear una copia del dataframe para no modificar el original
    df_calc = df.copy()
    
    # Normalizar los datos (0-1) para cada criterio
    scaler = MinMaxScaler()
    
    for i, (crit, _) in enumerate(criterios.items()):
        if crit in df_calc.columns:
            # Normalizar
            df_calc[f"{crit}_norm"] = scaler.fit_transform(df_calc[[crit]])
            
            # Invertir si es necesario (1 - valor)
            if crit in invertir:
                df_calc[f"{crit}_norm"] = 1 - df_calc[f"{crit}_norm"]
            
            # Aplicar peso
            df_calc[f"{crit}_weighted"] = df_calc[f"{crit}_norm"] * pesos[i]
    
    # Calcular puntuación total
    weighted_cols = [f"{crit}_weighted" for crit in criterios.keys() if f"{crit}_weighted" in df_calc.columns]
    df_calc["puntuacion_total"] = df_calc[weighted_cols].sum(axis=1)
    
    # Normalizar la puntuación total a escala 0-100
    max_score = df_calc["puntuacion_total"].max()
    df_calc["puntuacion_final"] = (df_calc["puntuacion_total"] / max_score) * 100
    
    # Ordenar por puntuación
    df_ranking = df_calc.sort_values("puntuacion_final", ascending=False)
    
    # Crear gráfico de barras para el ranking
    fig = px.bar(
        df_ranking,
        x="Estado",
        y="puntuacion_final",
        color="puntuacion_final",
        color_continuous_scale="viridis",
        labels={"puntuacion_final": "Puntuación (0-100)", "Estado": ""},
        title="Ranking de Estados para Ubicación de Planta"
    )
    
    fig.update_layout(
        xaxis={'categoryorder': 'total descending'},
        coloraxis_showscale=False,
        height=600
    )
    
    # Crear componente para mostrar los top 5 estados
    top5 = df_ranking.head(5)
    top_estados_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(f"#{i+1}: {row['Estado']}", className="text-center"),
                dbc.CardBody([
                    html.H5(f"{row['puntuacion_final']:.1f} puntos", className="text-center"),
                    html.P([
                        html.Strong("Fortalezas: "),
                        ", ".join([
                            criterios[crit] for crit, _ in criterios.items()
                            if f"{crit}_norm" in df_ranking.columns and row[f"{crit}_norm"] > 0.7
                        ])
                    ])
                ])
            ], className="h-100")
        ], width=12 // min(5, len(top5)))
        for i, (_, row) in enumerate(top5.iterrows())
    ])
    
    return fig, top_estados_cards

# Callback para actualizar el gráfico de indicador por estado
@app.callback(
    Output("graph-indicador", "figure"),
    [Input("dropdown-indicador", "value")]
)
def actualizar_indicador(indicador):
    # Ordenar por el indicador seleccionado
    df_sorted = df.sort_values(indicador, ascending=False)
    
    # Crear gráfico de barras
    fig = px.bar(
        df_sorted,
        x="Estado",
        y=indicador,
        color=indicador,
        color_continuous_scale="viridis",
        labels={indicador: criterios[indicador], "Estado": ""},
        title=f"{criterios[indicador]} por Estado"
    )
    
    fig.update_layout(
        xaxis={'categoryorder': 'total descending'},
        height=550
    )
    
    return fig

# Callback para actualizar el gráfico de correlación
@app.callback(
    Output("graph-correlacion", "figure"),
    [Input("btn-calcular", "n_clicks")]
)
def actualizar_correlacion(_):
    # Seleccionar solo las columnas numéricas para la correlación
    df_corr = df.select_dtypes(include=[np.number])
    
    # Calcular la matriz de correlación
    corr_matrix = df_corr.corr()
    
    # Crear mapa de calor
    fig = px.imshow(
        corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Matriz de Correlación entre Variables"
    )
    
    fig.update_layout(
        height=600,
        xaxis={'tickangle': 45}
    )
    
    return fig

# Callback para actualizar la comparación de estados
@app.callback(
    Output("graph-comparar", "figure"),
    [Input("dropdown-comparar", "value")]
)
def actualizar_comparacion(estados):
    if not estados:
        estados = df["Estado"].unique()[:5].tolist()
    
    # Filtrar el dataframe para los estados seleccionados
    df_comp = df[df["Estado"].isin(estados)]
    
    # Preparar datos para gráfico de radar
    categories = list(criterios.keys())
    
    # Normalizar los datos para el gráfico de radar
    df_radar = df_comp.copy()
    scaler = MinMaxScaler()
    
    for cat in categories:
        if cat in df_radar.columns:
            df_radar[f"{cat}_norm"] = scaler.fit_transform(df[[cat]])
    
    # Crear gráfico de radar
    fig = go.Figure()
    
    for i, estado in enumerate(df_radar["Estado"]):
        values = []
        theta_labels = []
        
        for cat in categories:
            if cat in df_radar.columns and f"{cat}_norm" in df_radar.columns:
                values.append(df_radar[df_radar["Estado"] == estado][f"{cat}_norm"].values[0])
                theta_labels.append(criterios[cat])
        
        # Cerrar el polígono
        if values:
            values.append(values[0])
            theta_labels.append(theta_labels[0])
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=theta_labels,
                fill='toself',
                name=estado
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Comparación de Estados por Indicadores Normalizados",
        height=550
    )
    
    return fig

# Callback para actualizar el mapa de calor de México
@app.callback(
    Output("graph-mapa", "figure"),
    [Input("dropdown-mapa", "value"),
     Input("btn-calcular", "n_clicks")],
    [State(f"slider-{crit}", "value") for crit in criterios.keys()] + 
    [State("checklist-invertir", "value")]
)
def actualizar_mapa(indicador, n_clicks, *args):
    # Determinar si mostrar el indicador seleccionado o la puntuación final
    if n_clicks is not None and indicador == "puntuacion_final":
        # Usar los mismos cálculos que en actualizar_ranking
        pesos = args[:-1]
        invertir = args[-1]
        
        # Crear una copia del dataframe para no modificar el original
        df_calc = df.copy()
        
        # Normalizar los datos (0-1) para cada criterio
        scaler = MinMaxScaler()
        
        for i, (crit, _) in enumerate(criterios.items()):
            if crit in df_calc.columns:
                # Normalizar
                df_calc[f"{crit}_norm"] = scaler.fit_transform(df_calc[[crit]])
                
                # Invertir si es necesario (1 - valor)
                if crit in invertir:
                    df_calc[f"{crit}_norm"] = 1 - df_calc[f"{crit}_norm"]
                
                # Aplicar peso
                df_calc[f"{crit}_weighted"] = df_calc[f"{crit}_norm"] * pesos[i]
        
        # Calcular puntuación total
        weighted_cols = [f"{crit}_weighted" for crit in criterios.keys() if f"{crit}_weighted" in df_calc.columns]
        df_calc["puntuacion_total"] = df_calc[weighted_cols].sum(axis=1)
        
        # Normalizar la puntuación total a escala 0-100
        max_score = df_calc["puntuacion_total"].max()
        df_calc["puntuacion_final"] = (df_calc["puntuacion_total"] / max_score) * 100
        
        # Usar df_calc para el mapa
        df_mapa = df_calc
        valor_mostrar = "puntuacion_final"
        titulo = "Puntuación Final por Estado"
        color_scale = "viridis"
    else:
        # Mostrar el indicador seleccionado
        df_mapa = df
        valor_mostrar = indicador
        titulo = f"{criterios[indicador]} por Estado"
        color_scale = "viridis"
    
    # Preparar datos para el mapa
    data_mapa = []
    
    # Añadir coordenadas de latitud y longitud al dataframe
    for estado in df_mapa["Estado"]:
        if estado in mexico_states_coords:
            idx = df_mapa[df_mapa["Estado"] == estado].index
            df_mapa.loc[idx, "lat"] = mexico_states_coords[estado][0]
            df_mapa.loc[idx, "lon"] = mexico_states_coords[estado][1]
    
    # Crear mapa de burbujas
    fig = px.scatter_mapbox(
        df_mapa,
        lat="lat",
        lon="lon",
        color=valor_mostrar,
        size=valor_mostrar,
        hover_name="Estado",
        hover_data=[valor_mostrar],
        color_continuous_scale=color_scale,
        size_max=25,
        zoom=4,
        center={"lat": 23.6345, "lon": -102.5528},  # Centro de México
        mapbox_style="carto-positron",
        title=titulo
    )
    
    # Añadir etiquetas de estados
    for estado in df_mapa["Estado"]:
        if estado in mexico_states_coords:
            valor = df_mapa[df_mapa["Estado"] == estado][valor_mostrar].values[0]
            fig.add_trace(
                go.Scattermapbox(
                    lat=[mexico_states_coords[estado][0]],
                    lon=[mexico_states_coords[estado][1]],
                    mode="text",
                    text=[estado],
                    textfont=dict(size=10, color="black"),
                    showlegend=False,
                    hoverinfo="none"
                )
            )
    
    fig.update_layout(
        height=550,
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )
    
    return fig

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True) 