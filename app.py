import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mexico_states import mexico_states_coords, mexico_states_iso
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

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
            html.H1("Dashboard de Análisis de Estados de México", className="text-center my-4"),
            html.P("Este dashboard permite analizar y comparar los estados de México según diferentes criterios económicos y sociales.", 
                   className="text-center text-muted mb-4"),
        ])
    ]),
    
    # Botón para mostrar/ocultar la sección de ponderación
    dbc.Row([
        dbc.Col([
            dbc.Button(
                "Mostrar/Ocultar Ponderación", 
                id="toggle-ponderacion", 
                color="secondary", 
                className="mb-3 w-100"
            ),
        ], width=12)
    ]),
    
    # Sección de ponderación (ahora con ID para poder ocultarla)
    dbc.Row([
        dbc.Col([
            dbc.Collapse(
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Ponderación de Criterios", className="card-title"),
                        html.Div([
                            html.Div([
                                html.Label(f"{desc}:"),
                                dcc.Slider(
                                    id=f"slider-{crit}",
                                    min=0,
                                    max=10,
                                    step=1,
                                    value=5,
                                    marks={i: str(i) for i in range(0, 11)},
                                    className="mb-4"
                                )
                            ]) for crit, desc in criterios.items()
                        ]),
                        html.Div([
                            html.Label("Criterios a invertir (menor es mejor):"),
                            dbc.Checklist(
                                id="checklist-invertir",
                                options=[
                                    {"label": "Días para trámites de apertura de empresa", "value": "Días_tramites_abrir_empresa"}
                                ],
                                value=["Días_tramites_abrir_empresa"],
                                className="mb-3"
                            ),
                        ]),
                        dbc.Button("Calcular", id="btn-calcular", color="primary", className="mt-3"),
                    ])
                ], className="mb-4"),
                id="collapse-ponderacion",
                is_open=True,
            )
        ], width=12)
    ]),
    
    # Visualizaciones principales - Reorganizadas para mejor distribución
    dbc.Row([
        # Mapa de México - Ahora ocupa la mitad superior
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Mapa de México", className="card-title"),
                    html.Div([
                        html.Label("Seleccione indicador:"),
                        dcc.Dropdown(
                            id="dropdown-mapa",
                            options=[{"label": desc, "value": crit} for crit, desc in criterios.items()] + 
                                    [{"label": "Puntuación Final", "value": "puntuacion_final"}],
                            value="puntuacion_final",
                            clearable=False,
                            className="mb-3"
                        ),
                    ]),
                    dcc.Graph(id="graph-mapa", figure={}, style={"height": "500px"})
                ])
            ], className="mb-4")
        ], width=12, lg=6),
        
        # Ranking de Estados - Ahora ocupa la otra mitad superior
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Ranking de Estados", className="card-title"),
                    dcc.Graph(id="graph-ranking", figure={}, style={"height": "400px"}),
                    html.Div(id="top-estados", className="mt-3")
                ])
            ], className="mb-4")
        ], width=12, lg=6),
    ]),
    
    dbc.Row([
        # Comparación de Estados - Ahora ocupa la mitad inferior
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Comparación de Estados", className="card-title"),
                    html.Div([
                        html.Label("Seleccione estados a comparar:"),
                        dcc.Dropdown(
                            id="dropdown-comparar",
                            options=[{"label": estado, "value": estado} for estado in df['Estado'].unique()],
                            value=df['Estado'].unique()[:5].tolist(),
                            multi=True,
                            className="mb-3"
                        ),
                    ]),
                    dcc.Graph(id="graph-comparar", figure={}, style={"height": "400px"})
                ])
            ], className="mb-4")
        ], width=12, lg=6),
        
        # Indicadores por Estado - Ahora ocupa la otra mitad inferior
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Indicadores por Estado", className="card-title"),
                    html.Div([
                        html.Label("Seleccione indicador:"),
                        dcc.Dropdown(
                            id="dropdown-indicador",
                            options=[{"label": desc, "value": crit} for crit, desc in criterios.items()],
                            value="PIB_Estatal",
                            clearable=False,
                            className="mb-3"
                        ),
                    ]),
                    dcc.Graph(id="graph-indicador", figure={}, style={"height": "400px"})
                ])
            ], className="mb-4")
        ], width=12, lg=6),
    ]),
    
    # Información del proyecto y enlaces
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Acerca de este proyecto", className="card-title"),
                    html.P("Dashboard desarrollado para análisis de estados de México basado en criterios económicos y sociales."),
                    html.P([
                        "Código fuente disponible en ",
                        html.A("GitHub", href="https://github.com/yourusername/mexico-dashboard", target="_blank")
                    ]),
                ])
            ])
        ], width=12)
    ]),
    
], fluid=True)

# Callback para mostrar/ocultar la sección de ponderación
@app.callback(
    Output("collapse-ponderacion", "is_open"),
    [Input("toggle-ponderacion", "n_clicks")],
    [State("collapse-ponderacion", "is_open")]
)
def toggle_ponderacion(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

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
        invertir = args[-1] if args[-1] else []  # Último argumento (checklist)
    
    try:
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
        
        # Ordenar por puntuación final
        df_ranking = df_calc.sort_values("puntuacion_final", ascending=False)
        
        # Crear gráfico de barras para el ranking
        fig = px.bar(
            df_ranking,
            y="Estado",
            x="puntuacion_final",
            orientation='h',
            title="Ranking de Estados por Puntuación Final",
            labels={"puntuacion_final": "Puntuación (0-100)", "Estado": ""},
            color="puntuacion_final",
            color_continuous_scale="viridis",
            text="puntuacion_final"
        )
        
        # Mejorar el diseño del gráfico
        fig.update_traces(
            texttemplate='%{text:.1f}',
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Puntuación: %{x:.1f}<extra></extra>'
        )
        
        fig.update_layout(
            xaxis_title="Puntuación (0-100)",
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(l=0, r=10, t=30, b=0),
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Crear tarjetas para los top 5 estados
        top_estados = df_ranking.head(5)
        
        # Identificar fortalezas para cada estado
        fortalezas = {}
        for _, row in top_estados.iterrows():
            estado = row["Estado"]
            fortalezas[estado] = []
            for crit, desc in criterios.items():
                if f"{crit}_norm" in df_calc.columns and row[f"{crit}_norm"] > 0.7:
                    fortalezas[estado].append(desc)
        
        top_estados_cards = html.Div([
            html.H5("Top 5 Estados Recomendados", className="mb-3"),
            html.Div([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5(f"{i+1}. {row['Estado']}", className="mb-0")
                    ], className="bg-primary text-white"),
                    dbc.CardBody([
                        html.P(f"Puntuación: {row['puntuacion_final']:.2f}", className="card-text"),
                        html.P("Fortalezas:", className="mt-2 mb-1 fw-bold") if fortalezas.get(row['Estado'], []) else None,
                        html.Ul([
                            html.Li(fortaleza) for fortaleza in fortalezas.get(row['Estado'], [])
                        ]) if fortalezas.get(row['Estado'], []) else html.P("No se identificaron fortalezas destacadas", className="text-muted"),
                    ])
                ], className="mb-3") for i, (_, row) in enumerate(top_estados.iterrows())
            ])
        ])
        
        return fig, top_estados_cards
    
    except Exception as e:
        print(f"Error en actualizar_ranking: {e}")
        # Devolver un gráfico de error
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error al generar el ranking: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        error_message = html.Div([
            html.H5("Error al generar el ranking", className="text-danger"),
            html.P(f"Detalles: {str(e)}", className="text-muted")
        ])
        
        return fig, error_message

# Callback para actualizar el gráfico de indicadores por estado
@app.callback(
    Output("graph-indicador", "figure"),
    [Input("dropdown-indicador", "value")]
)
def actualizar_indicador(indicador):
    try:
        if indicador is None:
            # Si no hay indicador seleccionado, mostrar un gráfico vacío
            fig = go.Figure()
            fig.add_annotation(
                text="Seleccione un indicador",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        
        # Ordenar por el indicador seleccionado
        df_sorted = df.sort_values(indicador, ascending=False)
        
        # Crear gráfico de barras
        fig = px.bar(
            df_sorted,
            y="Estado",
            x=indicador,
            orientation='h',
            title=f"{criterios[indicador]} por Estado",
            labels={indicador: criterios[indicador], "Estado": ""},
            color=indicador,
            color_continuous_scale="viridis",
            text=indicador
        )
        
        # Mejorar el diseño del gráfico
        fig.update_traces(
            texttemplate='%{text:.1f}',
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>%{x:.1f}<extra></extra>'
        )
        
        fig.update_layout(
            xaxis_title=criterios[indicador],
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(l=0, r=10, t=30, b=0),
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    except Exception as e:
        print(f"Error en actualizar_indicador: {e}")
        # Devolver un gráfico de error
        fig = go.Figure()
        fig.add_annotation(
            text="Error al generar el gráfico. Por favor, intente de nuevo.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

# Callback para actualizar la comparación de estados
@app.callback(
    Output("graph-comparar", "figure"),
    [Input("dropdown-comparar", "value"),
     Input("btn-calcular", "n_clicks")],
    [State(f"slider-{crit}", "value") for crit in criterios.keys()] + 
     [State("checklist-invertir", "value")]
)
def actualizar_comparacion(estados, n_clicks, *args):
    try:
        if not estados or len(estados) == 0:
            # Si no hay estados seleccionados, mostrar un gráfico vacío
            fig = go.Figure()
            fig.add_annotation(
                text="Seleccione estados para comparar",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        
        if n_clicks is None:
            # Valores predeterminados para la carga inicial
            pesos = [5] * len(criterios)
            invertir = ["Días_tramites_abrir_empresa"]
        else:
            pesos = args[:-1]  # Todos los argumentos excepto el último (checklist)
            invertir = args[-1] if args[-1] else []  # Último argumento (checklist)
        
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
        
        # Filtrar solo los estados seleccionados
        df_estados = df_calc[df_calc["Estado"].isin(estados)]
        
        # Verificar si hay datos
        if df_estados.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No hay datos disponibles para los estados seleccionados",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        
        # Crear una copia para evitar SettingWithCopyWarning
        df_radar = df_estados.copy()
        
        # Crear gráfico de radar
        fig = go.Figure()
        
        # Definir categorías para el radar (criterios)
        categorias = list(criterios.values())
        
        # Generar una paleta de colores distinta para cada estado
        colores = px.colors.qualitative.Plotly[:len(estados)]
        
        # Añadir cada estado al gráfico de radar
        for i, estado in enumerate(df_radar["Estado"].unique()):
            valores = []
            for crit in criterios.keys():
                if f"{crit}_norm" in df_radar.columns:
                    # Obtener el valor normalizado para este criterio y estado
                    estado_data = df_radar[df_radar["Estado"] == estado]
                    if not estado_data.empty and not estado_data[f"{crit}_norm"].isnull().all():
                        valor = estado_data[f"{crit}_norm"].values[0]
                        valores.append(valor)
                    else:
                        valores.append(0)  # Valor por defecto si no hay datos
            
            # Solo añadir al gráfico si hay valores
            if valores:
                # Añadir el primer valor al final para cerrar el polígono
                valores.append(valores[0])
                categorias_cerradas = categorias + [categorias[0]]
                
                # Obtener color con transparencia en formato rgba
                color = colores[i % len(colores)]
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                
                fig.add_trace(go.Scatterpolar(
                    r=valores,
                    theta=categorias_cerradas,
                    fill='toself',
                    name=estado,
                    line=dict(color=color, width=2),
                    fillcolor=f'rgba({r}, {g}, {b}, 0.3)'
                ))
        
        # Mejorar el diseño del gráfico
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10),
                    gridcolor='rgba(0,0,0,0.1)',
                    linecolor='rgba(0,0,0,0.1)'
                ),
                angularaxis=dict(
                    tickfont=dict(size=10, family="Arial, sans-serif"),
                    rotation=90,
                    direction="clockwise",
                    gridcolor='rgba(0,0,0,0.1)',
                    linecolor='rgba(0,0,0,0.1)'
                ),
                bgcolor='rgba(255,255,255,0.8)'
            ),
            title="Comparación de Estados por Criterio (Valores Normalizados)",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=10, family="Arial, sans-serif")
            ),
            margin=dict(l=40, r=40, t=40, b=60),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    except Exception as e:
        print(f"Error en actualizar_comparacion: {e}")
        # Devolver un gráfico de error
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error al generar el gráfico de comparación: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

# Callback para actualizar el mapa de México
@app.callback(
    Output("graph-mapa", "figure"),
    [Input("dropdown-mapa", "value"),
     Input("btn-calcular", "n_clicks")],
    [State(f"slider-{crit}", "value") for crit in criterios.keys()] + 
     [State("checklist-invertir", "value")]
)
def actualizar_mapa(indicador, n_clicks, *args):
    try:
        if indicador is None:
            # Si no hay indicador seleccionado, mostrar un mapa vacío
            fig = go.Figure()
            fig.add_annotation(
                text="Seleccione un indicador para visualizar en el mapa",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        
        # Determinar si mostrar el indicador seleccionado o la puntuación final
        if indicador == "puntuacion_final":
            # Usar los mismos cálculos que en actualizar_ranking
            if n_clicks is None:
                # Valores predeterminados para la carga inicial
                pesos = [5] * len(criterios)
                invertir = ["Días_tramites_abrir_empresa"]
            else:
                pesos = args[:-1]  # Todos los argumentos excepto el último (checklist)
                invertir = args[-1] if args[-1] else []  # Último argumento (checklist)
            
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
            if max_score > 0:
                df_calc["puntuacion_final"] = (df_calc["puntuacion_total"] / max_score) * 100
            else:
                df_calc["puntuacion_final"] = 0
            
            # Usar df_calc para el mapa
            df_mapa = df_calc
            valor_mostrar = "puntuacion_final"
            titulo = "Puntuación Final por Estado"
        else:
            # Mostrar el indicador seleccionado
            df_mapa = df
            valor_mostrar = indicador
            titulo = f"{criterios.get(indicador, indicador)} por Estado"
        
        # Crear un mapa de México usando choropleth
        fig = go.Figure()
        
        # Crear un colormap para los valores
        min_val = df_mapa[valor_mostrar].min()
        max_val = df_mapa[valor_mostrar].max()
        
        # Añadir datos para cada estado
        for estado in df_mapa["Estado"].unique():
            if estado in mexico_states_coords:
                valor = df_mapa[df_mapa["Estado"] == estado][valor_mostrar].values[0]
                
                # Normalizar el valor para el color (0-1)
                valor_norm = (valor - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                
                # Obtener color del mapa de colores viridis
                color_rgba = f"rgba(68, 1, 84, {0.3 + 0.7 * valor_norm})" if valor_norm < 0.25 else \
                             f"rgba(59, 82, 139, {0.3 + 0.7 * valor_norm})" if valor_norm < 0.5 else \
                             f"rgba(33, 144, 141, {0.3 + 0.7 * valor_norm})" if valor_norm < 0.75 else \
                             f"rgba(94, 201, 98, {0.3 + 0.7 * valor_norm})"
                
                # Añadir marcador para el estado
                fig.add_trace(go.Scattergeo(
                    lon=[mexico_states_coords[estado][1]],
                    lat=[mexico_states_coords[estado][0]],
                    text=[f"{estado}: {valor:.2f}"],
                    mode='markers',
                    marker=dict(
                        size=15 + (valor_norm * 20),  # Tamaño basado en el valor
                        color=color_rgba,
                        line=dict(width=1, color='black')
                    ),
                    name=estado,
                    hoverinfo='text'
                ))
        
        # Configurar el layout del mapa
        fig.update_geos(
            visible=False,
            resolution=50,
            scope="north america",
            showcountries=True,
            countrycolor="Black",
            showsubunits=True,
            subunitcolor="Blue",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            showocean=True,
            oceancolor="rgb(221, 236, 255)",
            lataxis=dict(range=[14, 33]),  # Ajustar para México
            lonaxis=dict(range=[-118, -86]),  # Ajustar para México
        )
        
        fig.update_layout(
            title=dict(
                text=titulo,
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top"
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            height=500,
            showlegend=False,
            geo=dict(
                projection_type="mercator",
                center=dict(lat=23, lon=-102),  # Centro de México
                projection_scale=5
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        return fig
    
    except Exception as e:
        print(f"Error en actualizar_mapa: {e}")
        # Devolver un mapa de error
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error al generar el mapa: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8070) 