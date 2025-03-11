import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mexico_states import mexico_states_coords, mexico_states_iso
import dash_draggable

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
            dbc.Button(
                "Personalizar Dashboard", 
                id="btn-personalizar", 
                color="primary", 
                className="mb-3 w-100"
            ),
            dbc.Collapse(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Opciones de Personalización", className="card-title"),
                        html.Div([
                            html.Label("Mostrar/Ocultar Elementos:"),
                            dbc.Checklist(
                                id="checklist-mostrar",
                                options=[
                                    {"label": "Panel de Ponderación", "value": "ponderacion"},
                                    {"label": "Ranking de Estados", "value": "ranking"},
                                    {"label": "Mapa de México", "value": "mapa"},
                                    {"label": "Comparación de Estados", "value": "comparar"},
                                    {"label": "Indicadores por Estado", "value": "indicador"},
                                    {"label": "Matriz de Correlación", "value": "correlacion"},
                                    {"label": "Top 5 Estados", "value": "top-estados"}
                                ],
                                value=["ponderacion", "ranking", "mapa", "comparar", "indicador", "correlacion", "top-estados"],
                                inline=False,
                                className="mb-3"
                            ),
                        ]),
                        html.Div([
                            html.Label("Tamaño de Gráficas:"),
                            dbc.RadioItems(
                                id="radio-tamano",
                                options=[
                                    {"label": "Pequeño", "value": "small"},
                                    {"label": "Mediano", "value": "medium"},
                                    {"label": "Grande", "value": "large"}
                                ],
                                value="medium",
                                inline=True,
                                className="mb-3"
                            ),
                        ]),
                        html.Div([
                            html.Label("Modo Presentación:"),
                            dbc.Checklist(
                                id="switch-modo-presentacion",
                                options=[{"label": "Activar", "value": True}],
                                value=[],
                                switch=True,
                                className="mb-3"
                            ),
                        ]),
                        html.Div([
                            html.Label("Modo Arrastrar y Redimensionar:"),
                            dbc.Checklist(
                                id="switch-modo-draggable",
                                options=[{"label": "Activar", "value": True}],
                                value=[],
                                switch=True,
                                className="mb-3"
                            ),
                        ]),
                        dbc.Button(
                            "Guardar Configuración", 
                            id="btn-guardar-config", 
                            color="success", 
                            className="mt-2"
                        ),
                    ]),
                    className="mb-3"
                ),
                id="collapse-personalizar",
                is_open=False,
            ),
        ])
    ]),
    
    # Contenedor principal que cambiará entre modo normal y modo draggable
    html.Div(id="contenedor-principal", children=[
        # Contenido normal (no draggable)
        html.Div(id="contenido-normal", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("Ponderación de Criterios", className="card-title"),
                            html.Div(id="panel-ponderacion", children=[
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
                            dbc.Button("Calcular", id="btn-calcular", color="primary", className="mt-2")
                        ]),
                        className="mb-4 shadow-sm"
                    ),
                ], id="col-ponderacion", width=3),
                
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H4("Ranking de Estados", className="card-title"),
                                    dcc.Graph(id="graph-ranking", figure={}, config={'displayModeBar': False})
                                ]),
                                className="mb-4 shadow-sm h-100"
                            ),
                        ], id="col-ranking", width=6),
                        
                        dbc.Col([
                            dbc.Card(
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
                                    dcc.Graph(id="graph-mapa", figure={}, config={'displayModeBar': False})
                                ]),
                                className="mb-4 shadow-sm h-100"
                            ),
                        ], id="col-mapa", width=6),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
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
                                    dcc.Graph(id="graph-comparar", figure={}, config={'displayModeBar': False})
                                ]),
                                className="mb-4 shadow-sm h-100"
                            ),
                        ], id="col-comparar", width=6),
                        
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H4("Indicadores por Estado", className="card-title"),
                                    html.Div([
                                        html.Label("Seleccione indicador:"),
                                        dcc.Dropdown(
                                            id="dropdown-indicador",
                                            options=[{"label": desc, "value": crit} for crit, desc in criterios.items()],
                                            value=list(criterios.keys())[0],
                                            clearable=False,
                                            className="mb-3"
                                        ),
                                    ]),
                                    dcc.Graph(id="graph-indicador", figure={}, config={'displayModeBar': False})
                                ]),
                                className="mb-4 shadow-sm h-100"
                            ),
                        ], id="col-indicador", width=6),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H4("Matriz de Correlación", className="card-title"),
                                    dcc.Graph(id="graph-correlacion", figure={}, config={'displayModeBar': False})
                                ]),
                                className="mb-4 shadow-sm h-100"
                            ),
                        ], id="col-correlacion", width=6),
                        
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H4("Top 5 Estados Recomendados", className="card-title"),
                                    html.Div(id="top-estados", className="mt-3")
                                ]),
                                className="mb-4 shadow-sm h-100"
                            ),
                        ], id="col-top-estados", width=6),
                    ]),
                ], id="col-resultados", width=9),
            ], id="container-principal", className="mb-4"),
        ]),
        
        # Contenido draggable
        html.Div(id="contenido-draggable", style={"display": "none"}, children=[
            dash_draggable.ResponsiveGridLayout(
                id="grid-layout",
                layouts={
                    "lg": [
                        {"i": "ponderacion", "x": 0, "y": 0, "w": 3, "h": 12, "static": False},
                        {"i": "ranking", "x": 3, "y": 0, "w": 4, "h": 6, "static": False},
                        {"i": "mapa", "x": 7, "y": 0, "w": 5, "h": 6, "static": False},
                        {"i": "comparar", "x": 3, "y": 6, "w": 4, "h": 6, "static": False},
                        {"i": "indicador", "x": 7, "y": 6, "w": 5, "h": 6, "static": False},
                        {"i": "correlacion", "x": 3, "y": 12, "w": 4, "h": 6, "static": False},
                        {"i": "top-estados", "x": 7, "y": 12, "w": 5, "h": 6, "static": False}
                    ]
                },
                children=[
                    html.Div(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Ponderación de Criterios", className="card-title"),
                                html.Div(id="panel-ponderacion-drag", children=[
                                    html.Div([
                                        html.Label(f"{desc}:"),
                                        dcc.Slider(
                                            id=f"slider-drag-{crit}",
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
                                        id="checklist-invertir-drag",
                                        options=[
                                            {"label": "Días para trámites de apertura de empresa", "value": "Días_tramites_abrir_empresa"}
                                        ],
                                        value=["Días_tramites_abrir_empresa"],
                                        className="mb-3"
                                    ),
                                ]),
                                dbc.Button("Calcular", id="btn-calcular-drag", color="primary", className="mt-2")
                            ]),
                            className="h-100 shadow-sm"
                        ),
                        key="ponderacion",
                        className="grid-item"
                    ),
                    html.Div(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Ranking de Estados", className="card-title"),
                                dcc.Graph(id="graph-ranking-drag", figure={}, config={'displayModeBar': False})
                            ]),
                            className="h-100 shadow-sm"
                        ),
                        key="ranking",
                        className="grid-item"
                    ),
                    html.Div(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Mapa de México", className="card-title"),
                                html.Div([
                                    html.Label("Seleccione indicador:"),
                                    dcc.Dropdown(
                                        id="dropdown-mapa-drag",
                                        options=[{"label": desc, "value": crit} for crit, desc in criterios.items()] + 
                                                [{"label": "Puntuación Final", "value": "puntuacion_final"}],
                                        value="puntuacion_final",
                                        clearable=False,
                                        className="mb-3"
                                    ),
                                ]),
                                dcc.Graph(id="graph-mapa-drag", figure={}, config={'displayModeBar': False})
                            ]),
                            className="h-100 shadow-sm"
                        ),
                        key="mapa",
                        className="grid-item"
                    ),
                    html.Div(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Comparación de Estados", className="card-title"),
                                html.Div([
                                    html.Label("Seleccione estados a comparar:"),
                                    dcc.Dropdown(
                                        id="dropdown-comparar-drag",
                                        options=[{"label": estado, "value": estado} for estado in df['Estado'].unique()],
                                        value=df['Estado'].unique()[:5].tolist(),
                                        multi=True,
                                        className="mb-3"
                                    ),
                                ]),
                                dcc.Graph(id="graph-comparar-drag", figure={}, config={'displayModeBar': False})
                            ]),
                            className="h-100 shadow-sm"
                        ),
                        key="comparar",
                        className="grid-item"
                    ),
                    html.Div(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Indicadores por Estado", className="card-title"),
                                html.Div([
                                    html.Label("Seleccione indicador:"),
                                    dcc.Dropdown(
                                        id="dropdown-indicador-drag",
                                        options=[{"label": desc, "value": crit} for crit, desc in criterios.items()],
                                        value=list(criterios.keys())[0],
                                        clearable=False,
                                        className="mb-3"
                                    ),
                                ]),
                                dcc.Graph(id="graph-indicador-drag", figure={}, config={'displayModeBar': False})
                            ]),
                            className="h-100 shadow-sm"
                        ),
                        key="indicador",
                        className="grid-item"
                    ),
                    html.Div(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Matriz de Correlación", className="card-title"),
                                dcc.Graph(id="graph-correlacion-drag", figure={}, config={'displayModeBar': False})
                            ]),
                            className="h-100 shadow-sm"
                        ),
                        key="correlacion",
                        className="grid-item"
                    ),
                    html.Div(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Top 5 Estados Recomendados", className="card-title"),
                                html.Div(id="top-estados-drag", className="mt-3")
                            ]),
                            className="h-100 shadow-sm"
                        ),
                        key="top-estados",
                        className="grid-item"
                    ),
                ],
                gridCols={"lg": 12, "md": 10, "sm": 6, "xs": 4, "xxs": 2},
                className="draggable-layout"
            )
        ])
    ]),
    
    html.Footer([
        html.P("Dashboard desarrollado para ROMPEX © 2024", className="text-center")
    ], className="mt-4 py-3")
], fluid=True, className="px-4")

# Callback para mostrar/ocultar el panel de personalización
@app.callback(
    Output("collapse-personalizar", "is_open"),
    [Input("btn-personalizar", "n_clicks")],
    [State("collapse-personalizar", "is_open")]
)
def toggle_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Callback para cambiar entre modo normal y modo draggable
@app.callback(
    [Output("contenido-normal", "style"),
     Output("contenido-draggable", "style")],
    [Input("switch-modo-draggable", "value")]
)
def toggle_modo_draggable(modo_draggable):
    if modo_draggable and True in modo_draggable:
        return {"display": "none"}, {"display": "block"}
    else:
        return {"display": "block"}, {"display": "none"}

# Callback para mostrar/ocultar elementos del dashboard
@app.callback(
    [Output("col-ponderacion", "style"),
     Output("col-ranking", "style"),
     Output("col-mapa", "style"),
     Output("col-comparar", "style"),
     Output("col-indicador", "style"),
     Output("col-correlacion", "style"),
     Output("col-top-estados", "style")],
    [Input("checklist-mostrar", "value")]
)
def toggle_elementos(elementos_mostrados):
    estilo_visible = {"display": "block"}
    estilo_oculto = {"display": "none"}
    
    return [
        estilo_visible if "ponderacion" in elementos_mostrados else estilo_oculto,
        estilo_visible if "ranking" in elementos_mostrados else estilo_oculto,
        estilo_visible if "mapa" in elementos_mostrados else estilo_oculto,
        estilo_visible if "comparar" in elementos_mostrados else estilo_oculto,
        estilo_visible if "indicador" in elementos_mostrados else estilo_oculto,
        estilo_visible if "correlacion" in elementos_mostrados else estilo_oculto,
        estilo_visible if "top-estados" in elementos_mostrados else estilo_oculto
    ]

# Callback para ajustar el tamaño de las gráficas
@app.callback(
    [Output("graph-ranking", "style"),
     Output("graph-mapa", "style"),
     Output("graph-comparar", "style"),
     Output("graph-indicador", "style"),
     Output("graph-correlacion", "style")],
    [Input("radio-tamano", "value")]
)
def ajustar_tamano_graficas(tamano):
    alturas = {
        "small": "250px",
        "medium": "350px",
        "large": "450px"
    }
    
    altura = alturas.get(tamano, "350px")
    estilo = {"height": altura}
    
    return [estilo, estilo, estilo, estilo, estilo]

# Callback para activar el modo presentación
@app.callback(
    Output("contenedor-principal", "className"),
    [Input("switch-modo-presentacion", "value")]
)
def toggle_modo_presentacion(modo_presentacion):
    if modo_presentacion and True in modo_presentacion:
        return "bg-dark text-white px-4"
    else:
        return "px-4"

# Solución para el ciclo de dependencia: usar un enfoque de un solo sentido
# En lugar de sincronizar en ambas direcciones, sincronizamos solo del modo normal al modo draggable
@app.callback(
    [Output(f"slider-drag-{crit}", "value") for crit in criterios.keys()],
    [Input(f"slider-{crit}", "value") for crit in criterios.keys()],
    prevent_initial_call=True
)
def sync_sliders_normal_to_drag(*valores):
    return valores

# Callback para sincronizar los valores de los checklists
@app.callback(
    Output("checklist-invertir-drag", "value"),
    [Input("checklist-invertir", "value")],
    prevent_initial_call=True
)
def sync_checklist_normal_to_drag(valores):
    return valores

# Callback para sincronizar los dropdowns
@app.callback(
    Output("dropdown-mapa-drag", "value"),
    [Input("dropdown-mapa", "value")],
    prevent_initial_call=True
)
def sync_dropdown_mapa_normal_to_drag(valor):
    return valor

@app.callback(
    Output("dropdown-indicador-drag", "value"),
    [Input("dropdown-indicador", "value")],
    prevent_initial_call=True
)
def sync_dropdown_indicador_normal_to_drag(valor):
    return valor

@app.callback(
    Output("dropdown-comparar-drag", "value"),
    [Input("dropdown-comparar", "value")],
    prevent_initial_call=True
)
def sync_dropdown_comparar_normal_to_drag(valor):
    return valor

# Callback para sincronizar los resultados
@app.callback(
    [Output("graph-ranking-drag", "figure"),
     Output("top-estados-drag", "children")],
    [Input("graph-ranking", "figure"),
     Input("top-estados", "children")],
    prevent_initial_call=True
)
def sync_resultados_normal_to_drag(figura, top_estados):
    return figura, top_estados

@app.callback(
    [Output("graph-mapa-drag", "figure")],
    [Input("graph-mapa", "figure")],
    prevent_initial_call=True
)
def sync_mapa_normal_to_drag(figura):
    return [figura]

@app.callback(
    [Output("graph-comparar-drag", "figure")],
    [Input("graph-comparar", "figure")],
    prevent_initial_call=True
)
def sync_comparar_normal_to_drag(figura):
    return [figura]

@app.callback(
    [Output("graph-indicador-drag", "figure")],
    [Input("graph-indicador", "figure")],
    prevent_initial_call=True
)
def sync_indicador_normal_to_drag(figura):
    return [figura]

@app.callback(
    [Output("graph-correlacion-drag", "figure")],
    [Input("graph-correlacion", "figure")],
    prevent_initial_call=True
)
def sync_correlacion_normal_to_drag(figura):
    return [figura]

# Callback para calcular desde el modo draggable
@app.callback(
    Output("btn-calcular", "n_clicks"),
    [Input("btn-calcular-drag", "n_clicks")],
    [State("btn-calcular", "n_clicks")],
    prevent_initial_call=True
)
def sync_calcular_drag_to_normal(n_clicks_drag, n_clicks):
    if n_clicks_drag:
        return n_clicks_drag
    return n_clicks

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True) 