import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import requests

# Initialisation de l'application Dash
app = dash.Dash(__name__)

# URL de l'API FastAPI
fastapi_url = "http://127.0.0.1:8000"  # Mettez à jour l'URL 

# Mise en page de l'application
app.layout = html.Div([
    html.H1("SAN FRANCISCO"),
    html.H3("Crime Prediction"),
    
    html.Div([
        # Formulaire
        html.Div([
            html.Label("X:"),
            dcc.Input(id="input-x", type="number", value=0),
            html.Label("Y:"),
            dcc.Input(id="input-y", type="number", value=0),
            html.Label("DayOfWeek:"),
            dcc.Input(id="input-dayofweek", type="text", value=""),
            html.Label("Hour:"),
            dcc.Input(id="input-hour", type="number", value=0),
            html.Button(id="submit-button", n_clicks=0, children="SHOW")
        ], style={"width": "48%", "display": "inline-block"}),
        
        # Carte
        html.Div([
            dcc.Graph(id="crime-map")
        ], style={"width": "48%", "display": "inline-block"})
    ]),
])

# Callback pour mettre à jour la carte en fonction des prédictions
@app.callback(
    Output("crime-map", "figure"),
    [Input("submit-button", "n_clicks")],
    [
        dash.dependencies.State("input-x", "value"),
        dash.dependencies.State("input-y", "value"),
        dash.dependencies.State("input-dayofweek", "value"),
        dash.dependencies.State("input-hour", "value"),
    ],
)
def update_map(n_clicks, x, y, dayofweek, hour):
    # Appel à l'API FastAPI
    payload = {
        "X": x,
        "Y": y,
        "DayOfWeek": dayofweek,
        "Hour": hour,
    }
    response = requests.post(f"{fastapi_url}/model/predict/crime_prediction_model", json=payload)
    prediction = response.json()["prediction"]

    # Affichage de la carte
    fig = px.scatter_geo(
        locations=[[x, y]],
        text=[f"Prediction: {prediction}"],
        projection="natural earth",
        title="San Francisco Crime Prediction",
    )
    return fig

# Exécution de l'application
if __name__ == "__main__":
    app.run_server(debug=True)
