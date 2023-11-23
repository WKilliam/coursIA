import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import requests

# Charger les données depuis le fichier crime.csv
crime_data = pd.read_csv("Frontend/train.csv")

# Initialiser l'application Dash
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# Mise en page de l'application
app.layout = html.Div([
    # En-tête
    html.Div([
        html.H1("SAN FRANCISCO", style={'textAlign': 'center'}),
        html.H2("Crime Prediction", style={'textAlign': 'center'}),
    ], className='header'),

    # Contenu principal
    html.Div([
        # Formulaire à gauche
        html.Div([
            html.Label('Longitude (X)'),
            dcc.Input(id='input-x', type='number', placeholder='Enter X'),
            
            html.Label('Latitude (Y)'),
            dcc.Input(id='input-y', type='number', placeholder='Enter Y'),
            
            html.Label('DayOfWeek'),
            dcc.Input(id='input-day', type='text', placeholder='Enter DayOfWeek'),
            
            html.Label('Hour'),
            dcc.Input(id='input-hour', type='number', placeholder='Enter Hour'),
            
            # Bouton "SHOW"
            html.Button('SHOW', id='show-button'),
            
            # Zone d'affichage des résultats (facultatif)
            html.Div(id='prediction-output')
        ], className='form-container'),

        # Carte de San Francisco à droite
        html.Div([
            dcc.Graph(
                id='crime-map',
                figure=px.scatter_mapbox(
                    crime_data,
                    lat='Y',
                    lon='X',
                    hover_name='Category',
                    hover_data=['Descript', 'Address'],
                    color='Category',
                    size_max=15,
                    zoom=10,
                    mapbox_style="carto-positron"
                )
            )
        ], className='map-container')
    ], className='main-container')
])

# Callback pour mettre à jour la carte en fonction des sélections du formulaire
@app.callback(
    Output('crime-map', 'figure'),
    [Input('input-x', 'value'),
     Input('input-y', 'value'),
     Input('input-day', 'value'),
     Input('input-hour', 'value'),
     Input('show-button', 'n_clicks')]
)
def update_map(input_x, input_y, input_day, input_hour, n_clicks):
    # Appeler l'API FastAPI ici avec les données du formulaire pour obtenir les résultats
    api_url = 'http://localhost:8000/predict'

    payload = {
        'X': float(input_x) if input_x else None,
        'Y': float(input_y) if input_y else None,
        'DayOfWeek': input_day,
        'Hour': int(input_hour) if input_hour else None,
    }

    response = requests.post(api_url, json=payload)

    # Gérer la réponse de l'API
    if response.status_code == 200:
        # Utilisez les résultats pour filtrer les données affichées sur la carte
        # Vous devrez extraire les informations pertinentes de la réponse JSON
        # ...

        # Exemple hypothétique :
        filtered_data = crime_data[crime_data['Prediction'] == response.json()['prediction']]

        # Mettre à jour la carte avec les données filtrées
        fig = px.scatter_mapbox(
            filtered_data,
            lat='Y',
            lon='X',
            hover_name='Category',
            hover_data=['Descript', 'Address', 'PdDistrict'],
            color='Category',
            size_max=15,
            zoom=10,
            mapbox_style="carto-positron"
        )

        return fig
    else:
        print(f"Erreur lors de l'appel à l'API : {response.text}")
        # Retourner une figure vide ou quelque chose qui indique une erreur
        return px.scatter()

# Exécuter l'application
if __name__ == '__main__':
    app.run_server(debug=True)