import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Charger les données depuis le fichier crime.csv
crime_data = pd.read_csv("Frontend/crime.csv")

# Initialiser l'application Dash
app = dash.Dash(__name__)

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
            html.Label('Type de Crime'),
            dcc.Dropdown(
                id='crime-type-dropdown',
                options=[
                    {'label': crime, 'value': crime} for crime in crime_data['Category'].unique()
                ],
                multi=True,
                value=[crime_data['Category'].unique()[0]]
            ),
            
            html.Label('Date'),
            dcc.Input(id='date-input', type='text', value='YYYY-MM-DD HH:MM:SS'),
            
            html.Label('Jour de la semaine'),
            dcc.Dropdown(
                id='day-of-week-dropdown',
                options=[
                    {'label': day, 'value': day} for day in crime_data['DayOfWeek'].unique()
                ],
                multi=True,
                value=[crime_data['DayOfWeek'].unique()[0]]
            ),
            
            html.Label('District de Police'),
            dcc.Dropdown(
                id='pd-district-dropdown',
                options=[
                    {'label': district, 'value': district} for district in crime_data['PdDistrict'].unique()
                ],
                multi=True,
                value=[crime_data['PdDistrict'].unique()[0]]
            ),
            
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

# CSS pour le style de la mise en page
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

# Callback pour mettre à jour la carte en fonction des sélections du formulaire
@app.callback(
    Output('crime-map', 'figure'),
    [Input('crime-type-dropdown', 'value'),
     Input('date-input', 'value'),
     Input('day-of-week-dropdown', 'value'),
     Input('pd-district-dropdown', 'value'),
     Input('show-button', 'n_clicks')]
)
def update_map(selected_crimes, selected_date, selected_days, selected_districts, n_clicks):
    # Appeler l'API FastAPI ici avec les données du formulaire pour obtenir les résultats (à implémenter dans la deuxième étape)
    # Utilisez les résultats pour filtrer les données affichées sur la carte
    filtered_data = crime_data[
        (crime_data['Category'].isin(selected_crimes)) &
        (crime_data['Dates'] == selected_date) &
        (crime_data['DayOfWeek'].isin(selected_days)) &
        (crime_data['PdDistrict'].isin(selected_districts))
    ]

    # Mettre à jour la carte avec les données filtrées
    fig = px.scatter_mapbox(
        filtered_data,
        lat='Y',
        lon='X',
        hover_name='Category',
        hover_data=['Descript', 'Address'],
        color='Category',
        size_max=15,
        zoom=10,
        mapbox_style="carto-positron"
    )

    return fig

# Exécuter l'application
if __name__ == '__main__':
    app.run_server(debug=True)