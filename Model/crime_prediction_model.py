import pandas as pd
from keras.src.layers import Dense
from keras.src.metrics import RootMeanSquaredError
from keras.src.models.cloning import Sequential
from sklearn.model_selection import train_test_split
# !pip install category-encoders
# !pip install tensorflow
# !pip install jupyter-tensorboard
from category_encoders import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.utils import to_categorical
from tensorflow.python.layers.core import Dropout

data = pd.read_csv('sample_data/train.csv')
columns_deleted = ['Descript', 'Resolution', 'Address', 'PdDistrict']
data = data.drop(columns_deleted, axis=1)
# Les valeurs connues pour la suppression
# category_a_supprimer = 'WARRANTS'
x_a_supprimer = -120.5
y_a_supprimer = 90
# Supprimer les lignes où les valeurs correspondent aux valeurs connues
# data = data[~((data['Category'] == category_a_supprimer) & (data['X'] == x_a_supprimer) & (data['Y'] == y_a_supprimer))]
data = data[~((data['X'] == x_a_supprimer) & (data['Y'] == y_a_supprimer))]
print(data['Category'].unique())
# Créer un dictionnaire pour les nouvelles catégories
new_categories = {
    'Person Crimes': ['OTHER OFFENSES', 'ASSAULT', 'RUNAWAY', 'KIDNAPPING', 'SUICIDE'],

    'Property Crimes': ['LARCENY/THEFT', 'VEHICLE THEFT', 'BURGLARY', 'STOLEN PROPERTY', 'RECOVERED VEHICLE',
                        'TRESPASS'],

    'Sexual Crimes': ['PROSTITUTION', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'PORNOGRAPHY/OBSCENE MAT',
                      'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', ],

    'Financial Crimes': ['FRAUD', 'BRIBERY', 'EMBEZZLEMENT', 'BAD CHECKS', 'FORGERY/COUNTERFEITING', 'EXTORTION'],

    'Depencense Crimes': ['DRUNKENNESS', 'LIQUOR LAWS', 'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS',
                          'LIQUOR LAWS'],

    'Vehicle Crimes': ['DRIVING UNDER THE INFLUENCE', 'VEHICLE THEFT', 'RECOVERED VEHICLE', 'DISORDERLY CONDUCT'],

    'Family Crimes': ['FAMILY OFFENSES', 'MISSING PERSON', 'RUNAWAY', 'FAMILY OFFENSES', 'MISSING PERSON'],

    'Society Crimes': ['VANDALISM', 'ROBBERY', 'WEAPON LAWS', 'ARSON', 'LOITERING', 'GAMBLING'],

    'Miscellaneous': ['SUSPICIOUS OCC', 'SECONDARY CODES', 'TREA', 'NON-CRIMINAL', 'WARRANTS']
}
# Créer un mapping inverse pour les nouvelles catégories
reverse_mapping = {val: key for key, values in new_categories.items() for val in values}
# Ajouter une nouvelle colonne "New_Category" basée sur les nouvelles catégories
data['Category'] = data['Category'].map(reverse_mapping)
# Vérifier le résultat en affichant un échantillon de données
print(data.head())
print(data['Category'].unique())
# Organiser la dataframe en fonction de la nouvelle colonne "New_Category"
data = data.sort_values('Category')

# Compter le nombre d'éléments pour chaque catégorie
counts_per_category = data['Category'].value_counts()
# Trouver la taille de la catégorie la plus petite
min_count = counts_per_category.min()
# Sélectionner un nombre équilibré d'éléments aléatoires pour chaque catégorie (basé sur la taille de la catégorie la plus petite)
balanced_data = pd.DataFrame()
for category, count in counts_per_category.items():
    category_data = data[data['Category'] == category].sample(min(count, min_count))
    balanced_data = pd.concat([balanced_data, category_data])
# Enregistrer ce nouvel ensemble de données équilibré dans un fichier CSV
balanced_data.to_csv('sample_data/balanced_dataset.csv', index=False)
# Charger à nouveau les données à partir du fichier nouvellement créé
balanced_data = pd.read_csv('sample_data/balanced_dataset.csv')
# Ramdomiser les données
balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)
print(min_count)

# data = balanced_data

# Mapper les catégories à partir des valeurs existantes de la colonne 'Category'
# data['Category'] = data['Category'].map(theme_categories).fillna(data['Category'])
# Convertir la colonne 'Dates' en format datetime
data['Dates'] = pd.to_datetime(data['Dates'])
# Créer de nouvelles colonnes pour la date et l'heure
data['Date'] = data['Dates'].dt.date
data['Time'] = data['Dates'].dt.time
# Supprimer la colonne 'Dates' maintenant qu'elle n'est plus nécessaire
data.drop(columns=['Dates'], inplace=True)
# Convertir la colonne 'Date' en composantes de date séparées
data['Year'] = pd.to_datetime(data['Date']).dt.year
data['Month'] = pd.to_datetime(data['Date']).dt.month
data['Day'] = pd.to_datetime(data['Date']).dt.day
# Convertir la colonne 'Time' en composantes d'heure séparées
data['Hour'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.hour
data['Minute'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.minute
data['Second'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.second
# Supprimer les colonnes d'origine 'Date' et 'Time'
data.drop(['Date', 'Time'], axis=1, inplace=True)
# Colonnes à encoder de façon ordinaire
ordinal_cols = ['Category', 'DayOfWeek']

# Créer un encodeur ordinal et appliquer la transformation
encoder = OrdinalEncoder(cols=ordinal_cols).fit(data)
data = encoder.transform(data)
# Enregistrer le nouveau fichier CSV sans modifier l'original
data.to_csv('sample_data/nouveau_fichier.csv', index=False)
# Filtrer les données pour inclure uniquement la catégorie 'Harassment'
larceny_theft_data = data[data['Category'] == 'Harassment']
# Créer un nuage de points avec la catégorie 'PROSTITUTION'
# fig = px.scatter(larceny_theft_data, x="X", y="Y", color="Category")
# fig.show()
# Vérifier s'il y a des valeurs manquantes (NaN) dans chaque colonne
missing_values = data.isnull().any()
data.isnull().sum()
# Afficher les colonnes contenant des valeurs manquantes (True) et le nombre de valeurs manquantes dans chaque colonne
print(missing_values)
print(data.isnull().sum())
print(data.head())
print(data.shape)
print(data.describe())
print(data.info())
print(data['Category'].unique())

#### Deep Learning (régression multinomiale) ####

# Sélection des colonnes pour les features (X) et la target variable (y)
X = data[['DayOfWeek', 'X', 'Y', 'Hour']]
y = data['Category']
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Normalisation des features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Créer le modèle
model = Sequential()
model.add(Dense(4, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(data['Category'].unique()), activation='softmax'))
# Compiler le modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Entraîner le modèle
model.fit(X_train_scaled, pd.get_dummies(y_train), epochs=2, batch_size=32,
          validation_data=(X_test_scaled, pd.get_dummies(y_test)))

# 6,3,-122.406842913454,37.7980587205991,2009,3,19,0,48,0

# Prédictions pour de nouvelles données personnalisées
custom_data = np.array([
    [3, -122.4068, 37.7980, 19],
    [3, -122.4608, 37.7122, 17]
])
# Normaliser les données personnalisées
custom_data_scaled = scaler.transform(custom_data)
# Prédictions pour les données personnalisées
predictions = model.predict(custom_data_scaled)
print(predictions)
# Affichage des prédictions (pourcentages de chance par catégorie)
for i, category_probabilities in enumerate(predictions):
    print(f"Pourcentage de la catégorie {i + 1}:")
    for j, probability in enumerate(category_probabilities):
        formatted_percentage = probability * 100.0
        print(f"    Sous-catégorie {j + 1}: {formatted_percentage:.2f}%")

# Sauvegarder le modèle Keras
# model.save('modele_criminalite.h5')
