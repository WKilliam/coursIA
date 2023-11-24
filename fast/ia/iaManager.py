from fast.models.models import Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import seaborn as sns
# !pip install category-encoders
# !pip install tensorflow
# !pip install jupyter-tensorboard
import numpy as np
from category_encoders import OrdinalEncoder
from sklearn.linear_model import LogisticRegression


class iaManager:
    def predictM(self, data: Data):
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

            'Sexual Crimes': ['PROSTITUTION', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE',
                              'PORNOGRAPHY/OBSCENE MAT',
                              'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', ],

            'Financial Crimes': ['FRAUD', 'BRIBERY', 'EMBEZZLEMENT', 'BAD CHECKS', 'FORGERY/COUNTERFEITING',
                                 'EXTORTION'],

            'Depencense Crimes': ['DRUNKENNESS', 'LIQUOR LAWS', 'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC',
                                  'DRUNKENNESS',
                                  'LIQUOR LAWS'],

            'Vehicle Crimes': ['DRIVING UNDER THE INFLUENCE', 'VEHICLE THEFT', 'RECOVERED VEHICLE',
                               'DISORDERLY CONDUCT'],

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

        data = balanced_data

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

        ####### Model 1 #######
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix

        # Sélection des colonnes souhaitées pour les entrées du modèle
        selected_columns = ['DayOfWeek', 'X', 'Y', 'Hour']
        X = data[selected_columns]
        y = data['Category']
        #
        # # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        #
        # # Create a RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=20)

        # Entraînement du modèle
        clf.fit(X_train, y_train)

        # Validation croisée (cross-validation)
        scores = cross_val_score(clf, X, y, cv=5)  # Utilisation de 5 folds pour la validation croisée

        # Affichage des scores de validation croisée
        print("Scores de validation croisée :", scores)
        print("Moyenne des scores :", scores.mean())

        # Prédictions sur l'ensemble de test
        predictions = clf.predict(X_test)

        # Évaluation du modèle sur l'ensemble de test
        print("Accuracy :", clf.score(X_test, y_test))
        print("Confusion Matrix :\n", confusion_matrix(y_test, predictions))
        print("Classification Report :\n", classification_report(y_test, predictions))

        # Nouvelles données à prédire
        new_data = [
            # [4, 37.7749, -122.4194, 8],
            [2, 37.7996, -122.4000, 4]
        ]

        # Obtention des probabilités des prédictions pour les nouvelles données
        predicted_probabilities = clf.predict_proba(new_data)[0]

        categories_uniques = data['Category'].unique()

        # Affichage des probabilités de chaque catégorie dans un tableau
        strings = []

        # un tableau de string qui contient les probabilités de chaque catégorie
        for i, category in enumerate(categories_uniques):
            strings.append(f"{category} : {predicted_probabilities[i]}")
            print(f"{category} : {predicted_probabilities[i]}")

        return strings

    def predictD(self, data: Data):
        predict = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36]
        return predict
