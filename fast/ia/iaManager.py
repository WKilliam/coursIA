from models.models import Data
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
    def __init__(self):
        pass

    def predictM(self, data: Data):
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
        strings = []
        # Affichage des probabilités de chaque catégorie
        for i, category in enumerate(categories_uniques):
            print(f"Prédiction de la catégorie {category} la plus probable : {predicted_probabilities[i] * 100:.2f}%")

        return strings

    def predictD(self, data: Data):
        predict = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36]
        return predict
