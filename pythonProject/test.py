from keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler

# Charger le modèle sauvegardé
loaded_model = load_model('version1/modele_criminalite_v1.h5')

# Créer une instance de StandardScaler
scaler = StandardScaler()

# Données de test (à remplacer par vos propres données)
custom_data = np.array([
    [3, -122.4068, 37.7980, 19],
    [3, -122.4608, 37.7122, 17]
])

# Normaliser les nouvelles données
custom_data_scaled = scaler.fit_transform(custom_data)

# Utiliser le modèle pour faire des prédictions sur les nouvelles données
predictions_loaded_model = loaded_model.predict(custom_data_scaled)

# Afficher les prédictions
for i, category_probabilities in enumerate(predictions_loaded_model):
    print(f"Pourcentage de la catégorie {i+1}:")
    for j, probability in enumerate(category_probabilities):
        formatted_percentage = probability * 100.0
        print(f"Sous-catégorie {j + 1}: {formatted_percentage:.2f}%")