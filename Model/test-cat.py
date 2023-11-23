import pandas as pd

# Charger le fichier CSV
data = pd.read_csv('sample_data/train.csv')  # Assurez-vous de remplacer 'votre_fichier.csv' par le nom réel de votre fichier

# Définir le dictionnaire des super-catégories
super_categories = {
    'Person Crimes': ['OTHER OFFENSES', 'ASSAULT', 'RUNAWAY', 'KIDNAPPING', 'SUICIDE'],
    'Property Crimes': ['LARCENY/THEFT', 'VEHICLE THEFT', 'BURGLARY', 'STOLEN PROPERTY', 'RECOVERED VEHICLE', 'TRESPASS'],
    'Sexual Crimes': ['PROSTITUTION', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'PORNOGRAPHY/OBSCENE MAT', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE'],
    'Financial Crimes': ['FRAUD', 'BRIBERY', 'EMBEZZLEMENT', 'BAD CHECKS', 'FORGERY/COUNTERFEITING', 'EXTORTION'],
    'Dependence Crimes': ['DRUNKENNESS', 'LIQUOR LAWS', 'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'LIQUOR LAWS'],
    'Vehicle Crimes': ['DRIVING UNDER THE INFLUENCE', 'VEHICLE THEFT', 'RECOVERED VEHICLE', 'DISORDERLY CONDUCT'],
    'Family Crimes': ['FAMILY OFFENSES', 'MISSING PERSON', 'RUNAWAY', 'FAMILY OFFENSES', 'MISSING PERSON'],
    'Society Crimes': ['VANDALISM', 'ROBBERY', 'WEAPON LAWS', 'ARSON', 'LOITERING', 'GAMBLING'],
    'Miscellaneous': ['SUSPICIOUS OCC', 'SECONDARY CODES', 'TREA', 'NON-CRIMINAL', 'WARRANTS']
}

# Créer une colonne pour les super-catégories en utilisant le dictionnaire
data['Super_Category'] = data['Category'].apply(lambda x: next((key for key, values in super_categories.items() if x in values), x))

# Calculer le pourcentage d'occurrence de chaque super-catégorie
super_category_percentages = data['Super_Category'].value_counts(normalize=True) * 100

# Afficher les résultats
print("Pourcentage d'occurrence de chaque super-catégorie :")
print(super_category_percentages)