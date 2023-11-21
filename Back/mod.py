import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import seaborn as sns
from category_encoders import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data = pd.read_csv('sample_data/train.csv')
print(data.head())
columns_deleted = ['Descript','Resolution','Address']
data = data.drop(columns_deleted, axis=1)
print(data.head())
data
fig = px.scatter(data, x="X", y="Y", color="Category")
fig.show()
categories_uniques = data['Category'].unique()
print(categories_uniques)
fig = px.scatter(data, x="X", y="Y", color="Category")
fig.show()
# Filtrer les données pour inclure uniquement la catégorie 'LARCENY/THEFT'
larceny_theft_data = data[data['Category'] == 'LARCENY/THEFT']
# Créer un nuage de points avec la catégorie 'LARCENY/THEFT'
fig = px.scatter(larceny_theft_data, x="X", y="Y", color="Category")
fig.show()
# Filtrer les données pour inclure uniquement la catégorie 'LARCENY/THEFT'
larceny_theft_data = data[data['Category'] == 'LARCENY/THEFT']
# Créer un nuage de points avec la catégorie 'LARCENY/THEFT'
fig = px.scatter(larceny_theft_data, x="Y", y="X", color="Category")
fig.show()
categorical_features = [col for col in data.columns if data[col].dtypes == 'object']
encoder = OrdinalEncoder(cols=categorical_features).fit(data)
data2 = encoder.transform(data)
data2