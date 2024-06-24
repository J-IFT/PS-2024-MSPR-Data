import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Ce script a pour but de traiter notre jeu de données pour qu'il soit le plus conforme et lisible possible

# Charger le jeu de données filtré
file_path = 'FichierFiltré/filtre_dataset_insecurite.csv'
df = pd.read_csv(file_path, delimiter=';')

# Remplacer '17' par '2017' dans la colonne Annee
df['Annee'] = df['Annee'].replace(17, '2017')
df['Annee'] = df['Annee'].replace(22, '2022')

# Vérifier la structure des données
print("Avant la préparation :")
print(df.info())
print("\nExemple de données avant la préparation :")
print(df.head())

# Gestion des données manquantes
# On utilise SimpleImputer pour remplacer les valeurs manquantes (ndiff) par NaN
df.replace('ndiff', np.nan, inplace=True)

# Conversion des colonnes numériques qui sont actuellement en format string à un format numérique
numeric_columns = ['Nombre_Incidents', 'Taux_Pour_Mille', 'Nombre_Logements']

# Assurez-vous que les colonnes à convertir sont bien de type 'object' (str) avant de les traiter
df[numeric_columns] = df[numeric_columns].astype(str).apply(lambda x: x.str.replace(',', '.'))

# Convertir en numérique après le remplacement des virgules par des points
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Gestion des valeurs aberrantes (outliers)
# Par exemple, remplacement des valeurs aberrantes par la médiane dans 'Nombre_Incidents'
median_incidents = df['Nombre_Incidents'].median()
df['Nombre_Incidents'] = np.where(df['Nombre_Incidents'] > 100, median_incidents, df['Nombre_Incidents'])

# Normalisation des données si nécessaire
# Utilisation de StandardScaler pour normaliser les colonnes numériques
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Création du dossier FichierTraité s'il n'existe pas
output_dir = 'FichierTraité'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Vérification après la préparation
print("\nAprès la préparation :")
print(df.info())
print("\nExemple de données après la préparation :")
print(df.head())

# Sauvegarde des données préparées
prepared_data_path = os.path.join(output_dir, 'traitement_dataset_insecurite.csv')
df.to_csv(prepared_data_path, index=False, sep=';')
print(f"\nDonnées préparées sauvegardées à {prepared_data_path}")
