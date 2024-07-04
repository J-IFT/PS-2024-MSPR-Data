import os
import pandas as pd
import numpy as np

# Charger le jeu de données filtré
file_path = 'FichierFiltré/filtre_dataset_insecurite.csv'
df = pd.read_csv(file_path, delimiter=';')

# Remplacer '17' par '2017' et '22' par '2022' dans la colonne Annee
df['Annee'] = df['Annee'].replace({17: '2017', 22: '2022'})

# Gestion des données manquantes
df.replace('', np.nan, inplace=True)

# Conversion des Colonnes Numériques
num_cols = ['Population (milliers)', 'Nombre de logements (milliers)', 'Nombre de faits']
for col in num_cols:
    df[col] = df[col].astype(float).apply(lambda x: int(x) if not np.isnan(x) else None)

# Calcul du taux pour mille
df['Taux pour mille'] = df['Nombre de faits'] / df['Population (milliers)'] * 1000

# Arrondir les valeurs du taux pour mille à trois décimales
df['Taux pour mille'] = df['Taux pour mille'].round(3)

# Formatter la colonne 'Taux pour mille' pour ne garder que le premier chiffre après la virgule
df['Taux pour mille'] = df['Taux pour mille'].apply(lambda x: f"{x:.1f}".replace('.', ','))

# Création du dossier FichierTraité s'il n'existe pas
output_dir = 'FichierTraité'
os.makedirs(output_dir, exist_ok=True)

# Sauvegarde des données préparées
prepared_data_path = os.path.join(output_dir, 'traitement_dataset_insecurite.csv')
df.to_csv(prepared_data_path, index=False, sep=';')

print(f"Données préparées sauvegardées à {prepared_data_path}")
