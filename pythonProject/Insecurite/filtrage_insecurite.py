import os
import pandas as pd

# Ce script a pour but de filtrer notre jeu de données avant de faire notre dictionnaire de données

# Charger le jeu de données
df = pd.read_csv('C:/Users/julie/OneDrive - Ifag Paris/Documents/MSPR/3/donnee-data.gouv-2023-geographie2023-produit-le2024-03-07.csv', delimiter=';')

# Filtrer les lignes où 'CODGEO_2023' commence par '28' et contient exactement 5 chiffres
filtered_df = df[df['CODGEO_2023'].astype(str).str.startswith('28') & (df['CODGEO_2023'].astype(str).str.len() == 5)]

# Filtrer les lignes pour les années 2017 et 2022
filtered_df = filtered_df[filtered_df['annee'].isin([17, 22])]

# Renommer les colonnes pour plus de clarté
filtered_df.rename(columns={
    'CODGEO_2023': 'Code_Commune',
    'annee': 'Annee',
    'classe': 'Categorie_Infraction',
    'unité.de.compte': 'Unite_Compte',
    'valeur.publiée': 'Valeur_Publiee',
    'faits': 'Nombre_Incidents',
    'tauxpourmille': 'Taux_Pour_Mille',
    'POP': 'Population',
    'LOG': 'Nombre_Logements'
}, inplace=True)

# Sélectionner les colonnes pertinentes
columns_to_keep = ['Code_Commune', 'Annee', 'Categorie_Infraction', 'Unite_Compte', 'Valeur_Publiee',
                   'Nombre_Incidents', 'Taux_Pour_Mille', 'Population', 'Nombre_Logements']
filtered_df = filtered_df[columns_to_keep]

# Créer le dossier de destination s'il n'existe pas
output_dir = 'FichierFiltré'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Sauvegarder le jeu de données filtré
output_path = os.path.join(output_dir, 'filtre_dataset_insecurite.csv')
filtered_df.to_csv(output_path, index=False, sep=';')

print(f"Filtered dataset saved to {output_path}")
