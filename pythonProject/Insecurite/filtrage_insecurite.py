import os
import pandas as pd

# Chemin vers le fichier CSV initial
file_path = 'C:/Users/julie/OneDrive - Ifag Paris/Documents/MSPR/3/donnee-dep-data.gouv-2023-geographie2023-produit-le2024-03-07.csv'

# Charger le fichier CSV en utilisant pandas
df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')

# Filtrer les lignes avec le code département '28'
filtered_df = df[df['Code.département'] == '28']

# Filtrer les lignes pour les années 2017 et 2022
filtered_df = filtered_df[filtered_df['annee'].isin([17, 22])]

# Renommer les colonnes pour une meilleure compréhension
filtered_df = filtered_df.rename(columns={
    'classe': 'Type de crime',
    'annee': 'Annee',
    'unité.de.compte': 'Unite',
    'faits': 'Nombre de faits',
    'millPOP': 'Population (milliers)',
    'millLOG': 'Nombre de logements (milliers)',
    'tauxpourmille': 'Taux pour mille'
})

# Colonnes à conserver
columns_to_keep = ['Type de crime', 'Annee', 'Unite', 'Nombre de faits', 'Population (milliers)', 'Nombre de logements (milliers)', 'Taux pour mille']

# Sélectionner seulement les colonnes à conserver
filtered_df = filtered_df[columns_to_keep]

# Créer le dossier de destination s'il n'existe pas
output_dir = 'FichierFiltré'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Sauvegarder le jeu de données filtré
output_path = os.path.join(output_dir, 'filtre_dataset_insecurite.csv')
filtered_df.to_csv(output_path, index=False, sep=';')

print(f"Filtered and sorted dataset saved to {output_path}")
