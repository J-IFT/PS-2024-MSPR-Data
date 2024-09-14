import pandas as pd
import numpy as np
import os

# Définir les colonnes
colonnes = [
    'total_au_chomage', 'hommes_au_chomage', 'femmes_au_chomage',
    'hommes_15_24_au_chomage', 'femmes_15_24_au_chomage',
    'hommes_25_54_au_chomage', 'femmes_25_54_au_chomage',
    'hommes_55_64_au_chomage', 'femmes_55_64_au_chomage',
    'CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6',
    'DIPLMIN_total', 'BEPC_total', 'CAPBEP_total', 'BAC_total',
    'SUP2_total', 'SUP34_total', 'SUP5_total',
    'DIPLMIN_hommes', 'BEPC_hommes', 'CAPBEP_hommes', 'BAC_hommes',
    'SUP2_hommes', 'SUP34_hommes', 'SUP5_hommes',
    'DIPLMIN_femmes', 'BEPC_femmes', 'CAPBEP_femmes', 'BAC_femmes',
    'SUP2_femmes', 'SUP34_femmes', 'SUP5_femmes',
    'sentiment_insecurite_14_29', 'sentiment_insecurite_30_44',
    'sentiment_insecurite_45_59', 'sentiment_insecurite_60_74',
    'sentiment_insecurite_75_plus', 'indicateur_synthetique',
    'indicateur_niveau_de_vie_passe', 'indicateur_niveau_de_vie_evolution',
    'indicateur_chomage_evolution'
]

# Nombre de lignes à générer
n_lignes = 10  
np.random.seed(42)
data = pd.DataFrame(np.random.randn(n_lignes, len(colonnes)), columns=colonnes)

# Ajouter des valeurs positives et des ajustements pour certaines colonnes
data['total_au_chomage'] = np.random.randint(5000, 20000, size=n_lignes)
data['hommes_au_chomage'] = data['total_au_chomage'] * np.random.uniform(0.4, 0.6, size=n_lignes)
data['femmes_au_chomage'] = data['total_au_chomage'] - data['hommes_au_chomage']
data['hommes_15_24_au_chomage'] = np.random.randint(100, 500, size=n_lignes)
data['femmes_15_24_au_chomage'] = np.random.randint(100, 500, size=n_lignes)
data['hommes_25_54_au_chomage'] = np.random.randint(200, 1000, size=n_lignes)
data['femmes_25_54_au_chomage'] = np.random.randint(200, 1000, size=n_lignes)
data['hommes_55_64_au_chomage'] = np.random.randint(100, 500, size=n_lignes)
data['femmes_55_64_au_chomage'] = np.random.randint(100, 500, size=n_lignes)

# Simuler des valeurs pour les autres colonnes (en général)
data = data.abs() 


output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
file_path = os.path.join(output_dir, "donnees-fictives-pour-analyse.xlsx")
data.to_excel(file_path, index=False)

print(f"Fichier créé : {file_path}")
