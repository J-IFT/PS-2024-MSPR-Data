import pandas as pd
import matplotlib.pyplot as plt

# Charger les données à partir du fichier préparé
file_path = 'FichierTraité/traitement_dataset_insecurite.csv'
df = pd.read_csv(file_path, delimiter=';')

# Préparation des données pour le graphique à barres
crime_years = df.groupby(['Type de crime', 'Annee'])['Nombre de faits'].sum().unstack()

# Création de la figure
plt.figure(figsize=(12, 8))

# Création du graphique à barres
bar_width = 0.35
index = range(len(crime_years.index))
bar1 = plt.bar(index, crime_years[2017], width=bar_width, label='2017')
bar2 = plt.bar([i + bar_width for i in index], crime_years[2022], width=bar_width, label='2022')

# Réglage des étiquettes et des titres
plt.xlabel('Type de crime')
plt.ylabel('Nombre de faits')
plt.title('Nombre de faits par type de crime et par année')

# Ajustement des étiquettes de l'axe x en rotation verticale
plt.xticks([i + bar_width / 2 for i in index], crime_years.index, rotation='vertical')

plt.legend(title='Année')
plt.tight_layout()

# Affichage du graphique
plt.show()
