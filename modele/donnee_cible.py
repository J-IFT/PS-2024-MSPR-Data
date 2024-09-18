import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Charger les données
df = pd.read_excel("data/resultats_2022_1er_tour_eure_et_loir.xls")

annee = 2022
# Créer un nouveau DataFrame pour stocker les résultats
resultats = pd.DataFrame(columns=['libelle_commune', 'annee', 'premier_parti', 'premier_parti_encoded', 'premiere_valeur', 'deuxieme_parti', 'deuxieme_parti_encoded', 'deuxieme_valeur'])

# Instancier un LabelEncoder pour encoder les noms de partis
le = LabelEncoder()

# Initialiser une liste pour stocker tous les partis (pour l'encodage global)
partis_a_encoder = []

# Parcourir chaque ligne du DataFrame
for index, row in df.iterrows():
    # Séparer la colonne "libelle_commune" des autres colonnes (les partis politiques)
    commune = row['libelle_commune']
    
    # Supprimer "libelle_commune" pour ne garder que les colonnes des partis
    valeurs = row.drop('libelle_commune')
    
    # Trier les colonnes en fonction des valeurs (ordre décroissant) pour trouver les deux premiers partis
    sorted_partis = valeurs.sort_values(ascending=False)
    
    # Obtenir les deux premiers partis et leurs valeurs
    premier_parti = sorted_partis.index[0]
    premier_valeur = sorted_partis.iloc[0]
    deuxieme_parti = sorted_partis.index[1]
    deuxieme_valeur = sorted_partis.iloc[1]
    
    # Ajouter les partis à la liste pour encodage ultérieur
    partis_a_encoder.extend([premier_parti, deuxieme_parti])
    
    # Créer un DataFrame temporaire avec les résultats de la ligne courante (sans l'encodage pour le moment)
    temp_df = pd.DataFrame({
        'libelle_commune': [commune],
        'annee': [annee],
        'premier_parti': [premier_parti],
        'premiere_valeur': [premier_valeur],
        'deuxieme_parti': [deuxieme_parti],
        'deuxieme_valeur': [deuxieme_valeur]
    })
    
    # Ajouter la ligne au DataFrame resultats
    resultats = pd.concat([resultats, temp_df], ignore_index=True)

# Encoder les partis après avoir parcouru toutes les lignes (encodage global pour cohérence)
le.fit(partis_a_encoder)
resultats['premier_parti_encoded'] = le.transform(resultats['premier_parti'])
resultats['deuxieme_parti_encoded'] = le.transform(resultats['deuxieme_parti'])

# Sauvegarder les résultats dans un fichier Excel
resultats.to_excel("data/resultats_premier_tour.xlsx", index=False)
