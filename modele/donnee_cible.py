import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Charger les données
df = pd.read_excel("data/resultats_2022_1er_tour_eure_et_loir.xls")

annee = 2022
# Créer un nouveau DataFrame pour stocker les résultats
resultats = pd.DataFrame(columns=['libelle_commune', 'annee', 'premier_parti', 'premiere_valeur', 'deuxieme_parti', 'deuxieme_valeur', 'resultat_premier_tour'])

# Définir une fonction pour convertir les résultats en valeurs numériques
def encode_resultat_premier_tour(df):
    le = LabelEncoder()
    df['resultat_premier_tour_encoded'] = le.fit_transform(df['resultat_premier_tour'])
    return df

# Parcourir chaque ligne du DataFrame
for index, row in df.iterrows():
    # Séparer la colonne "Libellé de la commune" des autres colonnes (les partis politiques)
    commune = row['libelle_commune']
    valeurs = row.drop('libelle_commune')
    
    # Trier les colonnes en fonction des valeurs (ordre décroissant)
    sorted_partis = valeurs.sort_values(ascending=False)
    
    # Obtenir les deux premiers partis avec les valeurs les plus élevées
    premier_parti = sorted_partis.index[0]
    premier_valeur = sorted_partis.iloc[0]
    deuxieme_parti = sorted_partis.index[1]
    deuxieme_valeur = sorted_partis.iloc[1]
    
    # Trier les noms des partis par ordre alphabétique pour garantir un ordre constant
    partis_ordonnes = sorted([premier_parti, deuxieme_parti])
    
    # Créer la combinaison des partis sous la forme "Premier_Parti - Deuxieme_Parti"
    combinaison_partis = f"{partis_ordonnes[0]} - {partis_ordonnes[1]}"
    
    # Créer un DataFrame temporaire avec les résultats de la ligne courante
    temp_df = pd.DataFrame({
        'libelle_commune': [commune],
        'annee': [annee],
        'premier_parti': [premier_parti],
        'premiere_valeur': [premier_valeur],
        'deuxieme_parti': [deuxieme_parti],
        'deuxieme_valeur': [deuxieme_valeur],
        'resultat_premier_tour': [combinaison_partis]
    })
    
    # Utiliser pd.concat() pour ajouter la ligne au DataFrame resultats
    resultats = pd.concat([resultats, temp_df], ignore_index=True)

# Encoder la colonne 'resultat_premier_tour'
resultats = encode_resultat_premier_tour(resultats)

# Sauvegarder les résultats dans un fichier Excel
resultats.to_excel("data/resultats_premier_tour.xlsx", index=False)
