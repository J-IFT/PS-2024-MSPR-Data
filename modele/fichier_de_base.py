
import pandas as pd
# Chargement des données
# Variable cible
resultats = pd.read_excel("data/resultats_2022_1er_tour.xls")

# Variables explicatives
chomage = pd.read_excel("data/chomage_par_genre_et_age.xlsx")
csp = pd.read_excel("data/population_par_CSP.xlsx")
education = pd.read_excel("data/niveau_diplome_par_genre.xlsx")
insecurite = pd.read_excel("data/sentiment_insecurite.xlsx")
confiance = pd.read_excel("data/confiance_des_menages.xlsx")
immigration = pd.read_excel("data/population_et_immigration.xlsx")
niveau_de_vie = pd.read_excel("data/niveau_de_vie.xlsx")
age_et_genre = pd.read_excel("data/population_par_age_et_genre.xlsx")
salaire_moyen = pd.read_excel("data/salaire_horaire_moyen.xlsx")
taux_abstention = pd.read_excel("data/taux_de_participation.xlsx")

data = pd.merge(resultats, chomage, on=['libelle_commune', 'annee'], how='left')
data = pd.merge(data, csp, on=['libelle_commune', 'annee'], how='left')
data = pd.merge(data, education, on=['libelle_commune', 'annee'], how='left')
data = pd.merge(data, immigration, on=['libelle_commune', 'annee'], how='left')
data = pd.merge(data, niveau_de_vie, on=['libelle_commune', 'annee'], how='left')
data = pd.merge(data, age_et_genre, on=['libelle_commune', 'annee'], how='left')
data = pd.merge(data, salaire_moyen, on=['libelle_commune', 'annee'], how='left')
data = pd.merge(data, taux_abstention, on=['libelle_commune', 'annee'], how='left')
# data = pd.merge(data, insecurite, on='annee', how='left')
# data = pd.merge(data, confiance, on='annee', how='left')cd mod    

# voir combien de valeurs manquantes il y a dans chaque colonne
# print(data.isna().sum())

# print(data.dtypes)

# Remplacer les valeurs manquantes par la médiane des colonnes numériques
data.fillna(data.median(numeric_only=True), inplace=True)

data.to_excel("data/fichier_final.xlsx", index=False)