
import pandas as pd
# Chargement des données
# Variable cible
resultats = pd.read_excel("data/resultats_premier_tour.xlsx")

# Variables explicatives
chomage = pd.read_excel("data/chomage_par_genre_et_age.xlsx")
csp = pd.read_excel("data/population_par_CSP.xlsx")
education = pd.read_excel("data/niveau_diplome_par_genre.xlsx")
insecurite = pd.read_excel("data/sentiment_insecurite.xlsx")
confiance = pd.read_excel("data/confiance_des_menages.xlsx")


data = pd.merge(resultats, chomage, on=['libelle_commune', 'annee'], how='left')
data = pd.merge(data, csp, on=['libelle_commune', 'annee'], how='left')
data = pd.merge(data, education, on=['libelle_commune', 'annee'], how='left')
data = pd.merge(data, insecurite, on='annee', how='left')
data = pd.merge(data, confiance, on='annee', how='left')

data.to_excel("data/fichier_final.xlsx", index=False)