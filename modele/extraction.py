import pandas as pd
from sqlalchemy import create_engine

# Connexion à la base de données MySQL avec SQLAlchemy
engine = create_engine('mysql://root:root@localhost:3306/elexxion')

# Fonction pour exécuter une requête SQL et sauvegarder les résultats dans un fichier Excel
def execute_query_to_excel(query, filename):
    df = pd.read_sql(query, engine)
    df.to_excel(f"data/{filename}", index=False)

# # Requête initiale : résultats électoraux
# query_resultats = """
# SELECT 
#     c.libelle_commune,
#     p.nom,
#     e.annee,
#     SUM(CASE WHEN e.type = '1er tour' THEN r.votes ELSE 0 END) AS total_voix_1er_tour,
#     SUM(CASE WHEN e.type = '2e tour' THEN r.votes ELSE 0 END) AS total_voix_2e_tour
# FROM resultat r
# JOIN election e ON r.id_election = e.id_election
# JOIN parti p ON r.id_parti = p.id_parti
# JOIN commune c ON r.code_postal = c.code_postal
# GROUP BY c.libelle_commune, p.nom, e.annee;
# """
# execute_query_to_excel(query_resultats, "resultats_electoraux.xlsx")

# # Requête 1 : Population au chômage par genre, tranche d'âge et année
# query_chomage = """
# SELECT commune.libelle_commune,
#        population_au_chomage_par_genre_et_age.annee AS annee,
#        SUM(population_au_chomage_par_genre_et_age.nb_personnes) AS total_au_chomage,
#        SUM(CASE WHEN genre.genre = 'Homme' THEN population_au_chomage_par_genre_et_age.nb_personnes ELSE 0 END) AS hommes_au_chomage,
#        SUM(CASE WHEN genre.genre = 'Femme' THEN population_au_chomage_par_genre_et_age.nb_personnes ELSE 0 END) AS femmes_au_chomage,
#        SUM(CASE WHEN genre.genre = 'Homme' AND tranche_age.tranche_age = '15-24' THEN population_au_chomage_par_genre_et_age.nb_personnes ELSE 0 END) AS hommes_15_24_au_chomage,
#        SUM(CASE WHEN genre.genre = 'Femme' AND tranche_age.tranche_age = '15-24' THEN population_au_chomage_par_genre_et_age.nb_personnes ELSE 0 END) AS femmes_15_24_au_chomage,
#        SUM(CASE WHEN genre.genre = 'Homme' AND tranche_age.tranche_age = '25-54' THEN population_au_chomage_par_genre_et_age.nb_personnes ELSE 0 END) AS hommes_25_54_au_chomage,
#        SUM(CASE WHEN genre.genre = 'Femme' AND tranche_age.tranche_age = '25-54' THEN population_au_chomage_par_genre_et_age.nb_personnes ELSE 0 END) AS femmes_25_54_au_chomage,
#        SUM(CASE WHEN genre.genre = 'Homme' AND tranche_age.tranche_age = '55-64' THEN population_au_chomage_par_genre_et_age.nb_personnes ELSE 0 END) AS hommes_55_64_au_chomage,
#        SUM(CASE WHEN genre.genre = 'Femme' AND tranche_age.tranche_age = '55-64' THEN population_au_chomage_par_genre_et_age.nb_personnes ELSE 0 END) AS femmes_55_64_au_chomage
# FROM population_au_chomage_par_genre_et_age
# JOIN commune ON population_au_chomage_par_genre_et_age.code_postal = commune.code_postal
# JOIN genre ON population_au_chomage_par_genre_et_age.id_genre = genre.id_genre
# JOIN tranche_age ON population_au_chomage_par_genre_et_age.id_tranche_age = tranche_age.id_tranche_age
# GROUP BY commune.libelle_commune, annee;

# """
# execute_query_to_excel(query_chomage, "chomage_par_genre_et_age.xlsx")

# # Requête 2 : Population par catégorie socio-professionnelle (CSP) et année
# query_csp = """
# SELECT commune.libelle_commune,
#        population_par_CSP.annee AS annee,
#        SUM(CASE WHEN categorie_socio_pro.nom_CSP = 'CS1' THEN population_par_CSP.nb_personnes ELSE 0 END) AS CS1,
#        SUM(CASE WHEN categorie_socio_pro.nom_CSP = 'CS2' THEN population_par_CSP.nb_personnes ELSE 0 END) AS CS2,
#        SUM(CASE WHEN categorie_socio_pro.nom_CSP = 'CS3' THEN population_par_CSP.nb_personnes ELSE 0 END) AS CS3,
#        SUM(CASE WHEN categorie_socio_pro.nom_CSP = 'CS4' THEN population_par_CSP.nb_personnes ELSE 0 END) AS CS4,
#        SUM(CASE WHEN categorie_socio_pro.nom_CSP = 'CS5' THEN population_par_CSP.nb_personnes ELSE 0 END) AS CS5,
#        SUM(CASE WHEN categorie_socio_pro.nom_CSP = 'CS6' THEN population_par_CSP.nb_personnes ELSE 0 END) AS CS6
# FROM population_par_CSP
# JOIN commune ON population_par_CSP.code_postal = commune.code_postal
# JOIN categorie_socio_pro ON population_par_CSP.id_CSP = categorie_socio_pro.id_CSP
# GROUP BY commune.libelle_commune, annee;
# """
# execute_query_to_excel(query_csp, "population_par_CSP.xlsx")

# # Requête 3 : Niveau de diplôme par genre et année
# query_education = """
# SELECT 
#     c.libelle_commune,
#     e.annee AS annee,
#     SUM(CASE WHEN d.niveau = 'DIPLMIN' THEN e.nombre_personnes ELSE 0 END) AS DIPLMIN_total,
#     SUM(CASE WHEN d.niveau = 'BEPC' THEN e.nombre_personnes ELSE 0 END) AS BEPC_total,
#     SUM(CASE WHEN d.niveau = 'CAPBEP' THEN e.nombre_personnes ELSE 0 END) AS CAPBEP_total,
#     SUM(CASE WHEN d.niveau = 'BAC' THEN e.nombre_personnes ELSE 0 END) AS BAC_total,
#     SUM(CASE WHEN d.niveau = 'SUP2' THEN e.nombre_personnes ELSE 0 END) AS SUP2_total,
#     SUM(CASE WHEN d.niveau = 'SUP34' THEN e.nombre_personnes ELSE 0 END) AS SUP34_total,
#     SUM(CASE WHEN d.niveau = 'SUP5' THEN e.nombre_personnes ELSE 0 END) AS SUP5_total,
#     SUM(CASE WHEN d.niveau = 'DIPLMIN' AND g.genre = 'Homme' THEN e.nombre_personnes ELSE 0 END) AS DIPLMIN_hommes,
#     SUM(CASE WHEN d.niveau = 'BEPC' AND g.genre = 'Homme' THEN e.nombre_personnes ELSE 0 END) AS BEPC_hommes,
#     SUM(CASE WHEN d.niveau = 'CAPBEP' AND g.genre = 'Homme' THEN e.nombre_personnes ELSE 0 END) AS CAPBEP_hommes,
#     SUM(CASE WHEN d.niveau = 'BAC' AND g.genre = 'Homme' THEN e.nombre_personnes ELSE 0 END) AS BAC_hommes,
#     SUM(CASE WHEN d.niveau = 'SUP2' AND g.genre = 'Homme' THEN e.nombre_personnes ELSE 0 END) AS SUP2_hommes,
#     SUM(CASE WHEN d.niveau = 'SUP34' AND g.genre = 'Homme' THEN e.nombre_personnes ELSE 0 END) AS SUP34_hommes,
#     SUM(CASE WHEN d.niveau = 'SUP5' AND g.genre = 'Homme' THEN e.nombre_personnes ELSE 0 END) AS SUP5_hommes,
#     SUM(CASE WHEN d.niveau = 'DIPLMIN' AND g.genre = 'Femme' THEN e.nombre_personnes ELSE 0 END) AS DIPLMIN_femmes,
#     SUM(CASE WHEN d.niveau = 'BEPC' AND g.genre = 'Femme' THEN e.nombre_personnes ELSE 0 END) AS BEPC_femmes,
#     SUM(CASE WHEN d.niveau = 'CAPBEP' AND g.genre = 'Femme' THEN e.nombre_personnes ELSE 0 END) AS CAPBEP_femmes,
#     SUM(CASE WHEN d.niveau = 'BAC' AND g.genre = 'Femme' THEN e.nombre_personnes ELSE 0 END) AS BAC_femmes,
#     SUM(CASE WHEN d.niveau = 'SUP2' AND g.genre = 'Femme' THEN e.nombre_personnes ELSE 0 END) AS SUP2_femmes,
#     SUM(CASE WHEN d.niveau = 'SUP34' AND g.genre = 'Femme' THEN e.nombre_personnes ELSE 0 END) AS SUP34_femmes,
#     SUM(CASE WHEN d.niveau = 'SUP5' AND g.genre = 'Femme' THEN e.nombre_personnes ELSE 0 END) AS SUP5_femmes
# FROM education e
# JOIN commune c ON e.code_postal = c.code_postal
# JOIN diplome d ON e.id_diplome = d.id_diplome
# JOIN genre g ON e.id_genre = g.id_genre
# GROUP BY c.libelle_commune, e.annee;
# """
# execute_query_to_excel(query_education, "niveau_diplome_par_genre.xlsx")

# # Requête 4 : Sentiment d'insécurité par genre, tranche d'âge et année
# query_insecurite = """
# SELECT sentiment_insecurite.annee AS annee,
#        AVG(CASE WHEN tranche_age.tranche_age = '14-29' THEN sentiment_insecurite.taux ELSE NULL END) AS sentiment_insecurite_14_29,
#        AVG(CASE WHEN tranche_age.tranche_age = '30-44' THEN sentiment_insecurite.taux ELSE NULL END) AS sentiment_insecurite_30_44,
#        AVG(CASE WHEN tranche_age.tranche_age = '45-59' THEN sentiment_insecurite.taux ELSE NULL END) AS sentiment_insecurite_45_59,
#        AVG(CASE WHEN tranche_age.tranche_age = '60-74' THEN sentiment_insecurite.taux ELSE NULL END) AS sentiment_insecurite_60_74,
#        AVG(CASE WHEN tranche_age.tranche_age = '75-+' THEN sentiment_insecurite.taux ELSE NULL END) AS sentiment_insecurite_75_plus
# FROM sentiment_insecurite
# JOIN tranche_age ON sentiment_insecurite.id_tranche_age = tranche_age.id_tranche_age
# GROUP BY annee;
# """
# execute_query_to_excel(query_insecurite, "sentiment_insecurite.xlsx")

# # Requête 5 : Confiance des ménages (indicateur synthétique) et année
# query_confiance = """
# SELECT pays.libelle_pays,
#        indicateur_confiance_menage.annee AS annee,
#        indicateur_confiance_menage.indicateur_synthetique,
#        indicateur_confiance_menage.indicateur_niveau_de_vie_passe,
#        indicateur_confiance_menage.indicateur_niveau_de_vie_evolution,
#        indicateur_confiance_menage.indicateur_chomage_evolution
# FROM indicateur_confiance_menage
# JOIN pays ON indicateur_confiance_menage.id_pays = pays.id_pays
# """
# execute_query_to_excel(query_confiance, "confiance_des_menages.xlsx")

# Requête 6 : Nombre d'habitants et part d'immigration par ville
query_population = """
SELECT nombre_habitants, (population.nombre_habitants * population.part_immigration) / 100 AS nombre_immigration, annee, libelle_commune
FROM  population 
JOIN commune ON commune.code_postal = population.code_postal
"""
execute_query_to_excel(query_population, "population_et_immigration.xlsx")

# Requête 6 : Population par âge et genre
query_age = """
SELECT nombre_habitants, genre,  tranche_age, annee, libelle_commune
FROM  population_par_age_et_genre 
JOIN commune ON population_par_age_et_genre.code_postal = population.code_postal
"""
execute_query_to_excel(query_population, "population_et_immigration.xlsx")

# Requête 6 : population par âge et genre
query_population_genre_age = """
SELECT 
    commune.libelle_commune,
    population_par_age_et_genre.annee AS annee,
    
    -- Groupe hommes et femmes
    SUM(CASE WHEN genre.genre = 'Homme' THEN population_par_age_et_genre.nb_personnes ELSE 0 END) AS total_hommes,
    SUM(CASE WHEN genre.genre = 'Femme' THEN population_par_age_et_genre.nb_personnes ELSE 0 END) AS total_femmes,
    
    -- Groupe par génération (tous genres confondus)
    SUM(CASE WHEN tranche_age.tranche_age = '15-24' THEN population_par_age_et_genre.nb_personnes ELSE 0 END) AS generation_15_24,
    SUM(CASE WHEN tranche_age.tranche_age = '25-54' THEN population_par_age_et_genre.nb_personnes ELSE 0 END) AS generation_25_54,
    SUM(CASE WHEN tranche_age.tranche_age = '55-64' THEN population_par_age_et_genre.nb_personnes ELSE 0 END) AS generation_55_64

FROM population_par_age_et_genre
JOIN commune ON population_par_age_et_genre.code_postal = commune.code_postal
JOIN genre ON population_par_age_et_genre.id_genre = genre.id_genre
JOIN tranche_age ON population_par_age_et_genre.id_tranche_age = tranche_age.id_tranche_age
GROUP BY commune.libelle_commune, annee;

"""
execute_query_to_excel(query_population_genre_age, "population_par_age_genre.xlsx")


# Requête 7 : CSP majoritaire
query_csp_max = """
SELECT
    c.libelle_commune,
    p.annee,
    cs.nom_CSP AS CSP_majoritaire
FROM
    population_par_CSP p
JOIN
    (SELECT
         code_postal,
         annee,
         id_CSP,
         COUNT(*) AS count
     FROM
         population_par_CSP
     GROUP BY
         code_postal,
         annee,
         id_CSP) pc
ON
    p.code_postal = pc.code_postal AND p.annee = pc.annee AND p.id_CSP = pc.id_CSP
JOIN
    (SELECT
         code_postal,
         annee,
         MAX(count) AS max_count
     FROM
         (SELECT
              code_postal,
              annee,
              id_CSP,
              COUNT(*) AS count
          FROM
              population_par_CSP
          GROUP BY
              code_postal,
              annee,
              id_CSP) AS temp
     GROUP BY
         code_postal,
         annee) mpc
ON
    pc.code_postal = mpc.code_postal AND pc.annee = mpc.annee AND pc.count = mpc.max_count
JOIN
    categorie_socio_pro cs
ON
    pc.id_CSP = cs.id_CSP
JOIN
    commune c
ON
    p.code_postal = c.code_postal;
"""
execute_query_to_excel(query_csp_max, "csp_max.xlsx")

# Requête 8 : diplôme majoritaire
query_dipl_max = """
SELECT
    e.code_postal,
    e.annee,
       d.niveau AS diplome_majoritaire
FROM
    education e
JOIN
    (SELECT
         code_postal,
         annee,
         id_diplome,
         COUNT(*) AS count
     FROM
         education
     GROUP BY
         code_postal,
         annee,
         id_diplome) dc
ON
    e.code_postal = dc.code_postal AND e.annee = dc.annee AND e.id_diplome = dc.id_diplome
JOIN
    (SELECT
         code_postal,
         annee,
         MAX(count) AS max_count
     FROM
         (SELECT
              code_postal,
              annee,
              id_diplome,
              COUNT(*) AS count
          FROM
              education
          GROUP BY
              code_postal,
              annee,
              id_diplome) AS temp
     GROUP BY
         code_postal,
         annee) mdc
ON
    dc.code_postal = mdc.code_postal AND dc.annee = mdc.annee AND dc.count = mdc.max_count
JOIN
    diplome d
ON
    dc.id_diplome = d.id_diplome
JOIN
    commune c
ON
    e.code_postal = c.code_postal;
"""
execute_query_to_excel(query_dipl_max, "dipl_max.xlsx")

#Requête 9 : ville par taille 
query_taille_ville = """
SELECT 
    code_postal,
    nombre_habitants,
    CASE 
        WHEN nombre_habitants < 200 THEN 'Hameau'
        WHEN nombre_habitants >= 200 AND nombre_habitants < 2000 THEN 'Village'
        WHEN nombre_habitants >= 2000 AND nombre_habitants < 5000 THEN 'Bourg'
        WHEN nombre_habitants >= 5000 AND nombre_habitants < 100000 THEN 'Petite/Moyenne Ville'
        WHEN nombre_habitants >= 100000 THEN 'Grande Ville'
    END AS categorie_ville
FROM 
    population
ORDER BY 
    nombre_habitants DESC;
"""
execute_query_to_excel(query_taille_ville, "taille_ville.xlsx")

# Fermeture de la connexion
engine.dispose()
