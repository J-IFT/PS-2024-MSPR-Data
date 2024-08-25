-- Table des Partis
CREATE TABLE parti (
    id_parti INT AUTO_INCREMENT PRIMARY KEY,
    nom VARCHAR(255) NOT NULL
);

-- Table des Elections
CREATE TABLE election (
    id_election INT AUTO_INCREMENT PRIMARY KEY,
    annee INT NOT NULL,
    type ENUM('1er tour', '2e tour') NOT NULL
);

CREATE TABLE departement (
    id_departement INT PRIMARY KEY AUTO_INCREMENT,
    libelle_departement VARCHAR(250)
);

CREATE TABLE pays (
    id_pays INT PRIMARY KEY AUTO_INCREMENT,
    libelle_pays VARCHAR(250)
);


-- Table des Communes
CREATE TABLE commune (
    code_postal INT PRIMARY KEY,
    libelle_commune VARCHAR(255) NOT NULL,
    id_departement INT,
    id_pays INT,
    FOREIGN KEY (id_departement) REFERENCES departement(id_departement),
    FOREIGN KEY (id_pays) REFERENCES pays(id_pays)
);

-- Table des Résultats
CREATE TABLE resultat (
    id_resultat INT AUTO_INCREMENT PRIMARY KEY,
    id_parti INT,
    id_election INT,
    code_postal INT,
    votes INT,
    FOREIGN KEY (id_parti) REFERENCES parti(id_parti),
    FOREIGN KEY (id_election) REFERENCES election(id_election),
    FOREIGN KEY (code_postal) REFERENCES commune(code_postal)
);

-- Table des Elections futures
CREATE TABLE election_future (
    id_election_future INT AUTO_INCREMENT PRIMARY KEY,
    annee INT NOT NULL,
    type ENUM('1er tour', '2e tour') NOT NULL
);

-- Table des Prédictions
CREATE TABLE prediction (
    id_prediction INT AUTO_INCREMENT PRIMARY KEY,
    id_parti INT,
    id_election INT,
    code_postal INT,
    votes_prevus INT,
    taux_precision FLOAT,
    FOREIGN KEY (id_parti) REFERENCES parti(id_parti),
    FOREIGN KEY (id_election) REFERENCES election(id_election),
    FOREIGN KEY (code_postal) REFERENCES commune(code_postal)
);

-- Table des données démographiques générales
CREATE TABLE population(
    id_population INT AUTO_INCREMENT PRIMARY KEY,
    code_postal INT,
    annee INT,
    nombre_habitants INT,
    part_immigration FLOAT,
    FOREIGN KEY (code_postal) REFERENCES commune(code_postal)
);

-- Table des Diplômes
CREATE TABLE diplome (
    id_diplome INT AUTO_INCREMENT PRIMARY KEY,
    niveau VARCHAR(255)
);

-- Table des niveaux d'éducation
CREATE TABLE education (
    id_education INT AUTO_INCREMENT PRIMARY KEY,
    annee INT,
    code_postal INT,
    id_diplome INT,
    nombre_total INT,
    nombre_hommes INT,
    nombre_femmes INT,
    FOREIGN KEY (id_diplome) REFERENCES diplome(id_diplome),
    FOREIGN KEY (code_postal) REFERENCES commune(code_postal)
);

-- Table des Âges
CREATE TABLE tranche_age (
    id_tranche_age INT PRIMARY KEY AUTO_INCREMENT,
    tranche_age VARCHAR(50) -- Ex : '18-25', '26-35', etc.
);

-- Table des Genres
CREATE TABLE genre (
    id_genre INT AUTO_INCREMENT PRIMARY KEY,
    genre VARCHAR(255)
);

CREATE TABLE categorie_socio_pro (
    id_CSP INT PRIMARY KEY AUTO_INCREMENT,
    nom_CSP VARCHAR(255) -- Ex : 'Agriculteurs', 'Cadres', etc.
);

CREATE TABLE population_par_âge (
    id_population_age INT PRIMARY KEY AUTO_INCREMENT,
    code_postal INT, -- Clé étrangère vers la table Commune
    annee INT, -- Année des données
    id_tranche_age INT, -- Clé étrangère vers la table TrancheAge
    nb_personnes INT, -- Nombre total de personnes dans cette tranche d'âge
    FOREIGN KEY (code_postal) REFERENCES commune(code_postal),
    FOREIGN KEY (id_tranche_age) REFERENCES tranche_age(id_tranche_age)
);

CREATE TABLE population_par_CSP (
    id_population_csp INT PRIMARY KEY AUTO_INCREMENT,
    code_postal INT, -- Clé étrangère vers la table Commune
    annee INT, -- Année des données
    id_CSP INT, -- Clé étrangère vers la table CategorieSocioPro
    nb_personnes INT, -- Nombre total de personnes dans cette catégorie
    FOREIGN KEY (code_postal) REFERENCES commune(code_postal),
    FOREIGN KEY (id_CSP) REFERENCES categorie_socio_pro(id_CSP)
);

-- Table de la Délinquance et Criminalité
CREATE TABLE delinquance_criminalite (
    id_delinquance_criminalite INT AUTO_INCREMENT PRIMARY KEY,
    id_departement INT,
    annee INT,
    taux_homicides FLOAT,
    taux_coups_blessures_volontaires_intrafamiliaux FLOAT,
    taux_autres_coups_blessures FLOAT,
    taux_violences_sexuelles FLOAT,
    taux_vols_avec_arme FLOAT,
    taux_vols_violents_sans_arme FLOAT,
    taux_sans_violence_contre_personnes FLOAT,
    taux_de_vehicules FLOAT,
    taux_vols_dans_vehicules FLOAT,
    taux_vols_accessoires_sur_vehicule FLOAT,
    taux_destructions_degradations_volontaires FLOAT,
    taux_escroqueries FLOAT,
    taux_trafic_stupefiants FLOAT,
    taux_usage_stupefiants FLOAT,
    FOREIGN KEY (id_departement) REFERENCES departement(id_departement)
);

-- Table des Sentiments d'Insécurité et Indice de Confiance des Ménages
CREATE TABLE sentiment_insecurite (
    id_sentiment_insecurite INT AUTO_INCREMENT PRIMARY KEY,
    id_pays INT,
    annee INT,
    id_genre INT,
    id_tranche_age INT,
    taux FLOAT,
    FOREIGN KEY (id_pays) REFERENCES pays(id_pays),
    FOREIGN KEY (id_tranche_age) REFERENCES tranche_age(id_tranche_age),
    FOREIGN KEY (id_genre) REFERENCES genre(id_genre)
);

-- Table de l'Indice de Confiance des Ménages
CREATE TABLE indicateur_confiance_menage (
    id_indicateur_confiance_menage INT AUTO_INCREMENT PRIMARY KEY,
    id_pays INT,
    annee INT,
    indicateur_niveau_de_vie_passe FLOAT,
    indicateur_niveau_de_vie_evolution FLOAT,
    indicateur_chomage_evolution FLOAT,
    indicateur_prix_passe FLOAT,
    indicateur_prix_evolution FLOAT,
    indicateur_opportunite_achats_importants FLOAT,
    indicateur_opportunite_epargne FLOAT,
    indicateur_capacite_epargne_actuelle FLOAT,
    indicateur_capacite_epargne_evolution FLOAT,
    indicateur_situation_financiere_personelle_passee FLOAT,
    indicateur_situation_financiere_personelle_evolution FLOAT,
    FOREIGN KEY (id_pays) REFERENCES pays(id_pays)
);