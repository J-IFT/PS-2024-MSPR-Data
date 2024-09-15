import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
import seaborn as sns
from sklearn.feature_selection import SelectFromModel

donnees_cibles = [
    'Lutte_ouvrière', 
    'Parti_communiste_français', 
    'Renaissance', 
    'Résistons', 
    'Rassemblement_national', 
    'Reconquête', 
    'La_France_Insoumise', 
    'Le_partie_socialiste', 
    'Europe_écologie_Les_verts', 
    'Les_républicains', 
    'Le_nouveau_parti_anticapitaliste', 
    'Debout_la_France'
]

# Préparer les données
def prepare_data(parti):
    data = pd.read_excel("data/fichier_final.xlsx", decimal=',')
    
    feature_columns = [
        parti, # Variable cible
        'total_au_chomage', 
        # 'CSP_majoritaire_encoded',
        'Agriculteurs exploitants', 'Artisans_commerçants_chefs_d_entreprise', 
        'Cadre_et_profession_intellectuelle_supérieure', 'Profession_intermédiaire', 'Employe', 'Ouvrier',
        'DIPLMIN_total', 'CAPBEP_total', 'BAC_total', 'SUP34_total', 'SUP5_total', 'nombre_habitants', 'nombre_immigration',
        'Médiane_du_niveau_de_vie', 'generation_55_64', 'Salaire_net_horaire_moyen', 'pourcentage_abstention'
        # 'sentiment_insecurite_14_29', 'sentiment_insecurite_30_44',
        # 'sentiment_insecurite_45_59', 'sentiment_insecurite_60_74', 'indicateur_synthetique',
        # 'indicateur_niveau_de_vie_passe', 'indicateur_niveau_de_vie_evolution',
        # 'indicateur_chomage_evolution', 'sentiment_insecurite_total'
    ]

    # Sélectionner les colonnes utiles dans les données
    data = data[feature_columns].apply(pd.to_numeric, errors='coerce')
    
    return data

for index, parti in enumerate(donnees_cibles):
    data = prepare_data(parti)

    # ANALYSE DE LA CORRELATION POUR CHOISIR LES VARIABLES EXPLICATIVES
    correlation_matrix = data.corr()

    # Sélectionner la corrélation avec la variable cible
    correlation_with_target = correlation_matrix[parti]

    # Trier les corrélations par ordre décroissant
    correlation_sorted = correlation_with_target.sort_values(ascending=False)

    #----------------------------------------
    # Afficher les indices de corrélation
    print("Corrélation des variables explicatives avec les résultats du premier tour:")
    print(correlation_sorted)

    #---------------------------------------
    # Création d'une heatmap
    # plt.figure(figsize=(6, 4))
    # correlation_sorted.plot(kind='barh')
    # plt.title('Corrélation des variables explicatives avec le parti au premier tour :')
    # plt.xlabel('Corrélation')
    # plt.ylabel('Variables')
    # plt.subplots_adjust(left=0.4)
    # plt.subplots_adjust(bottom=0.3)
    # plt.show()

    #--------------------------------------
    # Visualiser la matrice de corrélation avec une carte de chaleur
    # mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    # plt.title('Carte de Chaleur des Corrélations entre Variables Explicatives')
    # plt.subplots_adjust(left=0.2)
    # plt.subplots_adjust(bottom=0.4)
    # plt.show()


    #---------------------------------------
    # Séparation des données en caractéristiques (X) et cible (y)
    X = data.drop(columns=[parti])
    y = data[parti]

    # Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    #----------------------------------------------------------------------------------------------
    # Entraînement du modèle Ridge
    # ridge_model = Ridge(alpha=1.0)  
    # ridge_model.fit(X_train, y_train)

    # # Prédictions avec le modèle Ridge
    # y_pred_ridge = ridge_model.predict(X_test)

    # # Évaluation du modèle Ridge
    # mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    # r2_ridge = r2_score(y_test, y_pred_ridge)

    # print(f"Ridge Regression - Mean Squared Error: {mse_ridge}")
    # print(f"Ridge Regression - R² Score: {r2_ridge}")

    # #----------------------------------------------------------------------------------------------
    # Entraînement du modèle Lasso
    lasso_model = Lasso(alpha=1)
    lasso_model.fit(X_train, y_train)

    # Prédictions avec le modèle Lasso
    y_pred_lasso = lasso_model.predict(X_test)

    # Évaluation du modèle Lasso
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)

    print(f"Lasso Regression - Mean Squared Error: {mse_lasso}")
    print(f"Lasso Regression - R² Score: {r2_lasso}")

    #---------------------------------------------------------
    # Examiner les coefficients du modèle Lasso
    coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': lasso_model.coef_})
    # Sélectionner les variables avec des coefficients non nuls et supérieur à différents seuils
    # thresholds = [0.01, 0.05, 0.1, 0.2]  # Liste des seuils à tester

    # for threshold in thresholds:
    #     selected_features = coefficients[abs(coefficients['Coefficient']) > threshold]['Variable'].tolist()
    #     print(f"Variables sélectionnées avec un seuil de {threshold}:")
    #     print(selected_features)
    #     print()

    # plt.figure(figsize=(10, 6))
    # plt.hist(abs(coefficients['Coefficient']), bins=30, edgecolor='k', alpha=0.7)
    # plt.xlabel('Valeur Absolue des Coefficients')
    # plt.ylabel('Fréquence')
    # plt.title('Distribution des Coefficients du Modèle Lasso')
    # plt.show()

    #----------
    # Sélectionner les variables avec des coefficients non nuls et supérieur au seuil de 0.1
    # threshold = 0.1
    # selected_features = coefficients[abs(coefficients['Coefficient']) > threshold]['Variable'].tolist()

    #Sans seuil
    selected_features = coefficients[coefficients['Coefficient'] != 0]

     # Trier les variables par l'importance des coefficients absolus
    selected_features['abs_Coefficient'] = selected_features['Coefficient'].abs()
    top_features = selected_features.sort_values(by='abs_Coefficient', ascending=False).head(5)

    print("Top 5 variables sélectionnées par le modèle Lasso :")
    print(top_features[['Variable', 'Coefficient']])

    #---------------------------------------------------------
    plt.subplot(1, 2, 2)
    plt.bar(top_features['Variable'], top_features['Coefficient'])
    plt.title('Coefficients du modèle Lasso pour le parti ' + parti)
    plt.xlabel('Variables')
    plt.ylabel('Coefficients')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()

    #---------------------------------------------------------
    # Visualisation des coefficients des modèles
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.bar(X.columns, ridge_model.coef_)
    # plt.title('Coefficients du modèle Ridge')
    # plt.xlabel('Variables')
    # plt.ylabel('Coefficients')
    # plt.xticks(rotation=90)

