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
from sklearn.feature_selection import RFE
import seaborn as sns

# Chargement des données
# Variable cible
resultats = pd.read_excel("data/resultats_electoraux.xlsx")
# Variables explicatives
chomage = pd.read_excel("data/chomage_par_genre_et_age.xlsx")
csp = pd.read_excel("data/population_par_CSP.xlsx")
education = pd.read_excel("data/niveau_diplome_par_genre.xlsx")
insecurite = pd.read_excel("data/sentiment_insecurite.xlsx")
confiance = pd.read_excel("data/confiance_des_menages.xlsx")

# Préparer les données
def prepare_data():
    data = pd.merge(resultats, chomage, on=['libelle_commune', 'annee'], how='left')
    data = pd.merge(data, csp, on=['libelle_commune', 'annee'], how='left')
    data = pd.merge(data, education, on=['libelle_commune', 'annee'], how='left')
    data = pd.merge(data, insecurite, on='annee', how='left')
    data = pd.merge(data, confiance, on='annee', how='left')

    feature_columns = [
        'total_voix_1er_tour', 'total_voix_2e_tour',
        'total_au_chomage', 'hommes_au_chomage', 'femmes_au_chomage',
        'Agriculteurs exploitants', 'Artisans_commerçants_chefs_d_entreprise', 'Cadre_et_profession_intellectuelle_supérieure', 'Profession_intermédiaire', 'Employe', 'Ouvrier',
        'DIPLMIN_total', 'BEPC_total', 'CAPBEP_total', 'BAC_total',
        'SUP2_total', 'SUP34_total', 'SUP5_total',
        'sentiment_insecurite_14_29', 'sentiment_insecurite_30_44',
        'sentiment_insecurite_45_59', 'sentiment_insecurite_60_74',
        'sentiment_insecurite_75_plus', 'indicateur_synthetique',
        'indicateur_niveau_de_vie_passe', 'indicateur_niveau_de_vie_evolution',
        'indicateur_chomage_evolution', 'sentiment_insecurite_total'
    ]

    # Supprimer les colonnes avec uniquement des valeurs manquantes
    missing_cols = [col for col in feature_columns if data[col].isnull().all()]
    feature_columns = [col for col in feature_columns if col not in missing_cols]
    data = data[feature_columns]

    # Gestion des valeurs manquantes
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data)
    data_clean = pd.DataFrame(data_imputed, columns=feature_columns)

    return data_clean

# Préparer les données
data = prepare_data()

correlation_matrix = data.corr()

# Sélectionner la corrélation avec la variable cible 'total_voix_1er_tour'
correlation_with_target = correlation_matrix['total_voix_1er_tour'].drop(['total_voix_1er_tour', 'total_voix_2e_tour'])

# Trier les corrélations par ordre décroissant
correlation_sorted = correlation_with_target.sort_values(ascending=False)

# Trier les corrélations par ordre décroissant
correlation_sorted = correlation_with_target.sort_values(ascending=False)

#----------------------------------------------------------------------------------------------
# Afficher les indices de corrélation
# print("Corrélation des variables explicatives avec 'total_voix_1er_tour':")
# print(correlation_sorted)

#----------------------------------------------------------------------------------------------
# Création d'une heatmap
# plt.figure(figsize=(6, 4))
# correlation_sorted.plot(kind='barh')
# plt.title('Corrélation des variables explicatives avec total_voix_1er_tour')
# plt.xlabel('Corrélation')
# plt.ylabel('Variables')
# plt.subplots_adjust(left=0.3)
# plt.show()

#----------------------------------------------------------------------------------------------
# Visualiser la matrice de corrélation avec une carte de chaleur
# mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
# plt.title('Carte de Chaleur des Corrélations entre Variables Explicatives')
# plt.subplots_adjust(left=0.2)
# plt.subplots_adjust(bottom=0.4)
# plt.show()


# Séparation des données en caractéristiques (X) et cible (y)
X = data.drop(columns=['total_voix_1er_tour', 'total_voix_2e_tour'])
y = data['total_voix_1er_tour']


# # Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# # Séparation des données en ensembles d'entraînement et de test
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

#----------------------------------------------------------------------------------------------
# Entraînement du modèle Lasso
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)

# Prédictions avec le modèle Lasso
# y_pred_lasso = lasso_model.predict(X_test)

# # Évaluation du modèle Lasso
# mse_lasso = mean_squared_error(y_test, y_pred_lasso)
# r2_lasso = r2_score(y_test, y_pred_lasso)

# print(f"Lasso Regression - Mean Squared Error: {mse_lasso}")
# print(f"Lasso Regression - R² Score: {r2_lasso}")

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

# print("Variables sélectionnées par le modèle Lasso :")
# print(selected_features)
#---------------------------------------------------------

#---------------------------------------------------------
# Visualisation des coefficients des modèles
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.bar(X.columns, ridge_model.coef_)
# plt.title('Coefficients du modèle Ridge')
# plt.xlabel('Variables')
# plt.ylabel('Coefficients')
# plt.xticks(rotation=90)

# plt.subplot(1, 2, 2)
# plt.bar(X.columns, lasso_model.coef_)
# plt.title('Coefficients du modèle Lasso')
# plt.xlabel('Variables')
# plt.ylabel('Coefficients')
# plt.xticks(rotation=90)

# plt.tight_layout()
# plt.show()

#----------------------------------------------------------------------------------------------
#Mise en place du modèle final avec les variables les plus pertinentes
# Filtrer les données pour ne conserver que les variables sélectionnées
X_selected = X[selected_features]

# Standardisation des données sélectionnées
scaler_selected = StandardScaler()
X_selected_scaled = scaler_selected.fit_transform(X_selected)

# Séparation des données sélectionnées en ensembles d'entraînement et de test
X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(X_selected_scaled, y, test_size=0.2, random_state=42)

#-------------
# Entraînement du modèle final 
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Entraînement du modèle
    model.fit(X_train, y_train)
    
    # Prédictions avec le modèle
    y_pred = model.predict(X_test)
    
    # Évaluation du modèle
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name} - Mean Squared Error: {mse}")
    print(f"{model_name} - R² Score: {r2}")

# Exemple avec le modèle Ridge
final_model_ridge = Ridge(alpha=1.0)
evaluate_model(final_model_ridge, X_train_selected, X_test_selected, y_train_selected, y_test_selected, "Ridge Regression")

# Exemple avec le modèle Lasso
model_lasso = Lasso(alpha=1.0)
evaluate_model(model_lasso, X_train_selected, X_test_selected, y_train_selected, y_test_selected, "Lasso Regression")

# Exemple avec le modèle Forêts Aléatoires
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
evaluate_model(model_rf, X_train_selected, X_test_selected, y_train_selected, y_test_selected, "Random Forest")

# Exemple avec le modèle Gradient Boosting
model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
evaluate_model(model_gb, X_train_selected, X_test_selected, y_train_selected, y_test_selected, "Gradient Boosting")

#----------------------------------------------------------------------------------------------
# Validation croisée pour confirmer la performance du modèle
# cv_scores = cross_val_score(final_model, X_selected_scaled, y, cv=5, scoring='neg_mean_squared_error')
# cv_mse = -cv_scores.mean()

# print(f"Final Model (Ridge Regression) - Cross-Validated Mean Squared Error: {cv_mse}")

# # Visualisation des coefficients du modèle final
# plt.figure(figsize=(12, 6))
# plt.bar(selected_features, final_model.coef_)
# plt.title('Coefficients du modèle final (Ridge Regression)')
# plt.xlabel('Variables')
# plt.ylabel('Coefficients')
# plt.xticks(rotation=90)
# plt.show()

#----------------------------------------------------------------------------------------------
# # Entraînement du modèle de régression linéaire
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Prédictions
# y_pred = model.predict(X_test)

# # Évaluation du modèle
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"R² Score: {r2}")