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

# Charger les données
resultats = pd.read_excel("data/resultats_electoraux.xlsx")
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

# Définir les variables explicatives et la variable cible
X = data.drop(columns=['total_voix_1er_tour', 'total_voix_2e_tour'])
y = data['total_voix_1er_tour']

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fonction pour calculer l'accuracy
def calculate_accuracy(y_true, y_pred, tolerance=0.05):
    correct_predictions = np.abs(y_true - y_pred) <= tolerance * np.abs(y_true)
    return np.mean(correct_predictions)

# Sélection des caractéristiques importantes avec RFE et Importance des caractéristiques
def feature_selection_rfe(X_train, y_train, model, n_features=10):
    selector = RFE(model, n_features_to_select=n_features)
    selector = selector.fit(X_train, y_train)
    return X.columns[selector.support_]

def feature_selection_importance(X_train, y_train, model, threshold=0.05):
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    return importance_df[importance_df['Importance'] > threshold]['Feature'].tolist()

# Test des modèles
models = {
    'Régression Linéaire': LinearRegression(),
    'Régression Ridge': Ridge(),
    'Régression Lasso': Lasso(),
    'Arbre de Décision': DecisionTreeRegressor(),
    'Forêt Aléatoire': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'SVM': SVR(),
    'k-NN': KNeighborsRegressor()
}

# Évaluation des caractéristiques avec RFE et Importance des caractéristiques
selected_features = {}
for name, model in models.items():
    if name in ['Régression Linéaire', 'Régression Ridge', 'Régression Lasso', 'Arbre de Décision', 'Forêt Aléatoire', 'Gradient Boosting']:
        selected_features[name] = feature_selection_rfe(X_scaled, y, model)
    elif name in ['SVM', 'k-NN']:
        # Ne pas utiliser la sélection basée sur l'importance pour ces modèles
        selected_features[name] = X.columns.tolist()

# Affichage des caractéristiques sélectionnées
for model_name, features in selected_features.items():
    print(f"Caractéristiques sélectionnées pour {model_name}:")
    print(features)
    print()

# Essai avec les caractéristiques sélectionnées
results = {}
for name, model in models.items():
    selected_cols = selected_features.get(name, X.columns)
    X_selected = pd.DataFrame(X_scaled, columns=X.columns)[selected_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model.fit(X_train_scaled, y_train)
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    accuracy = calculate_accuracy(y_test, y_pred_test)
    
    results[name] = {
        'Train Mean Squared Error': train_mse,
        'Train R2 Score': train_r2,
        'Test Mean Squared Error': test_mse,
        'Test R2 Score': test_r2,
        'Test Accuracy': accuracy
    }

# Affichage des résultats
results_df = pd.DataFrame(results).T
print("Résultats des différents modèles :")
print(results_df)

# Sauvegarde des résultats dans un fichier Excel
results_df.to_excel('resultats_modeles_avec_selection.xlsx')

# Visualisation des résultats (optionnel)
results_df[['Test Mean Squared Error', 'Test R2 Score']].plot(kind='bar', figsize=(12, 6))
plt.title('Performance des Modèles')
plt.ylabel('Score')
plt.xlabel('Modèles')
plt.show()
