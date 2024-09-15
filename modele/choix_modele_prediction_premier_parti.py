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
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# Préparer les données
def prepare_data():
    data = pd.read_excel("data/fichier_final.xlsx", decimal=',')
    
    feature_columns = [
        'Lutte_ouvrière', # Variable cible
        'CAPBEP_total','SUP5_total',
        'pourcentage_abstention', 'Agriculteurs exploitants'
    ]

    # Sélectionner les colonnes utiles dans les données
    data = data[feature_columns].apply(pd.to_numeric, errors='coerce')
    
    return data

data = prepare_data()

# TEST DE PLUSIEURS MODELES POUR TROUVER LE PLUS PREDICTIF
#---------------------------------------
# Séparation des données en caractéristiques (X) et cible (y)
X = data.drop(columns=['Lutte_ouvrière'])
y = data['Lutte_ouvrière']

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -------------
# Entraînement du modèle final 
# def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
#     # Entraînement du modèle
#     model.fit(X_train, y_train)
    
#     # Prédictions avec le modèle
#     y_pred = model.predict(X_test)
    
#     # Évaluation du modèle
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
    
#     print(f"{model_name} - Mean Squared Error: {mse}")
#     print(f"{model_name} - R² Score: {r2}")

# # Exemple avec le modèle Ridge
# final_model_ridge = Ridge(alpha=1.0)
# evaluate_model(final_model_ridge, X_train, X_test, y_train, y_test, "Ridge Regression")

# # Exemple avec le modèle Lasso
# model_lasso = Lasso(alpha=1.0)
# evaluate_model(model_lasso, X_train, X_test, y_train, y_test, "Lasso Regression")

# # Exemple avec le modèle Forêts Aléatoires
# model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
# evaluate_model(model_rf, X_train, X_test, y_train, y_test, "Random Forest")

# # Exemple avec le modèle Gradient Boosting
# model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
# evaluate_model(model_gb, X_train, X_test, y_train, y_test, "Gradient Boosting")


#-------------------------------------------------------------
#Validation croisée
# Entraînement du modèle Ridge avec validation croisée
ridge_model = RidgeCV(cv=5).fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"Ridge Regression - Mean Squared Error: {mse_ridge}")
print(f"Ridge Regression - R² Score: {r2_ridge}")

# Entraînement du modèle Lasso avec validation croisée
lasso_model = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f"Lasso Regression - Mean Squared Error: {mse_lasso}")
print(f"Lasso Regression - R² Score: {r2_lasso}")

# Entraînement du modèle Random Forest avec GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5).fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest - Mean Squared Error: {mse_rf}")
print(f"Random Forest - R² Score: {r2_rf}")

# Entraînement du modèle Gradient Boosting avec GridSearchCV
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
gb_model = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb, cv=5).fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
print(f"Gradient Boosting - Mean Squared Error: {mse_gb}")
print(f"Gradient Boosting - R² Score: {r2_gb}")
