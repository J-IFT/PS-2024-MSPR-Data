import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------------------------------------------------------------------------------
#PREDICTIONS POUR LE SECOND TOUR
# print(final_df)
# final_df['Prédiction du nombre de votes'] = final_df['Prédiction du nombre de votes'].astype(float)
# final_df_sorted = final_df.sort_values(by='Prédiction du nombre de votes', ascending=False)
# donnees_cibles_second_tour = final_df_sorted.head(2)['Parti']
# print(donnees_cibles_second_tour)
# donnees_cibles_second_tour = donnees_cibles_second_tour + '_second_tour'

# final_results = []

donnees_cibles_second_tour = [
    'Renaissance_second_tour', 
    'Rassemblement_national_second_tour'
]

# Préparer les données
def prepare_data(parti):
    data = pd.read_excel("data/fichier_final.xlsx", decimal=',')
    
    feature_columns = [
        parti, # Variable cible
        'total_au_chomage', 'Lutte_ouvrière', 
        'Parti_communiste_français', 'Renaissance', 'Résistons', 
        'Rassemblement_national', 'Reconquête', 'La_France_Insoumise', 
        'Le_partie_socialiste', 'Europe_écologie_Les_verts', 
        'Les_républicains', 'Le_nouveau_parti_anticapitaliste', 'Debout_la_France',
        'Agriculteurs exploitants', 'Artisans_commerçants_chefs_d_entreprise', 
        'Cadre_et_profession_intellectuelle_supérieure', 'Profession_intermédiaire', 'Employe', 'Ouvrier',
        'DIPLMIN_total', 'CAPBEP_total', 'BAC_total', 'SUP34_total', 'SUP5_total', 'nombre_habitants', 'nombre_immigration',
        'Médiane_du_niveau_de_vie', 'generation_55_64', 'Salaire_net_horaire_moyen', 'pourcentage_abstention'
    ]

    # Sélectionner les colonnes utiles dans les données
    data = data[feature_columns].apply(pd.to_numeric, errors='coerce')
    
    return data

final_results = []


for index, parti in enumerate(donnees_cibles_second_tour):
    data = prepare_data(parti)

    #---------------------------------------
    # Séparation des données en caractéristiques (X) et cible (y)
    X = data.drop(columns=[parti])
    y = data[parti]

    # Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Entraînement du modèle Lasso
    # lasso_model = Lasso(alpha=1)
    # lasso_model.fit(X_train, y_train)

    # # Prédictions avec le modèle Lasso
    # y_pred_lasso = lasso_model.predict(X_test)

    # # Évaluation du modèle Lasso
    # mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    # r2_lasso = r2_score(y_test, y_pred_lasso)
    # coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': lasso_model.coef_})

    # #Selection des 5 variables les plus explicatives
    # selected_features = coefficients[coefficients['Coefficient'] != 0]
    # selected_features['abs_Coefficient'] = selected_features['Coefficient'].abs()
    # top_features = selected_features.sort_values(by='abs_Coefficient', ascending=False).head(5)
    # top_features_list = '\n'.join(top_features['Variable'].tolist())

    # print("Top 5 variables sélectionnées par le modèle Lasso :")
    # print(top_features[['Variable', 'Coefficient']])

    # Entraînement du modèle Gradient Boosting
    # gb_model = GradientBoostingRegressor(random_state=42)
    # gb_model.fit(X_train, y_train)

    # # Prédictions avec le modèle Gradient Boosting
    # y_pred_gb = gb_model.predict(X_test)

    # # Évaluation du modèle Gradient Boosting
    # mse_gb = mean_squared_error(y_test, y_pred_gb)
    # r2_gb = r2_score(y_test, y_pred_gb)

    # # Extraction des importances des features
    # importances = gb_model.feature_importances_
    # importance_df = pd.DataFrame({'Variable': X.columns, 'Importance': importances})
    
    # Entraînement du modèle Random Forest
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)

    # Prédictions avec le modèle Random Forest
    y_pred_rf = rf_model.predict(X_test)
    
    # Extraction des importances des features
    importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({'Variable': X.columns, 'Importance': importances})

    # Sélection des 5 variables les plus explicatives
    top_features = importance_df.sort_values(by='Importance', ascending=False).head(5)
    top_features_list = '\n'.join(top_features['Variable'].tolist())

    def prepare_data_with_selected_features(parti, top_features):
        data = pd.read_excel("data/fichier_final.xlsx", decimal=',')
        feature_columns = list(top_features['Variable']) + [parti]
        data = data[feature_columns].apply(pd.to_numeric, errors='coerce')
        
        return data

    data = prepare_data_with_selected_features(parti, top_features)
    
    X = data.drop(columns=[parti])
    X_scaled = scaler.fit_transform(X)

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Dictionnaire pour stocker les résultats des modèles
    model_results = {}

    # Entraînement du modèle Ridge avec validation croisée
    ridge_model = RidgeCV(cv=5).fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    model_results['Ridge'] = {'mse': mse_ridge, 'r2': r2_ridge}

    # Entraînement du modèle Lasso avec validation croisée
    lasso_model = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
    y_pred_lasso = lasso_model.predict(X_test)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)
    model_results['Lasso'] = {'mse': mse_lasso, 'r2': r2_lasso}

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
    model_results['Random Forest'] = {'mse': mse_rf, 'r2': r2_rf}

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
    model_results['Gradient Boosting'] = {'mse': mse_gb, 'r2': r2_gb}

    # Sélection du meilleur modèle basé sur le r² le plus élevé et le MSE le plus faible
    best_model = max(model_results.items(), key=lambda x: (x[1]['r2'], -x[1]['mse']))
    
    best_model_instance = None
    if best_model[0] == 'Gradient Boosting':
        best_model_instance = gb_model
    elif best_model[0] == 'Random Forest':
        best_model_instance = rf_model
    elif best_model[0] == 'Lasso':
        best_model_instance = lasso_model
    elif best_model[0] == 'Ridge':
        best_model_instance = ridge_model

    # Prédiction du nombre de votes pour ce parti
    y_pred = best_model_instance.predict(X_test)
    
    # Calcul de la moyenne des prédictions
    avg_prediction = np.mean(y_pred)

    final_results.append({
        'Parti': parti,
        'Variables Explicatives': top_features_list,  
        'MSE et R²': f"MSE: {best_model[1]['mse']:.4f}, R²: {best_model[1]['r2']:.4f}",
        'Modèle Retenu': best_model[0],
        'Prédiction du nombre de votes': f"{avg_prediction:.2f}"
    })
    
final_df = pd.DataFrame(final_results)

# Afficher le tableau avec matplotlib
fig, ax = plt.subplots(figsize=(12, 8))  
ax.axis('tight')
ax.axis('off')

# Créer une table avec matplotlib
table = ax.table(cellText=final_df.values, colLabels=final_df.columns, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 6)  

colors = ['#f0f0f0', '#d0e0e3']
for i, key in enumerate(final_df.columns):
    table[0, i].set_facecolor('#4CAF50')  
    table[0, i].set_text_props(color='white', weight='bold') 
    for j in range(1, len(final_df) + 1):
        table[j, i].set_facecolor(colors[j % 2]) 

for i in range(len(final_df.columns)):
    table.auto_set_column_width(i)

plt.show()
