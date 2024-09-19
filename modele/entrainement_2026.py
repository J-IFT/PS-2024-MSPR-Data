import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


donnees_cibles_premier_tour = [
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

def prepare_data(parti):
    data = pd.read_excel("data/fichier_final.xlsx", decimal=',')
    
    feature_columns = [
        parti, # Variable cible
        'total_au_chomage', 
        'Agriculteurs exploitants', 'Artisans_commerçants_chefs_d_entreprise', 
        'Cadre_et_profession_intellectuelle_supérieure', 'Profession_intermédiaire', 'Employe', 'Ouvrier',
        'DIPLMIN_total', 'CAPBEP_total', 'BAC_total', 'SUP34_total', 'SUP5_total', 'nombre_habitants', 
        'nombre_immigration',
        'Médiane_du_niveau_de_vie', 'generation_55_64', 'Salaire_net_horaire_moyen', 'pourcentage_abstention'
    ]

    data = data[feature_columns].apply(pd.to_numeric, errors='coerce')
    
    return data

final_results = []

for index, parti in enumerate(donnees_cibles_premier_tour):
    data = prepare_data(parti)

    #---------------------------------------
    # Séparation des données en caractéristiques (X) et cible (y)
    X = data.drop(columns=[parti])
    y = data[parti]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Gradient Boosting
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_model.fit(X_train, y_train)

    y_pred_gb = gb_model.predict(X_test)

    # Extraction des importances des features
    importances = gb_model.feature_importances_
    importance_df = pd.DataFrame({'Variable': X.columns, 'Importance': importances})

    top_features = importance_df.sort_values(by='Importance', ascending=False).head(5)
    top_features_list = '\n'.join(top_features['Variable'].tolist())

    def prepare_data_with_selected_features(parti, importance_df):
        data = pd.read_excel("data/fichier_final.xlsx", decimal=',')
        feature_columns = list(importance_df['Variable']) + [parti]
        data = data[feature_columns].apply(pd.to_numeric, errors='coerce')
        
        return data

    data = prepare_data_with_selected_features(parti, importance_df)
    
    X = data.drop(columns=[parti])
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Gradient Boosting
    param_grid_gb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
    gb_model = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb, cv=5).fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)

    # Prédiction du nombre de votes pour ce parti
    y_pred = gb_model.predict(X_test)
    avg_prediction = np.mean(y_pred)
    
    #----------------------------------------------------------------------------
    #prédictions 2026
    def prepare_data_2026(parti):
        data = pd.read_excel("data/fichier_final.xlsx", decimal=',')
        
        feature_columns = [
            parti,
            'total_au_chomage_2026', 
            'Agriculteurs exploitants', 'Artisans_commerçants_chefs_d_entreprise', 
            'Cadre_et_profession_intellectuelle_supérieure', 'Profession_intermédiaire', 'Employe', 'Ouvrier',
            'DIPLMIN_total', 'CAPBEP_total', 'BAC_total', 'SUP34_total', 'SUP5_total', 'nombre_habitants', 
            'nombre_immigration_2026',
            'Médiane_du_niveau_de_vie', 'generation_55_64', 'Salaire_net_horaire_moyen', 'pourcentage_abstention'
        ]

        data = data[feature_columns].apply(pd.to_numeric, errors='coerce')  
        return data
    data_2026 = prepare_data_2026(parti)

    X_2026 = data_2026.drop(columns=[parti])
    y = data_2026[parti]
    X_2026_scaled = scaler.fit_transform(X_2026)
    y_pred_2026 = gb_model.predict(X_2026_scaled)
    avg_prediction_2026 = np.mean(y_pred_2026)
    
    final_results.append({
        'Parti': parti,
        'Prédiction pour 2023': f"{avg_prediction:.2f}",
        'Prédiction pour 2026': f"{avg_prediction_2026:.2f}"
    })
    
final_df = pd.DataFrame(final_results)

fig, ax = plt.subplots(figsize=(10, 5))  
ax.axis('tight')
ax.axis('off')

# Créer une table avec matplotlib
table = ax.table(cellText=final_df.values, colLabels=final_df.columns, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(6)
table.scale(1.2, 2)  

# Coloration des colonnes (optionnel)
colors = ['#f0f0f0', '#d0e0e3']
for i, key in enumerate(final_df.columns):
    table[0, i].set_facecolor('#4CAF50')  
    table[0, i].set_text_props(color='white', weight='bold') 
    for j in range(1, len(final_df) + 1):
        table[j, i].set_facecolor(colors[j % 2]) 

for i in range(len(final_df.columns)):
    table.auto_set_column_width(i)

plt.show()
