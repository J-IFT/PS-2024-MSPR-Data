import pandas as pd

df14 = pd.read_excel('Emploi-Chômage-Retraite/traitement knime/Données-emploi-démographie-2014.xlsx', engine='openpyxl')
def remove_year_prefixes(df14):
    # Créer un dictionnaire pour les nouveaux noms de colonnes
    new_columns = {}
    
    # Parcourir les noms de colonnes actuels
    for col in df14.columns:
        # Enlever les préfixes comme 'P14-' ou 'P20-'
        new_name = col.replace('P14_', '').replace('C14_', '')
        # Ajouter le nouveau nom au dictionnaire
        new_columns[col] = new_name
    
    # Renommer les colonnes en utilisant le dictionnaire
    df14 = df14.rename(columns=new_columns)
    return df14

# Appliquer la fonction pour renommer les colonnes
df14 = remove_year_prefixes(df14)

df14.to_excel('Emploi-Chômage-Retraite/traitement knime/Données-emploi-démographie-2014.xlsx', index=False, engine='openpyxl')

df20 = pd.read_excel('Emploi-Chômage-Retraite/traitement knime/Données-emploi-démographie-2020.xlsx', engine='openpyxl')
def remove_year_prefixes(df20):
    # Créer un dictionnaire pour les nouveaux noms de colonnes
    new_columns = {}
    
    # Parcourir les noms de colonnes actuels
    for col in df20.columns:
        # Enlever les préfixes comme 'P14-' ou 'P20-'
        new_name = col.replace('P20_', '').replace('C20_', '')
        # Ajouter le nouveau nom au dictionnaire
        new_columns[col] = new_name
    
    # Renommer les colonnes en utilisant le dictionnaire
    df20 = df20.rename(columns=new_columns)
    return df20

# Appliquer la fonction pour renommer les colonnes
df20 = remove_year_prefixes(df20)

df20.to_excel('Emploi-Chômage-Retraite/traitement knime/Données-emploi-démographie-2020.xlsx', index=False, engine='openpyxl')



df17 = pd.read_excel('Diplômes/traitement knime/diplomes-formation-commune-2017.xlsx', engine='openpyxl')

def remove_year_prefixes(df17):
    # Créer un dictionnaire pour les nouveaux noms de colonnes
    new_columns = {}
    
    # Parcourir les noms de colonnes actuels
    for col in df17.columns:
        # Enlever les préfixes comme 'P14-' ou 'P20-'
        new_name = col.replace('P17_', '')
        # Ajouter le nouveau nom au dictionnaire
        new_columns[col] = new_name
    
    # Renommer les colonnes en utilisant le dictionnaire
    df17 = df17.rename(columns=new_columns)
    return df17

# Appliquer la fonction pour renommer les colonnes
df17 = remove_year_prefixes(df17)

df17.to_excel('Diplômes/traitement knime/diplomes-formation-commune-2017.xlsx', index=False, engine='openpyxl')



df21 = pd.read_excel('Diplômes/traitement knime/diplomes-formation-commune-2021.xlsx', engine='openpyxl')

def remove_year_prefixes(df21):
    # Créer un dictionnaire pour les nouveaux noms de colonnes
    new_columns = {}
    
    # Parcourir les noms de colonnes actuels
    for col in df21.columns:
        # Enlever les préfixes comme 'P14-' ou 'P20-'
        new_name = col.replace('P21_', '')
        # Ajouter le nouveau nom au dictionnaire
        new_columns[col] = new_name
    
    # Renommer les colonnes en utilisant le dictionnaire
    df21 = df21.rename(columns=new_columns)
    return df21

# Appliquer la fonction pour renommer les colonnes
df21 = remove_year_prefixes(df21)

df21.to_excel('Diplômes/traitement knime/diplomes-formation-commune-2021.xlsx', index=False, engine='openpyxl')
