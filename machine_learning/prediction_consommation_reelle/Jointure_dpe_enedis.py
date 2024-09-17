import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import missingno as msno
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Définir les chemins des fichiers DPE neufs
file_path_neuf_2021 = './Fichiers_Sources/dpe_logements_neufs_2021.csv'
# Définir les chemins des fichiers DPE existants
file_path_existant_21 = './Fichiers_Sources/dpe_logements_existants_2021.csv'

# Lire et concaténer les fichiers DPE neufs + existent
df_neuf_2021 = pd.read_csv(file_path_neuf_2021)
df_existant_21 = pd.read_csv(file_path_existant_21)

#j'ai ajouté une colonne type batiment pour chaque df pour distinguer apres la concaténation (si on veut filtrer )
df_neuf_2021['type_batiment_add'] = 'neuf'
df_existant_21['type_batiment_add'] = 'existant'
# Trouver les colonnes communes entre les deux DataFrames
colonnes_communes = df_neuf_2021.columns.intersection(df_existant_21.columns)
# Concaténer uniquement les colonnes communes
df_final_combined = pd.concat([df_neuf_2021[colonnes_communes], df_existant_21[colonnes_communes]], ignore_index=True)

df_final_combined['Adresse_(BAN)'] = df_final_combined['Adresse_(BAN)'].str.lower()
df_final_combined['Adresse_(BAN)'] = df_final_combined['Adresse_(BAN)'].str.strip()
df_final_combined['Adresse_(BAN)'] = df_final_combined['Adresse_(BAN)'].str.replace(r'\s+', ' ', regex=True)

# Garder la dernière occurrence pour chaque 'Identifiant__BAN'
df_sans_doublons = df_final_combined.drop_duplicates(subset=['Identifiant__BAN'], keep='last')
# Supprimer les colonnes ayant un % de null >90
seuil = 90
colonnes_a_supprimer = missing_values[missing_values > seuil].index
df_ss_doublons = df_sans_doublons.drop(columns=colonnes_a_supprimer)

# Imputer avec la médiane pour les colonnes numériques
num_cols = df_sans_doublons.select_dtypes(include=['number']).columns
for col in num_cols :
    df_sans_doublons[col].fillna(df_sans_doublons[col].median(), inplace=True)

# Imputer avec la valeur la plus fréquente pour les colonnes catégorielles
cat_cols = df_sans_doublons.select_dtypes(include=['object', 'category']).columns

for col in cat_cols:
    df_sans_doublons[col].fillna(df_sans_doublons[col].mode()[0], inplace=True)

# Supprimer les lignes ayant plus de 50% de valeurs manquantes
seuil_lignes = 50
df_combined_2 = df_sans_doublons[df_sans_doublons.isnull().mean(axis=1) < (seuil_lignes / 100)]

df_combined_2 = df_combined_2.drop(columns=['_score'])

# Calcul des quartiles (Q1 et Q3) et de l'IQR (Interquartile Range)
Q1 = df_combined_2['Conso_5_usages_é_finale'].quantile(0.25)
Q3 = df_combined_2['Conso_5_usages_é_finale'].quantile(0.75)
IQR = Q3 - Q1

# Définir les seuils pour identifier les outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtrer les valeurs aberrantes (outliers)
outliers = df_combined_2[(df_combined_2['Conso_5_usages_é_finale'] < lower_bound) | (df_combined_2['Conso_5_usages_é_finale'] > upper_bound)]

# Remplacer les valeurs en dehors des limites par Q1 ou Q3
df_combined_2['Conso_5_usages_é_finale'] = df_combined_2['Conso_5_usages_é_finale'].apply(
    lambda x: Q1 if x < lower_bound else (Q3 if x > upper_bound else x)
)

# **************************ENEDIS**********************
# Définir les chemins des fichiers
file_path_enedis_23 = './Fichiers_Sources/consommation-annuelle-residentielle_2023/part-00000-4302ba05-6ec8-458a-ab7c-112d8d56f816-c000.csv'
file_path_enedis_22 = './Fichiers_Sources/consommation-annuelle-residentielle_2022/part-00000-79cc601a-85e7-4b4e-ab4d-ba2bad69af14-c000.csv'
file_path_enedis_21 = './Fichiers_Sources/consommation-annuelle-residentielle_2021/part-00000-11e6d32b-59a4-475d-881e-57c3277239c1-c000.csv'

try:
    Enedis_23 = pd.read_csv(file_path_enedis_23, delimiter=';')
    print("Lecture réussie pour le fichier 2023")
except Exception as e:
    print(f"Erreur lors de la lecture du fichier 2023 : {e}")

try:
    Enedis_22 = pd.read_csv(file_path_enedis_22, delimiter=';')
    print("Lecture réussie pour le fichier 2022")
except Exception as e:
    print(f"Erreur lors de la lecture du fichier 2022 : {e}")

try:
    Enedis_21 = pd.read_csv(file_path_enedis_21, delimiter=';')
    print("Lecture réussie pour le fichier 2021")
except Exception as e:
    print(f"Erreur lors de la lecture du fichier 2021 : {e}")

# Fusionner les DataFrames en un seul
df_combined_enedis = pd.concat([Enedis_23, Enedis_22, Enedis_21], ignore_index=True)

# Afficher les premières lignes du DataFrame fusionné
# print(df_combined_enedis.head())

# # Sauvegarder le DataFrame fusionné dans un nouveau fichier CSV (optionnel)
# df_combined_enedis.to_csv('C:/Users/33780/pfe/Dpe_J_ENDEIS_VF/consommation-annuelle-residentielle_combined.csv', index=False, sep=';')
# 2021
df_combined_enedis_clean = Enedis_21.drop_duplicates(subset=['id'], keep='first')
# Calculer le mode de 'type_de_voie'
mode_type_de_voie = df_combined_enedis_clean['type_de_voie'].mode()[0]
df_combined_enedis_clean['indice_de_repetition'] = df_combined_enedis_clean['indice_de_repetition'].fillna('')
df_combined_enedis_clean['type_de_voie'] = df_combined_enedis_clean['type_de_voie'].fillna(mode_type_de_voie)

colonnes_interet = ['id', 'annee', 'consommation_annuelle_totale_de_l_adresse_mwh', 
                    'consommation_annuelle_moyenne_par_site_de_l_adresse_mwh',  'consommation_annuelle_moyenne_de_la_commune_mwh','adresse_normalisee_insee','nombre_de_logements']

df_combined_enedis_clean = df_combined_enedis_clean[colonnes_interet]
# Renommer la colonne 'Identifiant__BAN' en 'id' dans df_combined_2
df_combined_2.rename(columns={'Identifiant__BAN': 'id'}, inplace=True)
# Effectuer la jointure droite (right join) sur la colonne 'id'
df_joint = pd.merge(df_combined_2, df_combined_enedis_clean, on='id', how='inner')

# Sélection des colonnes numériques valides (sans NaN ou valeurs non numériques)
colonnes_numeriques_valides = df_joint.select_dtypes(include=[np.number]).columns

# # Remplacer les valeurs manquantes par la médiane pour éviter les erreurs lors du calcul des quantiles
# df_joint[colonnes_numeriques_valides] = df_joint[colonnes_numeriques_valides].fillna(df_joint[colonnes_numeriques_valides].median())

# Calcul des quartiles et de l'IQR pour chaque colonne numérique
Q1 = df_joint[colonnes_numeriques_valides].quantile(0.25)  # Premier quartile (25%)
Q3 = df_joint[colonnes_numeriques_valides].quantile(0.75)  # Troisième quartile (75%)
IQR = Q3 - Q1  # Intervalle interquartile

# Boucle pour traiter les outliers dans chaque colonne numérique
for col in colonnes_numeriques_valides:
    lower_bound = Q1[col] - 1.5 * IQR[col]  # Borne inférieure
    upper_bound = Q3[col] + 1.5 * IQR[col]  # Borne supérieure
    # Limiter les valeurs aux bornes
    df_joint[col] = np.clip(df_joint[col], lower_bound, upper_bound)
df_joint['consommation_estimee_dpe_mwh'] = df_joint['Conso_5_usages_é_finale'] / 1000  # conversion en MWh 
df_joint['ecart_consommation'] = df_joint['consommation_annuelle_moyenne_par_site_de_l_adresse_mwh'] - df_joint['consommation_estimee_dpe_mwh']
# Sélectionner les colonnes spécifiques
df_selection = df_joint[['ecart_consommation', 'consommation_estimee_dpe_mwh', 'consommation_annuelle_moyenne_par_site_de_l_adresse_mwh', 'Adresse_(BAN)', 'annee']]

# 2022

# Définir les chemins des fichiers DPE neufs
file_path_neuf_2022 = './Fichiers_Sources/dpe_logements_neufs_2022.csv'
df_neuf_2022 = pd.read_csv(file_path_neuf_2022)
# Définir les chemins des fichiers DPE existants

file_path_existant_22 = './Fichiers_Sources/dpe_logements_existants_2022.csv'
df_existant_22 = pd.read_csv(file_path_existant_22)
#j'ai ajouté une colonne type batiment pour chaque df pour distinguer apres la concaténation (si on veut filtrer )
df_neuf_2022['type_batiment_add'] = 'neuf'
df_existant_22['type_batiment_add'] = 'existant'
# Trouver les colonnes communes entre les deux DataFrames
colonnes_communes = df_neuf_2022.columns.intersection(df_existant_22.columns)
# Concaténer uniquement les colonnes communes
df_final_combined_22 = pd.concat([df_neuf_2022[colonnes_communes], df_existant_22[colonnes_communes]], ignore_index=True)
# Garder la dernière occurrence pour chaque 'Identifiant__BAN'
df_sans_doublons_22 = df_final_combined_22.drop_duplicates(subset=['Identifiant__BAN'], keep='last')
colonnes_a_supprimer = missing_values[missing_values > seuil].index
df_sans_doublons_22 = df_sans_doublons_22.drop(columns=colonnes_a_supprimer)
# Imputer avec la médiane pour les colonnes numériques
num_cols = df_sans_doublons_22.select_dtypes(include=['number']).columns
for col in num_cols :
    df_sans_doublons_22[col].fillna(df_sans_doublons_22[col].median(), inplace=True)
# Imputer avec la valeur la plus fréquente pour les colonnes catégorielles
cat_cols = df_sans_doublons_22.select_dtypes(include=['object', 'category']).columns

for col in cat_cols:
    df_sans_doublons_22[col].fillna(df_sans_doublons_22[col].mode()[0], inplace=True)
df_combined_22 = df_sans_doublons_22[df_sans_doublons_22.isnull().mean(axis=1) < (seuil_lignes / 100)]

df_combined_22 = df_combined_22.drop(columns=['_score'])
# Calcul des quartiles (Q1 et Q3) et de l'IQR (Interquartile Range)
Q1 = df_combined_22['Conso_5_usages_é_finale'].quantile(0.25)
Q3 = df_combined_22['Conso_5_usages_é_finale'].quantile(0.75)
IQR = Q3 - Q1

# Définir les seuils pour identifier les outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtrer les valeurs aberrantes (outliers)
outliers = df_combined_22[(df_combined_22['Conso_5_usages_é_finale'] < lower_bound) | (df_combined_22['Conso_5_usages_é_finale'] > upper_bound)]

# Remplacer les valeurs en dehors des limites par Q1 ou Q3
df_combined_22['Conso_5_usages_é_finale'] = df_combined_22['Conso_5_usages_é_finale'].apply(
    lambda x: Q1 if x < lower_bound else (Q3 if x > upper_bound else x)
)
df_combined_enedis_clean_22 = Enedis_22.drop_duplicates(subset=['id'], keep='first')



# Calculer le mode de 'type_de_voie'
mode_type_de_voie = df_combined_enedis_clean_22['type_de_voie'].mode()[0]
df_combined_enedis_clean_22['indice_de_repetition'] = df_combined_enedis_clean_22['indice_de_repetition'].fillna('')
df_combined_enedis_clean_22['type_de_voie'] = df_combined_enedis_clean_22['type_de_voie'].fillna(mode_type_de_voie)

# Appliquer une transformation logarithmique
df_combined_enedis_clean_22['log_conso'] = np.log1p(df_combined_enedis_clean_22['consommation_annuelle_moyenne_par_site_de_l_adresse_mwh'])  # log(1 + x) pour éviter log(0)
df_combined_enedis_clean_22 = df_combined_enedis_clean_22[colonnes_interet]

# Renommer la colonne 'Identifiant__BAN' en 'id' dans df_combined_2
df_combined_22.rename(columns={'Identifiant__BAN': 'id'}, inplace=True)
# Effectuer la jointure droite (right join) sur la colonne 'id'
df_joint_22= pd.merge(df_combined_22, df_combined_enedis_clean_22, on='id', how='inner')
df_joint_22['consommation_estimee_dpe_mwh'] = df_joint_22['Conso_5_usages_é_finale'] / 1000  # conversion en MWh 
df_joint_22['ecart_consommation'] = df_joint_22['consommation_annuelle_moyenne_par_site_de_l_adresse_mwh']-df_joint_22['consommation_estimee_dpe_mwh']


# Définir les chemins des fichiers DPE neufs
file_path_neuf_2023 = './Fichiers_Sources/dpe_logements_neufs_2023.csv'
df_neuf_2023 = pd.read_csv(file_path_neuf_2023)
# Définir les chemins des fichiers DPE existants
file_path_existant_23 = './Fichiers_Sources/dpe_logements_existants_2023.csv'
df_existant_23 = pd.read_csv(file_path_existant_23)

df_neuf_2023['type_batiment_add'] = 'neuf'
df_existant_23['type_batiment_add'] = 'existant'
# Trouver les colonnes communes entre les deux DataFrames
colonnes_communes = df_neuf_2023.columns.intersection(df_existant_23.columns)
# Concaténer uniquement les colonnes communes
df_final_combined_23 = pd.concat([df_neuf_2023[colonnes_communes], df_existant_23[colonnes_communes]], ignore_index=True)

# Garder la dernière occurrence pour chaque 'Identifiant__BAN'
df_sans_doublons_23 = df_final_combined_23.drop_duplicates(subset=['Identifiant__BAN'], keep='last')

colonnes_a_supprimer = missing_values[missing_values > seuil].index
df_sans_doublons_23 = df_sans_doublons_23.drop(columns=colonnes_a_supprimer)

# Imputer avec la médiane pour les colonnes numériques
num_cols = df_sans_doublons_23.select_dtypes(include=['number']).columns
for col in num_cols :
    df_sans_doublons_23[col].fillna(df_sans_doublons_23[col].median(), inplace=True)
# Imputer avec la valeur la plus fréquente pour les colonnes catégorielles
cat_cols = df_sans_doublons_23.select_dtypes(include=['object', 'category']).columns

for col in cat_cols:
    df_sans_doublons_23[col].fillna(df_sans_doublons_23[col].mode()[0], inplace=True)

# Vérification des valeurs manquantes restantes
missing_values_restantes = df_sans_doublons_23.isnull().mean() * 100
# print(missing_values_restantes[missing_values_restantes > 0])
df_sans_doublons_23 = df_sans_doublons_23.drop(columns=['_score'])

# Calcul des quartiles (Q1 et Q3) et de l'IQR (Interquartile Range)
Q1 = df_sans_doublons_23['Conso_5_usages_é_finale'].quantile(0.25)
Q3 = df_sans_doublons_23['Conso_5_usages_é_finale'].quantile(0.75)
IQR = Q3 - Q1

# Définir les seuils pour identifier les outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtrer les valeurs aberrantes (outliers)
outliers = df_sans_doublons_23[(df_combined_22['Conso_5_usages_é_finale'] < lower_bound) | (df_sans_doublons_23['Conso_5_usages_é_finale'] > upper_bound)]

# Remplacer les valeurs en dehors des limites par Q1 ou Q3
df_sans_doublons_23['Conso_5_usages_é_finale'] = df_sans_doublons_23['Conso_5_usages_é_finale'].apply(
    lambda x: Q1 if x < lower_bound else (Q3 if x > upper_bound else x)
)

# # Vérifier le résultat
# print(df_sans_doublons_23['Conso_5_usages_é_finale'].describe())

# Supprimer les doublons en gardant uniquement la première occurrence pour chaque combinaison 'id' et 'annee'
df_combined_enedis_clean_23 = Enedis_23.drop_duplicates(subset=['id'], keep='first')


# Calculer le mode de 'type_de_voie'
mode_type_de_voie = df_combined_enedis_clean_23['type_de_voie'].mode()[0]
df_combined_enedis_clean_23['indice_de_repetition'] = df_combined_enedis_clean_23['indice_de_repetition'].fillna('')
df_combined_enedis_clean_23['type_de_voie'] = df_combined_enedis_clean_23['type_de_voie'].fillna(mode_type_de_voie)
df_combined_enedis_clean_23['log_conso'] = np.log1p(df_combined_enedis_clean_23['consommation_annuelle_moyenne_par_site_de_l_adresse_mwh'])  # log(1 + x) pour éviter log(0)

df_combined_enedis_clean_23 = df_combined_enedis_clean_23[colonnes_interet]
df_combined_23=df_sans_doublons_23.copy()
# Renommer la colonne 'Identifiant__BAN' en 'id' dans df_combined_2
df_combined_23.rename(columns={'Identifiant__BAN': 'id'}, inplace=True)
# Effectuer la jointure droite (right join) sur la colonne 'id'
df_joint_23= pd.merge(df_combined_23, df_combined_enedis_clean_23, on='id', how='inner')
df_joint_23['consommation_estimee_dpe_mwh'] = df_joint_23['Conso_5_usages_é_finale'] / 1000  # conversion en MWh 
df_joint_23['ecart_consommation'] = df_joint_23['consommation_annuelle_moyenne_par_site_de_l_adresse_mwh'] - df_joint_23['consommation_estimee_dpe_mwh']

df_dpe_enedis_23=df_joint_23.copy()
df_dpe_enedis_22=df_joint_22.copy()

# Faire l'union des trois DataFrames
df_union = pd.concat([df_dpe_enedis_22, df_dpe_enedis_23, enedis_21_dpe], ignore_index=True)
df_union['consommation_dpe_annuelle'] = df_union['Conso_5_usages_par_m²_é_primaire'] * df_union['Surface_habitable_logement']
