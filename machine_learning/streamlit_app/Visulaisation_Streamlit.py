import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Charger les données
# current_dir = os.getcwd()
# file_path = os.path.join(current_dir, 'output_', 'Dpe_Join_Enedis.csv')


# Obtenir le chemin du répertoire courant
current_dir = os.getcwd()
st.write(f"Current working directory: {current_dir}")
# file_path = os.path.join(current_dir, 'machine_learning/streamlit_app/output_', 'meilleur_modele.pkl')

# Lister le contenu du répertoire courant
st.write("Files and directories in the current directory:")
files = os.listdir(current_dir)
for file in files:
    st.write(file)

file_path = os.path.join(current_dir, 'machine_learning/streamlit_app/output_', 'meilleur_modele.pkl')
df = pd.read_csv(file_path, low_memory=False)
st.dataframe(df.head())
# # Traitement des données
# df['consommation_annuelle_moyenne_par_site_de_l_adresse_mwh'] = pd.to_numeric(df['consommation_annuelle_moyenne_par_site_de_l_adresse_mwh'], errors='coerce')
# df['Surface_habitable_logement'] = pd.to_numeric(df['Surface_habitable_logement'], errors='coerce')
# df['Conso_5_usages_par_m²_é_primaire'] = pd.to_numeric(df['Conso_5_usages_par_m²_é_primaire'], errors='coerce')
# df['consommation_reelle_kwh'] = df['consommation_annuelle_moyenne_par_site_de_l_adresse_mwh'] * 1000
# df['consommation_dpe_annuelle'] = df['Conso_5_usages_par_m²_é_primaire'] * df['Surface_habitable_logement']
# df['consommation_reelle_par_m2'] = df['consommation_reelle_kwh'] / df['Surface_habitable_logement']

# # Charger le modèle et les colonnes utilisées lors de l'entraînement

# file_path = os.path.join(current_dir, 'output_', 'meilleur_modele.pkl')
# best_model = joblib.load(file_path)
# columns_used_for_training = joblib.load('./output_/columns_used_for_training.pkl')

# # Prétraitement
# cat_cols = ['Type_installation_ECS_(général)', 'Qualité_isolation_menuiseries', 'Qualité_isolation_murs',
#             'Modèle_DPE', 'Indicateur_confort_été', 'Type_énergie_n°1',
#             'Date_fin_validité_DPE', 'Type_bâtiment', 'Zone_climatique_',
#             'Type_installation_chauffage', 'Type_énergie_principale_chauffage',
#             'Qualité_isolation_enveloppe', 'Etiquette_GES', 'Etiquette_DPE',
#             'Qualité_isolation_plancher_bas', 'Qualité_isolation_plancher_haut_comble_aménagé',
#             'Besoin_refroidissement', 'Besoin_ECS']

# columns_to_exclude = [
#     'consommation_annuelle_totale_de_l_adresse_mwh',
#     'consommation_annuelle_moyenne_par_site_de_l_adresse_mwh',
#     'consommation_annuelle_moyenne_de_la_commune_mwh',
#     'nombre_de_logements', 'consommation_estimee_dpe_mwh', 'consommation_dpe_annuelle',
# ]

# # Encoder et normaliser
# file_path = os.path.join(current_dir, 'output_', 'ordinal_encoder.pkl')
# if os.path.exists(file_path):
#     ordinal_encoder = joblib.load(file_path)
# else:
#     df_cat = df[cat_cols].astype(str)
#     ordinal_encoder = OrdinalEncoder()
#     encoded_cat = ordinal_encoder.fit_transform(df_cat)
#     joblib.dump(ordinal_encoder, 'ordinal_encoder.pkl')

# file_path = os.path.join(current_dir, 'output_', 'scaler.pkl')

# if os.path.exists(file_path):
#     scaler = joblib.load(file_path)
# else:
#     features_num = df.select_dtypes(include=[float]).drop(columns=columns_to_exclude)
#     scaler = StandardScaler()
#     scaler.fit(features_num)
#     joblib.dump(scaler, file_path)

# df_cat = df[cat_cols].astype(str)
# encoded_cat = ordinal_encoder.transform(df_cat)
# encoded_cat_df = pd.DataFrame(encoded_cat, columns=cat_cols)
# features_num = df.select_dtypes(include=[float]).drop(columns=columns_to_exclude)
# features = pd.concat([features_num, encoded_cat_df], axis=1)

# extra_features = set(features.columns) - set(columns_used_for_training)
# features = features.drop(columns=list(extra_features))
# missing_features = set(columns_used_for_training) - set(features.columns)
# for col in missing_features:
#     features[col] = 0

# features = features[columns_used_for_training]
# features_scaled = scaler.transform(features)

# df['Consommation_Predite'] = best_model.predict(features_scaled)
# df['Consommation_Predite'] = pd.to_numeric(df['Consommation_Predite'], errors='coerce') * 1000
# df['ecart_conso'] = df['consommation_reelle_kwh'] - df['consommation_dpe_annuelle']

# # Filtrer les données
# df_filtered = df[(df['ecart_conso'] >= -800) & (df['ecart_conso'] <= 1000)]

# st.sidebar.header("Filtres")
# selected_year = st.sidebar.selectbox("Sélectionnez l'année", options=[2021, 2022, 2023])
# filtered_by_year = df_filtered[df_filtered['annee'] == selected_year]
# addresses = filtered_by_year['Adresse_(BAN)'].unique()
# selected_address = st.sidebar.selectbox("Adresse", options=addresses)
# filtered_by_address = filtered_by_year[filtered_by_year['Adresse_(BAN)'] == selected_address]

# if not filtered_by_address.empty:
#     real_consumption = filtered_by_address['consommation_reelle_kwh'].iloc[0]
#     dpe_consumption = filtered_by_address['consommation_dpe_annuelle'].iloc[0]
#     predicted_consumption = filtered_by_address['Consommation_Predite'].iloc[0]
#     surface_area = filtered_by_address['Surface_habitable_logement'].iloc[0]

#     # Define consumption_diff and percentage_diff
#     consumption_diff = real_consumption - dpe_consumption
#     # percentage_diff = abs(consumption_diff) / dpe_consumption * 100

#     # Affichage des métriques de consommation
#     st.subheader(f"Consommation pour l'adresse : {selected_address}")
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric(label="Consommation Réelle (kWh)", value=f"{real_consumption:.2f}")
#     col2.metric(label="Consommation Estimée DPE (kWh)", value=f"{dpe_consumption:.2f}")
#     col3.metric(label="Consommation Prédite (kWh)", value=f"{predicted_consumption:.2f}")
#     col4.metric(label="Écart (kWh)", value=f"{abs(consumption_diff):.2f}", delta=f"{consumption_diff:.2f}")

#     # Informations sur le logement
#     # st.subheader("Informations du logement")
#     # col5, col6, col7 = st.columns(3)
#     # col5.metric(label="Surface habitable (m²)", value=f"**{surface_area:.2f} m²**")
#     # col6.metric(label="Type de bâtiment", value=f"{filtered_by_address['Type_bâtiment'].iloc[0]}")
#     # col7.metric(label="Classe DPE", value=f"{filtered_by_address['Etiquette_DPE'].iloc[0]}")

#     # Calcul des classes DPE réelles et estimées
#     dpe_intervals = [
#         (0, 50, "A"),
#         (51, 90, "B"),
#         (91, 150, "C"),
#         (151, 230, "D"),
#         (231, 330, "E"),
#         (331, 450, "F"),
#         (451, float('inf'), "G")
#     ]

#     def determine_dpe_class(consumption_per_m2):
#         for lower_bound, upper_bound, dpe_class in dpe_intervals:
#             if lower_bound <= consumption_per_m2 <= upper_bound:
#                 return dpe_class
#         return "Inconnu"

#     consommation_dpe_m2 = filtered_by_address['Conso_5_usages_par_m²_é_primaire'].iloc[0]
#     consommation_reelle_m2 = filtered_by_address['consommation_reelle_par_m2'].mean()
#     classe_dpe_reelle = determine_dpe_class(consommation_reelle_m2)
#     dpe_class = determine_dpe_class(dpe_consumption / surface_area)

#     # Affichage de la table des informations supplémentaires du logement
#     table_html = f"""
#     <table style="width:100%; border-collapse: collapse; font-family: Arial, sans-serif;">
#       <tr style="background-color: #f2f2f2;">
#         <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Propriété</th>
#         <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Valeur</th>
#       </tr>
#       <tr>
#         <td style="padding: 8px; border: 1px solid #ddd;">Type de Chauffage</td>
#         <td style="padding: 8px; border: 1px solid #ddd;">{filtered_by_address['Type_installation_chauffage'].iloc[0]}</td>
#       </tr>
#       <tr>
#         <td style="padding: 8px; border: 1px solid #ddd;">Énergie Principale Chauffage</td>
#         <td style="padding: 8px; border: 1px solid #ddd;">{filtered_by_address['Type_énergie_principale_chauffage'].iloc[0]}</td>
#       </tr>
#       <tr>
#         <td style="padding: 8px; border: 1px solid #ddd;">Qualité Isolation Murs</td>
#         <td style="padding: 8px; border: 1px solid #ddd;">{filtered_by_address['Qualité_isolation_murs'].iloc[0]}</td>
#       </tr>
#       <tr>
#         <td style="padding: 8px; border: 1px solid #ddd;">Qualité Isolation Menuiseries</td>
#         <td style="padding: 8px; border: 1px solid #ddd;">{filtered_by_address['Qualité_isolation_menuiseries'].iloc[0]}</td>
#       </tr>
#       <tr>
#         <td style="padding: 8px; border: 1px solid #ddd;">Zone Climatique</td>
#         <td style="padding: 8px; border: 1px solid #ddd;">{filtered_by_address['Zone_climatique_'].iloc[0]}</td>
#       </tr>
#       <tr>
#         <td style="padding: 8px; border: 1px solid #ddd;">Indicateur Confort Été</td>
#         <td style="padding: 8px; border: 1px solid #ddd;">{filtered_by_address['Indicateur_confort_été'].iloc[0]}</td>
#       </tr>
#       <tr>
#         <td style="padding: 8px; border: 1px solid #ddd; color:red;">Consommation DPE par m²</td>
#         <td style="padding: 8px; border: 1px solid #ddd; color:red;">{consommation_dpe_m2:.2f} kWh/m²</td>
#       </tr>
#       <tr>
#         <td style="padding: 8px; border: 1px solid #ddd; color:green;">Consommation réelle par m²</td>
#         <td style="padding: 8px; border: 1px solid #ddd; color:green;">{consommation_reelle_m2:.2f} kWh/m²</td>
#       </tr>
#       <tr>
#         <td style="padding: 8px; border: 1px solid #ddd; font-weight:bold;">Surface habitable (m²)</td>
#         <td style="padding: 8px; border: 1px solid #ddd; font-weight:bold;">{surface_area:.2f} m²</td>
#       </tr>
#       <tr>
#         <td style="padding: 8px; border: 1px solid #ddd;">Classe DPE réelle</td>
#         <td style="padding: 8px; border: 1px solid #ddd;">{classe_dpe_reelle}</td>
#       </tr>
#       <tr>
#         <td style="padding: 8px; border: 1px solid #ddd;">Classe DPE estimée</td>
#         <td style="padding: 8px; border: 1px solid #ddd;">{dpe_class}</td>
#       </tr>
#        <tr>
#         <td style="padding: 8px; border: 1px solid #ddd;">Type bâtiment</td>
#         <td style="padding: 8px; border: 1px solid #ddd;">{filtered_by_address['type_batiment_add'].iloc[0]}</td>
#       </tr>
#     </table>
#     """

#     st.markdown(table_html, unsafe_allow_html=True)

#     # Visualisation des consommations avec plotly
#     colors = ['#58C153', '#A8E06E', '#88C057']  # Ajoutez une troisième couleur si nécessaire

#     # Visualisation des consommations avec plotly
#     fig = px.bar(
#         x=['Consommation Réelle', 'Consommation Estimée DPE', 'Consommation Prédite'],
#         y=[real_consumption, dpe_consumption, predicted_consumption],
#         labels={'x': 'Type de Consommation', 'y': 'Consommation (kWh)'},
#         title=f"Comparaison des Consommations pour {selected_address}",
#         color_discrete_sequence=colors  # Applique les couleurs personnalisées
#     )

#     st.plotly_chart(fig)
  
#     # Calcul et affichage du passage de classe DPE
#     st.subheader(f"Passage de la classe {classe_dpe_reelle} à une autre classe DPE")
#     target_class = st.selectbox("Sélectionnez la classe DPE cible", options=['A', 'B', 'C', 'D', 'E', 'F', 'G'])

#     if target_class != classe_dpe_reelle:
#         # Moyennes des intervalles des classes DPE
#         dpe_class_consumption_avg = {
#             'A': 25,  # Moyenne de l'intervalle 0-50
#             'B': 70.5,  # Moyenne de l'intervalle 51-90
#             'C': 120.5,  # Moyenne de l'intervalle 91-150
#             'D': 190.5,  # Moyenne de l'intervalle 151-230
#             'E': 280.5,  # Moyenne de l'intervalle 231-330
#             'F': 390.5,  # Moyenne de l'intervalle 331-450
#             'G': 500  # Moyenne de l'intervalle 451+
#         }

#         # Calcul de la consommation avant et après passage
#         consumption_before = consommation_reelle_m2
#         consumption_after = dpe_class_consumption_avg[target_class]

#         # Calcul de la différence de consommation
#         total_consumption_difference = abs(consumption_after - consumption_before) * surface_area
#         consumption_after_tot = consumption_after * surface_area

#         # Calcul du gain annuel en coût
#         st.sidebar.subheader("Tarification")
#         tariff_option = st.sidebar.radio(
#             "Choisir le tarif d'électricité :",
#             ('Base', 'Heures Pleines', 'Heures Creuses')
#         )
#         tariff_euro_kwh = 0.2516 if tariff_option == 'Base' else 0.27 if tariff_option == 'Heures Pleines' else 0.2068
#         cost_before = real_consumption * tariff_euro_kwh
#         cost_after = consumption_after_tot * tariff_euro_kwh
#         cost_diff = cost_after - cost_before

#         # Affichage des résultats avec des symboles et des couleurs
#         if consumption_before > consumption_after:
#             result_text = "Réduction de la consommation"
#             symbol = "↓"
#             color = "green"
#             cost_text = "Réduction du coût énergétique"
#         else:
#             result_text = "Augmentation de la consommation"
#             symbol = "↑"
#             color = "red"
#             cost_text = "Augmentation du coût énergétique"

#         st.subheader(f"📊 **Passage de la classe {classe_dpe_reelle} à {target_class}**")

#         # Metrics display
#         col1, col2, col3 = st.columns(3)
#         col1.metric(label="Consommation actuelle par m² (kWh/m²)", value=f"{consumption_before:.2f}")
#         col2.metric(label="Consommation cible par m² (kWh/m²)", value=f"{consumption_after:.2f}")
#         col3.metric(label="Écart par m² (kWh/m²)", value=f"{total_consumption_difference/surface_area:.2f}")

#         col4, col5, col6 = st.columns(3)
#         col4.metric(label="Consommation totale actuelle (kWh)", value=f"{real_consumption:.2f}")
#         col5.metric(label="Consommation totale cible (kWh)", value=f"{consumption_after_tot:.2f}")
#         col6.metric(label="Écart total (kWh)", value=f"{total_consumption_difference:.2f}")

#         col7, col8, col9 = st.columns(3)
#         col7.metric(label="Coût actuel (€)", value=f"{cost_before:.2f} €")
#         col8.metric(label="Coût cible (€)", value=f"{cost_after:.2f} €")
#         col9.metric(label="Écart de coût (€)", value=f"{cost_diff:.2f} €")

#         st.markdown(f"""
#         <div style="background-color:#f9f9f9;padding:10px;border-radius:5px;">
#             <p style="color:{color};font-size:20px;">
#                 {symbol} <strong>{result_text}</strong> : 
#                 <strong>{total_consumption_difference:.2f} kWh</strong> 
#             </p>
#             <p style="font-size:18px;">• Nouvelle consommation totale après passage : <strong>{consumption_after_tot:.2f} kWh</strong></p>
#         </div>
#         """, unsafe_allow_html=True)
