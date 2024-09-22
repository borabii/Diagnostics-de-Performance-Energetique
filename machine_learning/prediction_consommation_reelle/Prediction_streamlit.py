import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import matplotlib.pyplot as plt

# Load the data
file_path = './output_/Dpe_Join_Enedis.csv'
df = pd.read_csv(file_path, low_memory=False)

# Load the model, encoder, scaler, and columns used during training
best_model = joblib.load('./output_/meilleur_modele.pkl')
columns_used_for_training = joblib.load('./output_/columns_used_for_training.pkl')
ordinal_encoder = joblib.load('./output_/ordinal_encoder.pkl')
scaler = joblib.load('./output_/scaler.pkl')

# Select the address
st.sidebar.header("Sélection d'adresse")
addresses = df['Adresse_(BAN)'].unique()
selected_address = st.sidebar.selectbox("Adresse", options=addresses)
df_address = df[df['Adresse_(BAN)'] == selected_address].copy()

# Display modifiable characteristics
st.sidebar.header("Caractéristiques du logement")

# Exclude "surface" and include another quality characteristic
selected_isolation = st.sidebar.selectbox(
    "Qualité de l'isolation des murs", 
    options=df['Qualité_isolation_murs'].unique(), 
    index=list(df['Qualité_isolation_murs'].unique()).index(df_address['Qualité_isolation_murs'].iloc[0])
)
selected_energie = st.sidebar.selectbox(
    "Type d'énergie principale de chauffage", 
    options=df['Type_énergie_principale_chauffage'].unique(),
    index=list(df['Type_énergie_principale_chauffage'].unique()).index(df_address['Type_énergie_principale_chauffage'].iloc[0])
)
selected_fenetres = st.sidebar.selectbox(
    "Qualité de l'isolation des fenêtres", 
    options=df['Qualité_isolation_menuiseries'].unique(),
    index=list(df['Qualité_isolation_menuiseries'].unique()).index(df_address['Qualité_isolation_menuiseries'].iloc[0])
)
selected_zone = st.sidebar.selectbox(
    "Zone climatique", 
    options=df['Zone_climatique_'].unique(),
    index=list(df['Zone_climatique_'].unique()).index(df_address['Zone_climatique_'].iloc[0])
)

# Update selected characteristics in the DataFrame
df_address['Qualité_isolation_murs'] = selected_isolation
df_address['Type_énergie_principale_chauffage'] = selected_energie
df_address['Qualité_isolation_menuiseries'] = selected_fenetres
df_address['Zone_climatique_'] = selected_zone

# Handle categorical columns by encoding
categorical_columns = [col for col in columns_used_for_training if col in df_address.columns and df_address[col].dtype == 'object']

# Check if ordinal encoder is properly fit with the categories
if len(ordinal_encoder.categories_) != len(categorical_columns):
    st.error("Le nombre de colonnes catégorielles ne correspond pas au nombre de catégories dans l'encodeur.")
else:
    df_address_encoded = ordinal_encoder.transform(df_address[categorical_columns])

    # Handle numerical columns
    numerical_columns = [col for col in columns_used_for_training if col in df_address.columns and df_address[col].dtype != 'object']
    df_address_numerical = df_address[numerical_columns]

    # Combine encoded categorical and numerical features
    df_to_predict = pd.DataFrame(df_address_encoded, columns=categorical_columns)
    df_to_predict = pd.concat([df_to_predict, df_address_numerical.reset_index(drop=True)], axis=1)

    # Ensure the DataFrame is aligned with the columns used during training
    df_to_predict = df_to_predict.reindex(columns=columns_used_for_training, fill_value=0)

    # Vérification des colonnes manquantes
    missing_columns = set(columns_used_for_training) - set(df_to_predict.columns)
    if missing_columns:
        st.error(f"Colonnes manquantes après la préparation des données : {missing_columns}")
    else:
        # Load the scaler and apply it
        if df_to_predict.shape[1] != len(scaler.mean_):
            st.error("Le nombre de colonnes après le scaling ne correspond pas à ce qui est attendu.")
        else:
            df_to_predict_scaled = scaler.transform(df_to_predict)

            # Predict consumption
            try:
                predicted_consumption = best_model.predict(df_to_predict_scaled)[0] * 1000  # Convert to kWh
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {str(e)}")

            # Convert real consumption to kWh
            real_consumption_kwh = df_address['consommation_annuelle_moyenne_par_site_de_l_adresse_mwh'].iloc[0] * 1000

            # Display metrics
            st.subheader(f"Consommation pour l'adresse : {selected_address}")
            col1, col2 = st.columns(2)
            col1.metric(label="Consommation Réelle (kWh)", value=f"{real_consumption_kwh:.2f}")
            col2.metric(label="Consommation Prédite (kWh)", value=f"{predicted_consumption:.2f}")

            # Display current characteristics
            st.write("Caractéristiques actuelles du logement :")
            st.write(f"- Qualité de l'isolation des murs : {selected_isolation}")
            st.write(f"- Type d'énergie principale de chauffage : {selected_energie}")
            st.write(f"- Qualité de l'isolation des fenêtres : {selected_fenetres}")
            st.write(f"- Zone climatique : {selected_zone}")

            # Add a representative graph
            st.subheader("Comparaison entre la consommation réelle et prédite")
            fig, ax = plt.subplots()
            ax.bar(['Consommation Réelle', 'Consommation Prédite'], [real_consumption_kwh, predicted_consumption], color=['blue', 'green'])
            ax.set_ylabel('Consommation (kWh)')
            st.pyplot(fig)
