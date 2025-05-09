import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(page_title="Honey Price Prediction", page_icon=":honey_pot:", layout="wide")

# Define a professional color palette
primary_color = "#F5F5DC"  # Beige/Off-white background
secondary_color = "#A0522D"  # Sienna/Brown accents
accent_color = "#D2691E"  # Chocolate/Orange-brown for emphasis
text_color = "#333333"  # Dark gray for readability
header_bg_color = "#EEE8AA" # Pale goldenrod for header

# Custom CSS for a professional UI
# Custom CSS for a professional UI
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {primary_color};
        color: {text_color};
    }}
    .header-container {{
        background-color: {header_bg_color};
        color: {secondary_color};
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
    }}
    .header-container h1 {{
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }}
    .header-container p {{
        font-size: 1.2rem;
        color: {text_color};
    }}
    .input-section {{
        background-color: white;
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        border: 1px solid #e0e0e0;
    }}
    .input-section h2 {{
        color: {secondary_color};
        margin-top: 0;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 0.75rem;
        font-size: 1.6rem;
    }}
    .stNumberInput > label {{
        color: {text_color};
    }}
    .stSelectbox > label {{
        color: {text_color};
    }}
    .stButton > button {{
        background-color: {accent_color};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }}
    .stButton > button:hover {{
        background-color: {secondary_color};
    }}
    .prediction-box {{
        background-color: {secondary_color};
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 2rem;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }}
    .st-expander {{
        background-color: white;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        border: 1px solid #e0e0e0;
        padding: 1rem;
    }}
    .st-expander-title {{
        color: {secondary_color};
        font-size: 1.4rem;
        font-weight: bold;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the model
model = joblib.load('./xgboost_model.pkl')

# Header section
st.markdown(f'<div class="header-container"><h1>🍯 Honey Price Prediction</h1><p>Enter the characteristics of your honey to get an estimated price.</p></div>', unsafe_allow_html=True)

# Input sections using expanders for better organization
with st.expander("Basic Honey Characteristics", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        CS = st.number_input('Color Score (CS) (1.0 - 10.0)', min_value=1.0, max_value=10.0, step=0.1, help="Indicates the intensity of the honey's color.")
        Density = st.number_input('Density (1.21 - 1.86)', min_value=1.21, max_value=1.86, step=0.01, help="Measures how dense the honey is.")
        WC = st.number_input('Water Content (%) (12.0 - 25.0)', min_value=12.0, max_value=25.0, step=0.1, help="The percentage of water present in the honey.")
        pH = st.number_input('pH Value (2.50 - 7.50)', min_value=2.50, max_value=7.50, step=0.01, help="The acidity or alkalinity level of the honey.")
    with col2:
        Viscosity = st.number_input('Viscosity (1500 - 10000)', min_value=1500, max_value=10000, step=10, help="Measures the thickness and stickiness of the honey.")
        Purity = st.number_input('Purity (0.01 - 1.00)', min_value=0.01, max_value=1.00, step=0.01, help="A ratio indicating how pure the honey is.")
        pollen_options = ['Blueberry', 'Alfalfa', 'Chestnut', 'Borage', 'Sunflower',
                            'Orange Blossom', 'Acacia', 'Tupelo', 'Clover', 'Wildflower',
                            'Thyme', 'Sage', 'Avocado', 'Lavender', 'Eucalyptus', 'Buckwheat',
                            'Rosemary', 'Heather', 'Manuka']
        Pollen_analysis = st.selectbox('Floral Source (Pollen Analysis)', pollen_options, help="Select the primary floral source of the honey.")

with st.expander("Sugar Levels", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        F = st.number_input('Fructose Level (20 - 50)', min_value=20.0, max_value=50.0, step=0.1, help="The amount of fructose found in the honey.")
    with col2:
        G = st.number_input('Glucose Level (20 - 45)', min_value=20.0, max_value=45.0, step=0.1, help="The amount of glucose found in the honey.")

# pH for group labeling (remains the same logic)
ph_bins = [2, 4, 6, 8]
ph_labels = ['Low', 'Medium', 'Normal']
ph_group = pd.cut([pH], bins=ph_bins, labels=ph_labels, right=False)[0]

# Purity for group labeling (remains the same logic)
purity_bins = [0.6, 0.8, 0.9, 1.0]
purity_labels = ['Low', 'Medium', 'High']
purity_group_auto = pd.cut([Purity], bins=purity_bins, labels=purity_labels, right=False)[0]

# Calculate ratios (remains the same logic)
FG = F + G
pur_G_ratio = G / Purity if Purity != 0 else 0
pur_F_ratio = F / Purity if Purity != 0 else 0
pur_WC_ratio = WC / Purity if Purity != 0 else 0
pur_FG_ratio = FG / Purity if Purity != 0 else 0
pur_CS_ratio = CS / Purity if Purity != 0 else 0
den_Vis_ratio = (Density * Viscosity) / Purity if Purity != 0 else 0
vis_FG_ratio = (F**2) + (G**2) / Viscosity if Viscosity != 0 else 0

# Convert input data to a numpy array (remains the same logic)
input_data = np.array([
    CS, Density, WC, pH, F, G, Viscosity, Purity,
    pur_G_ratio,
    pur_F_ratio,
    pur_WC_ratio,
    pur_FG_ratio,
    pur_CS_ratio,
    den_Vis_ratio,
    vis_FG_ratio,
    purity_labels.index(purity_group_auto) if pd.notna(purity_group_auto) else -1,
    pollen_options.index(Pollen_analysis),
    ph_labels.index(ph_group) if pd.notna(ph_group) else -1
]).reshape(1, -1)

# Prediction button
if st.button('Predict Price'):
    prediction = model.predict(input_data)
    st.markdown(f'<div class="prediction-box">Estimated Price: ${prediction[0]:.2f}</div>', unsafe_allow_html=True)

# Optional sidebar information (can be further styled if needed)
st.sidebar.header("Information")
st.sidebar.info("This application estimates the price of honey based on the characteristics you provide. The predictions are based on a trained machine learning model.")
st.sidebar.info("Factors influencing the price include color score, density, water content, pH value, sugar levels, viscosity, purity, and floral source.")
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.markdown("This is a simple application demonstrating price prediction using machine learning.")