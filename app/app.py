import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Load model and scaler
model = joblib.load("model/Drilling Efficiency.pkl")
scaler = joblib.load("model/scaler.pkl")




st.set_page_config(page_title="ROP Predictor", layout="centered")
st.title("🛠️ Drilling Efficiency Optimization")
st.subheader("Predict Rate of Penetration (ROP)")

# Input features
wob = st.number_input("Weight on Bit (klbf)", 5.0, 35.0, 20.0)
rpm = st.number_input("Rotary Speed (RPM)", 60.0, 180.0, 120.0)
torque = st.number_input("Torque (kNm)", 5.0, 20.0, 10.0)
spp = st.number_input("Standpipe Pressure (psi)", 1000.0, 5000.0, 2500.0)
flow = st.number_input("Flow Rate (gpm)", 100.0, 800.0, 400.0)
mud_weight = st.number_input("Mud Weight (ppg)", 8.0, 15.0, 10.0)
formation = st.selectbox("Formation", ["Shale", "Sandstone", "Limestone"])

# Encode formation
formation_map = {"Shale": 0, "Sandstone": 1, "Limestone": 2}
formation_val = formation_map[formation]

# Feature engineering
wob_per_rpm = wob / rpm if rpm != 0 else 0
hydraulic_eff = (flow * spp) / 1714  # Simplified hydraulic efficiency

# Feature order should match training
input_df = pd.DataFrame([[
    wob, rpm, torque, spp, flow, mud_weight,
    formation_val, wob_per_rpm, hydraulic_eff
]], columns=[
    'WOB_klbf', 'RPM', 'Torque_kNm', 'SPP_psi',
    'Flow_Rate_gpm', 'Mud_Weight_ppg', 'Formation_Encoded',
    'WOB_per_RPM', 'Hydraulic_Efficiency'
])

# Apply scaling
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict ROP"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"Predicted ROP: {prediction:.2f} ft/hr")
