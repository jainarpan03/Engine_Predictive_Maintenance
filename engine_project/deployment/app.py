import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="jarpan03/engine-predictive-maintenance-model", filename="best_engine_maintenance_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Predictive Maintenance Prediction
st.title("Engine Predictive Maintenance Prediction")
st.write("Fill the engine details below to predict if they'll need a maintenance")

# Collect user input
Engine_RPM = st.number_input("Engine_RPM", min_value=1, max_value=10000, value=100,step=1)
Lub_Oil_Pressure = st.number_input("Lub_Oil_Pressure", min_value=0.0, value=1.0,step=0.000001,format="%.6f")
Fuel_Pressure = st.number_input("Fuel_Pressure", min_value=0.0, value=1.0,step=0.000001,format="%.6f")
Coolant_Pressure = st.number_input("Coolant_Pressure", min_value=0.0, value=1.0,step=0.000001,format="%.6f")
Lub_Oil_Temperature = st.number_input("Lub_Oil_Temperature", min_value=0.0, value=1.0,step=0.000001,format="%.6f")
Coolant_Temperature = st.number_input("Coolant_Temperature", min_value=0.0, value=1.0,step=0.000001,format="%.6f")

# ----------------------------
# Prepare input data
# ----------------------------
input_data = pd.DataFrame([{
    'Engine rpm': Engine_RPM,
    'Lub oil pressure': Lub_Oil_Pressure,
    'Fuel pressure': Fuel_Pressure,
    'Coolant pressure': Coolant_Pressure,
    'lub oil temp': Lub_Oil_Temperature,
    'Coolant temp': Coolant_Temperature
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prob = model.predict_proba(input_data)[0,1]
    pred = int(prob >= classification_threshold)
    result = "will need the engine maintenance" if pred == 1 else "maintenance not needed"
    st.write(f"Prediction: Engine {result}")
