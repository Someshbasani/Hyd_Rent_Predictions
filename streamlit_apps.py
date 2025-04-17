import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and dummy column list
model = joblib.load("best_LinearRegression_model.pkl")
model_columns1 = joblib.load("model_columns.pkl")

dummy_columns = joblib.load("dummy_columns.pkl") 
model_columns = dummy_columns
#st.write("All dummy columns loaded:", dummy_columns)


st.set_page_config(page_title="Hyd_rent Predictions", layout="centered")
st.title("Hyd_Rents Prediction using LinearRegression")

st.markdown("Enter features to predict the Hyd rent:")

# ---- Feature Inputs ----
# 1. Locality selectbox
locality_list = sorted([col.replace("locality_", "") for col in dummy_columns if col.startswith("locality_")])

if locality_list:
    selected_locality = st.selectbox("Select Locality", locality_list)
else:
  selected_locality = None
  st.warning("⚠️ No locality options found in your model columns.")
# 2. Other numerical fields
floor = st.number_input("Floor", min_value=0.0, max_value=180.0)
lift = st.number_input("Lift", min_value=0.0, max_value=200.0)

# 3. Parking Type selectbox
parking_options = {"Bike": 1, "Car": 2, "Bike and Car": 3, "None": 0}
parking_choice = st.selectbox("Parking Description", list(parking_options.keys()))
parkingDesc = parking_options[parking_choice]

# 4. BHK
bhk = st.number_input("BHK", min_value=0.0, max_value=10.0)

# ---- Prediction ----
if st.button("Submit"):
    if not locality_list:
        st.error("Locality selection is missing. Cannot proceed.")
    else:
        # Create input dictionary
        input_dict = dict.fromkeys(model_columns1, 0)
    # Set the matching locality column to 1
    locality_col = f"locality_{selected_locality}"
    if locality_col in input_dict:
        input_dict[locality_col] = 1

    # Add other features
    input_dict['floor'] = floor
    input_dict['lift'] = lift
    input_dict['parkingDesc'] = parkingDesc
    input_dict['bhk'] = bhk

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=model_columns1, fill_value=0)

   # st.write("Input columns:", input_df.columns.tolist())
   # st.write("Model expects:", model_columns1)

    # Predict
    prediction = model.predict(input_df)[0]
    st.subheader("🏷️ Predicted Rent:")
    st.success(f"₹ {prediction:,.2f}")
