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
# floor = st.number_input("Floor", min_value=0.0, max_value=180.0)
# lift = st.number_input("Lift", min_value=0.0, max_value=200.0)

# ---- Floor Selectbox ----
floor_options = list(range(0, 21))  # Floor values from 0 to 20
floor = st.selectbox("Floor", floor_options)

# ---- Lift Selectbox ----
lift_options = {'Yes': 1, 'No': 0}
lift_choice = st.selectbox("Lift", list(lift_options.keys()))
lift = lift_options[lift_choice]

# 3. Parking Type selectbox
parking_options = {"Bike": 1, "Car": 2, "Bike and Car": 3, "None": 0}
parking_choice = st.selectbox("Parking Description", list(parking_options.keys()))
parkingDesc = parking_options[parking_choice]

# 4. BHK
# bhk = st.selectbox("BHK", ['1Rk', '1BHK','2BHK', '3BHK'])
# bhk = st.number_input("BHK", min_value=0.0, max_value=10.0)
# ---- BHK Selectbox ----
bhk_options = {
    "1 RK": 0.5,  # or 1 if you prefer
    "1 BHK": 1,
    "2 BHK": 2,
    "3 BHK": 3,
    "4 BHK": 4,
    "5 BHK": 5
}
bhk_choice = st.selectbox("BHK Type", list(bhk_options.keys()))
bhk = bhk_options[bhk_choice]


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
   # input_df = input_df.apply(pd.to_numeric, errors='coerce')
   # st.write("Input columns:", input_df.columns.tolist())
   # st.write("Model expects:", model_columns1)

    # Predict
    prediction = model.predict(input_df)[0]
    st.subheader("🏷️ Predicted Rent:")
    st.success(f"₹ {prediction:,.2f}")
