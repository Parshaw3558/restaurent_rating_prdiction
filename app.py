import streamlit as st
import pandas as pd
import pickle
import os

# -------------------------
# Load Models & Encoders
# -------------------------
MODEL_DIR = "models"

@st.cache_resource
def load_all():
    try:
        with open(os.path.join(MODEL_DIR, "linear_regression.pkl"), "rb") as f:
            lr = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "decision_tree.pkl"), "rb") as f:
            dt = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "rb") as f:
            rf = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "rb") as f:
            encoders = pickle.load(f)

        return lr, dt, rf, encoders

    except Exception as e:
        st.error(f"Failed to load model files: {e}")
        return None, None, None, None


# Load Models
lr_model, dt_model, rf_model, label_encoders = load_all()

st.title("🍽 Restaurant Rating Prediction App")
st.write("Enter details to predict restaurant rating using trained ML models.")

if not all([lr_model, dt_model, rf_model, label_encoders]):
    st.stop()

# -------------------------
# Input fields – dynamic from encoders
# -------------------------
price_col = list(label_encoders.keys())[0]
city_col = list(label_encoders.keys())[1]
cuisine_col = list(label_encoders.keys())[2]

price_options = label_encoders[price_col].classes_
city_options = label_encoders[city_col].classes_
cuisine_options = label_encoders[cuisine_col].classes_

price = st.selectbox("Select Price Category", price_options)
city = st.selectbox("Select City", city_options)
cuisine = st.selectbox("Select Cuisine", cuisine_options)

model_choice = st.selectbox(
    "Choose a Model",
    ["Linear Regression", "Decision Tree", "Random Forest"]
)

# -------------------------
# Prediction Logic
# -------------------------
if st.button("Predict Rating ⭐"):
    try:
        p = label_encoders[price_col].transform([price])[0]
        c = label_encoders[city_col].transform([city])[0]
        cu = label_encoders[cuisine_col].transform([cuisine])[0]

        input_df = pd.DataFrame([[p, c, cu]], columns=[price_col, city_col, cuisine_col])

        if model_choice == "Linear Regression":
            pred = lr_model.predict(input_df)[0]
        elif model_choice == "Decision Tree":
            pred = dt_model.predict(input_df)[0]
        else:
            pred = rf_model.predict(input_df)[0]

        st.success(f"⭐ **Predicted Rating: {round(pred, 2)}**")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
