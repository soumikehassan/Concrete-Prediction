import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="Concrete Property Prediction", layout="centered")

# --- Helper function to load model safely ---
def load_model(path):
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    else:
        st.error(f"Model file not found: {path}")
        return None

# --- Load models ---
model_sts = load_model("xgb_STS.pkl")
model_cs = load_model("Catboost_CS.pkl")
model_slump = load_model("CatBoost_Slump.pkl")

# --- App title ---
st.title("🧱 Concrete Property Predictor")

# --- Sidebar model selector ---
model_choice = st.sidebar.selectbox("Select prediction target:", ["STS", "CS", "Slump"])

st.subheader(f"Input features for predicting **{model_choice}**")

# --- Input fields grouped in 3 columns ---
col1, col2, col3 = st.columns(3)

with col1:
    BNHF = st.number_input("BNHF (%)", value=0.0)
    Fiber = st.number_input("Fiber (kg/m3)", value=0.0)
    Fiber_length = st.number_input("Fiber Length (mm)", value=0.0)

with col2:
    wc = st.number_input("W/C", value=0.0)
    Cement = st.number_input("Cement (kg/m3)", value=0.0)
    Fine_agg = st.number_input("Fine Aggregate (kg/m3)", value=0.0)

with col3:
    Coarse_agg = st.number_input("Coarse Aggregate (kg/m3)", value=0.0)
    Water = st.number_input("Water (kg/m3)", value=0.0)
    if model_choice in ["STS", "CS"]:
        Curing_time = st.number_input("Curing Time (days)", value=0.0)
    else:
        Curing_time = None

# --- Predict button ---
if st.button("🔍 Predict"):
    if model_choice == "STS" and model_sts:
        input_data = np.array([[BNHF, Fiber, Fiber_length, wc, Cement, Fine_agg, Coarse_agg, Water, Curing_time]])
        prediction = model_sts.predict(input_data)[0]
    elif model_choice == "CS" and model_cs:
        input_data = np.array([[BNHF, Fiber, Fiber_length, wc, Cement, Fine_agg, Coarse_agg, Water, Curing_time]])
        prediction = model_cs.predict(input_data)[0]
    elif model_choice == "Slump" and model_slump:
        input_data = np.array([[BNHF, Fiber, Fiber_length, wc, Cement, Fine_agg, Coarse_agg, Water]])
        prediction = model_slump.predict(input_data)[0]
    else:
        prediction = None

    if prediction is not None:
        st.success(f"✅ Predicted {model_choice}: **{prediction:.3f}**")
    else:
        st.error("Prediction failed. Please check the input or model.")

# --- Footer ---
# Add a horizontal rule
st.markdown("<hr>", unsafe_allow_html=True)

# Add the footer with proper styling and content
st.markdown("""
<style>
.footer {
    text-align: center;
    font-size: 16px;
    padding: 10px;
    line-height: 1.6;
    margin-top: 20px;
}
.footer a {
    color: #4CAF50; /* Nice green color */
    text-decoration: none;
    font-weight: bold;
}
.footer a:hover {
    color: #45a049; /* Slightly darker green on hover */
}
</style>

<div class="footer">
    Developed by <strong>Md Soumike Hassan</strong><br>
    📧 Contact: <a href='mailto:md.soumikehassan@gmail.com'>md.soumikehassan@gmail.com</a><br>
    🛠️ Project Repository: <a href='https://github.com/soumikehassan/Concrete-Prediction' target='_blank'>GitHub - Concrete Prediction</a><br><br>

</div>
<p style='text-align: center;'>
    <strong>Contributors</strong><br>
    Mehedi Hasan, Md Soumike Hassan, Kamrul Hasan, Fazlul Hoque Tushar, Majid Khan, Ramadhansya Putra Jaya
</p>            
""", unsafe_allow_html=True)
