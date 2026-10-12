import sys
import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.predict import load_trained_model, make_prediction

# Set page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# Load model (cached so it doesn't reload on every interaction)
@st.cache_resource
def get_model():
    return load_trained_model(BASE_DIR / "models" / "best_model.joblib")

try:
    model = get_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please run `python src/train.py` first.")
    st.stop()

st.title("🏠 House Price Prediction")
st.markdown("""
This app predicts the **Median House Value** for a block group in California based on various features.
Simply adjust the features in the sidebar and click **Predict Price**.
""")

# Sidebar for inputs
st.sidebar.header("Input Features")

def user_input_features():
    MedInc = st.sidebar.slider("Median Income (in 10,000s)", 0.0, 15.0, 3.5)
    HouseAge = st.sidebar.slider("House Age (Years)", 1.0, 52.0, 20.0)
    AveRooms = st.sidebar.slider("Average Rooms", 1.0, 10.0, 5.0)
    AveBedrms = st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0)
    Population = st.sidebar.slider("Population", 10.0, 10000.0, 1500.0)
    AveOccup = st.sidebar.slider("Average Occupancy", 1.0, 10.0, 3.0)
    Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 35.0)
    Longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -120.0)
    
    data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }
    return data

input_data = user_input_features()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Your Input")
    st.write(pd.DataFrame([input_data]))
    
    if st.button("Predict Price", type="primary"):
        with st.spinner("Calculating..."):
            pred = make_prediction(model, input_data)
            # The target in California housing dataset is expressed in hundreds of thousands of dollars ($100,000)
            final_price = pred * 100000
            st.success(f"### Predicted Price: {final_price:,.2f}")

with col2:
    st.subheader("Feature Importance")
    plot_path = BASE_DIR / "plots" / "feature_importance.png"
    if plot_path.exists():
        image = Image.open(plot_path)
        st.image(image, caption="Feature Importances from Best Model", use_container_width=True)
    else:
        st.info("Feature importance plot not found. Run `python src/evaluate.py` to generate it.")
