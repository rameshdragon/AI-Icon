import streamlit as st
import pandas as pd
import numpy as np
from model import CropYieldPredictor
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="ICRISAT Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    h1 {
        color: #2c5f2d;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    h2 {
        color: #2c5f2d;
        margin-top: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize predictor
@st.cache_resource
def load_predictor():
    predictor = CropYieldPredictor()
    try:
        predictor.load_models()
    except FileNotFoundError:
        st.warning("Training models for the first time...")
        predictor.train()
    return predictor

predictor = load_predictor()

# Header
st.markdown("<h1>🌾 ICRISAT Crop Yield Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555; font-size: 1.2rem;'>Machine Learning-Based Agricultural Yield Forecasting for Semi-Arid Tropics</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Green_Leaf_Icon.svg/1200px-Green_Leaf_Icon.svg.png", width=100)
    st.markdown("### About")
    st.info("""
    This system uses ensemble ML models (Linear Regression + Random Forest) 
    to predict crop yield based on soil properties, location, and environmental factors.
    
    **Models:** 70% LR + 30% RF  
    **Accuracy:** ~70% R² score  
    **Features:** 12 input parameters
    """)
    
    st.markdown("### How to Use")
    st.markdown("""
    1. Select your crop type
    2. Adjust soil parameter sliders
    3. Enter location coordinates
    4. Set environmental factors
    5. Click **Predict Yield**
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📍 Crop & Location")
    
    crop_type = st.selectbox(
        "Select Crop Type",
        options=['Sorghum', 'Millet', 'Chickpea', 'Groundnut', 'Pigeon Pea'],
        help="Choose the crop you want to predict yield for"
    )
    
    col_lat, col_lon = st.columns(2)
    with col_lat:
        latitude = st.number_input(
            "Latitude (°)",
            min_value=10.0,
            max_value=25.0,
            value=17.4,
            step=0.1,
            help="Geographic latitude of your field"
        )
    with col_lon:
        longitude = st.number_input(
            "Longitude (°)",
            min_value=70.0,
            max_value=85.0,
            value=78.5,
            step=0.1,
            help="Geographic longitude of your field"
        )
    
    st.markdown("## 🌱 Soil Properties")
    
    nitrogen = st.slider(
        "Nitrogen (N) - kg/ha",
        min_value=20.0,
        max_value=80.0,
        value=50.0,
        step=1.0,
        help="Nitrogen content in soil"
    )
    
    phosphorus = st.slider(
        "Phosphorus (P) - kg/ha",
        min_value=10.0,
        max_value=60.0,
        value=35.0,
        step=1.0,
        help="Phosphorus content in soil"
    )
    
    potassium = st.slider(
        "Potassium (K) - kg/ha",
        min_value=10.0,
        max_value=50.0,
        value=30.0,
        step=1.0,
        help="Potassium content in soil"
    )
    
    ph = st.slider(
        "Soil pH",
        min_value=5.5,
        max_value=8.5,
        value=6.5,
        step=0.1,
        help="Soil acidity/alkalinity level"
    )
    
    organic_carbon = st.slider(
        "Organic Carbon (%)",
        min_value=0.2,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Organic carbon content in soil"
    )
    
    moisture = st.slider(
        "Soil Moisture (%)",
        min_value=10.0,
        max_value=40.0,
        value=25.0,
        step=1.0,
        help="Current soil moisture level"
    )

with col2:
    st.markdown("## 🌦️ Environmental Factors")
    
    rainfall = st.slider(
        "Annual Rainfall (mm)",
        min_value=400.0,
        max_value=1200.0,
        value=800.0,
        step=10.0,
        help="Expected annual rainfall in mm"
    )
    
    temperature = st.slider(
        "Average Temperature (°C)",
        min_value=20.0,
        max_value=35.0,
        value=27.0,
        step=0.5,
        help="Average growing season temperature"
    )
    
    # Predict button
    st.markdown("---")
    predict_button = st.button("🚀 Predict Crop Yield", use_container_width=True, type="primary")
    
    if predict_button:
        with st.spinner("Analyzing soil and environmental data..."):
            # Prepare input
            input_data = {
                'nitrogen': nitrogen,
                'phosphorus': phosphorus,
                'potassium': potassium,
                'ph': ph,
                'organic_carbon': organic_carbon,
                'moisture': moisture,
                'latitude': latitude,
                'longitude': longitude,
                'rainfall': rainfall,
                'temperature': temperature,
                'crop_type': crop_type
            }
            
            # Get prediction
            result = predictor.predict(input_data)
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            # Main prediction
            st.markdown(f"""
                <div class="prediction-box">
                    <div style="font-size: 1rem; opacity: 0.9;">Predicted Crop Yield</div>
                    <div style="font-size: 3rem; font-weight: bold;">{result['ensemble_prediction']:.0f}</div>
                    <div style="font-size: 1.2rem; opacity: 0.9;">kg/hectare</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Model breakdown
            st.markdown("### Model Breakdown")
            col_lr, col_rf, col_sq = st.columns(3)
            
            with col_lr:
                st.metric(
                    "Linear Regression",
                    f"{result['linear_regression']:.0f} kg/ha",
                    delta="70% weight"
                )
            
            with col_rf:
                st.metric(
                    "Random Forest",
                    f"{result['random_forest']:.0f} kg/ha",
                    delta="30% weight"
                )
            
            with col_sq:
                st.metric(
                    "Soil Quality Score",
                    f"{result['soil_quality']:.1f}/100",
                    delta="Calculated"
                )
            
            # Gauge chart for soil quality
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['soil_quality'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Soil Quality Index", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkgreen"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 33], 'color': '#ffcccc'},
                        {'range': [33, 66], 'color': '#ffffcc'},
                        {'range': [66, 100], 'color': '#ccffcc'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "#2c5f2d", 'family': "Arial"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            recommendations = []
            
            if result['soil_quality'] < 50:
                recommendations.append("⚠️ Soil quality is below optimal. Consider adding organic matter and balanced fertilizers.")
            elif result['soil_quality'] < 70:
                recommendations.append("✅ Soil quality is moderate. Minor improvements in nutrient balance recommended.")
            else:
                recommendations.append("✅ Excellent soil quality! Maintain current soil management practices.")
            
            if ph < 6.0:
                recommendations.append("📊 Soil is acidic. Consider lime application to raise pH.")
            elif ph > 7.5:
                recommendations.append("📊 Soil is alkaline. Consider sulfur application to lower pH.")
            
            if nitrogen < 40:
                recommendations.append("🌱 Low nitrogen levels. Consider nitrogen-rich fertilizers or legume rotation.")
            
            if moisture < 20:
                recommendations.append("💧 Low soil moisture. Ensure adequate irrigation during growing season.")
            
            if rainfall < 600:
                recommendations.append("🌧️ Low rainfall area. Drought-resistant varieties recommended.")
            
            for rec in recommendations:
                st.info(rec)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>ICRISAT Crop Yield Prediction System</strong></p>
        <p>Developed using Machine Learning | Linear Regression + Random Forest Ensemble</p>
        <p style='font-size: 0.9rem;'>Note: This is a demonstration system. Predictions are based on synthetic data for educational purposes.</p>
    </div>
""", unsafe_allow_html=True)
