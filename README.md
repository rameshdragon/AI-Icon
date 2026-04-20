# ICRISAT Crop Yield Prediction

This is a demo version of the crop yield prediction system I built during my internship at ICRISAT. The original was deployed on-premises for research scientists. This replica uses the same approach but with synthetic data since I can't share their actual dataset.

## What it does

Predicts crop yield based on soil properties, location, and weather. Uses two ML models together:
- Linear Regression (70% weight)
- Random Forest (30% weight)

## How to run

1. Extract the zip file
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

That's it. The models are already trained and the dataset is included.

## What's inside

- `app.py` - Streamlit interface with sliders for soil inputs
- `model.py` - ML training and prediction code
- `crop_data.csv` - 1000 sample synthetic dataset
- Pre-trained model files (`.pkl`)

## Tech used

Python, Streamlit, scikit-learn, pandas, NumPy, Plotly

The real version at ICRISAT used actual agricultural data from their research fields and was accessible via local WiFi to 3-4 scientists.

