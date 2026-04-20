import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate synthetic agricultural data
data = {
    'nitrogen': np.random.uniform(20, 80, n_samples),
    'phosphorus': np.random.uniform(10, 60, n_samples),
    'potassium': np.random.uniform(10, 50, n_samples),
    'ph': np.random.uniform(5.5, 8.5, n_samples),
    'organic_carbon': np.random.uniform(0.2, 2.0, n_samples),
    'moisture': np.random.uniform(10, 40, n_samples),
    'latitude': np.random.uniform(15.0, 20.0, n_samples),  # Semi-arid tropics region
    'longitude': np.random.uniform(75.0, 80.0, n_samples),
    'rainfall': np.random.uniform(400, 1200, n_samples),  # mm per year
    'temperature': np.random.uniform(20, 35, n_samples),  # Celsius
    'crop_type': np.random.choice(['Sorghum', 'Millet', 'Chickpea', 'Groundnut', 'Pigeon Pea'], n_samples)
}

df = pd.DataFrame(data)

# Create crop type encoding for yield calculation
crop_base_yield = {
    'Sorghum': 2500,
    'Millet': 1800,
    'Chickpea': 1500,
    'Groundnut': 2000,
    'Pigeon Pea': 1200
}

# Calculate yield based on soil features and crop type
df['yield_kg_per_hectare'] = df.apply(lambda row: 
    crop_base_yield[row['crop_type']] + 
    (row['nitrogen'] * 10) + 
    (row['phosphorus'] * 8) + 
    (row['potassium'] * 5) + 
    ((row['ph'] - 6.5) ** 2 * -50) +  # Optimal pH around 6.5
    (row['organic_carbon'] * 200) + 
    (row['moisture'] * 15) + 
    (row['rainfall'] * 0.5) + 
    ((row['temperature'] - 27) ** 2 * -10) +  # Optimal temp around 27
    np.random.normal(0, 200),  # Add some noise
    axis=1
)

# Ensure yield is positive
df['yield_kg_per_hectare'] = df['yield_kg_per_hectare'].clip(lower=500)

# Save to CSV
df.to_csv('crop_data.csv', index=False)
print(f"Dataset created with {n_samples} samples")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nCrop distribution:\n{df['crop_type'].value_counts()}")
