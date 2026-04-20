import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

class CropYieldPredictor:
    def __init__(self):
        self.lr_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        
    def create_soil_quality_feature(self, df):
        """Create soil quality feature from soil parameters"""
        # Normalize each component to 0-1 scale then average
        soil_quality = (
            (df['nitrogen'] - df['nitrogen'].min()) / (df['nitrogen'].max() - df['nitrogen'].min()) * 0.25 +
            (df['phosphorus'] - df['phosphorus'].min()) / (df['phosphorus'].max() - df['phosphorus'].min()) * 0.25 +
            (df['potassium'] - df['potassium'].min()) / (df['potassium'].max() - df['potassium'].min()) * 0.20 +
            (df['organic_carbon'] - df['organic_carbon'].min()) / (df['organic_carbon'].max() - df['organic_carbon'].min()) * 0.15 +
            (df['moisture'] - df['moisture'].min()) / (df['moisture'].max() - df['moisture'].min()) * 0.15
        ) * 100  # Scale to 0-100
        
        return soil_quality
    
    def prepare_features(self, df):
        """Prepare features including soil quality and encoding"""
        df = df.copy()
        
        # Create soil quality feature
        df['soil_quality'] = self.create_soil_quality_feature(df)
        
        # Encode crop type
        df['crop_encoded'] = self.label_encoder.fit_transform(df['crop_type'])
        
        # Select features for model
        feature_cols = ['nitrogen', 'phosphorus', 'potassium', 'ph', 'organic_carbon', 
                       'moisture', 'latitude', 'longitude', 'rainfall', 'temperature', 
                       'soil_quality', 'crop_encoded']
        
        X = df[feature_cols]
        y = df['yield_kg_per_hectare'] if 'yield_kg_per_hectare' in df.columns else None
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train(self, data_path='crop_data.csv'):
        """Train both models with 80-20 split"""
        print("Loading data...")
        df = pd.read_csv(data_path)
        
        print("Preparing features...")
        X, y = self.prepare_features(df)
        
        # 80-20 train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training Linear Regression...")
        self.lr_model.fit(X_train_scaled, y_train)
        lr_train_pred = self.lr_model.predict(X_train_scaled)
        lr_test_pred = self.lr_model.predict(X_test_scaled)
        
        print("\nLinear Regression Performance:")
        print(f"Train R²: {r2_score(y_train, lr_train_pred):.4f}")
        print(f"Test R²: {r2_score(y_test, lr_test_pred):.4f}")
        print(f"Train MSE: {mean_squared_error(y_train, lr_train_pred):.2f}")
        print(f"Test MSE: {mean_squared_error(y_test, lr_test_pred):.2f}")
        
        print("\nTraining Random Forest...")
        self.rf_model.fit(X_train_scaled, y_train)
        rf_train_pred = self.rf_model.predict(X_train_scaled)
        rf_test_pred = self.rf_model.predict(X_test_scaled)
        
        print("\nRandom Forest Performance:")
        print(f"Train R²: {r2_score(y_train, rf_train_pred):.4f}")
        print(f"Test R²: {r2_score(y_test, rf_test_pred):.4f}")
        print(f"Train MSE: {mean_squared_error(y_train, rf_train_pred):.2f}")
        print(f"Test MSE: {mean_squared_error(y_test, rf_test_pred):.2f}")
        
        # Ensemble prediction (70% LR, 30% RF)
        ensemble_train_pred = 0.7 * lr_train_pred + 0.3 * rf_train_pred
        ensemble_test_pred = 0.7 * lr_test_pred + 0.3 * rf_test_pred
        
        print("\nEnsemble Model Performance (70% LR + 30% RF):")
        print(f"Train R²: {r2_score(y_train, ensemble_train_pred):.4f}")
        print(f"Test R²: {r2_score(y_test, ensemble_test_pred):.4f}")
        print(f"Train MSE: {mean_squared_error(y_train, ensemble_train_pred):.2f}")
        print(f"Test MSE: {mean_squared_error(y_test, ensemble_test_pred):.2f}")
        
        self.is_trained = True
        
        # Save models
        self.save_models()
        
        return {
            'lr_r2': r2_score(y_test, lr_test_pred),
            'rf_r2': r2_score(y_test, rf_test_pred),
            'ensemble_r2': r2_score(y_test, ensemble_test_pred),
            'lr_mse': mean_squared_error(y_test, lr_test_pred),
            'rf_mse': mean_squared_error(y_test, rf_test_pred),
            'ensemble_mse': mean_squared_error(y_test, ensemble_test_pred)
        }
    
    def predict(self, input_data):
        """Make ensemble prediction on new data"""
        if not self.is_trained:
            self.load_models()
        
        # Prepare input
        df_input = pd.DataFrame([input_data])
        df_input['crop_type'] = input_data['crop_type']
        
        X, _ = self.prepare_features(df_input)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from both models
        lr_pred = self.lr_model.predict(X_scaled)[0]
        rf_pred = self.rf_model.predict(X_scaled)[0]
        
        # Ensemble: 70% LR + 30% RF
        ensemble_pred = 0.7 * lr_pred + 0.3 * rf_pred
        
        return {
            'linear_regression': lr_pred,
            'random_forest': rf_pred,
            'ensemble_prediction': ensemble_pred,
            'soil_quality': self.create_soil_quality_feature(df_input).values[0]
        }
    
    def save_models(self):
        """Save trained models and scaler"""
        joblib.dump(self.lr_model, 'lr_model.pkl')
        joblib.dump(self.rf_model, 'rf_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.label_encoder, 'label_encoder.pkl')
        print("\nModels saved successfully!")
    
    def load_models(self):
        """Load trained models and scaler"""
        if os.path.exists('lr_model.pkl'):
            self.lr_model = joblib.load('lr_model.pkl')
            self.rf_model = joblib.load('rf_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.label_encoder = joblib.load('label_encoder.pkl')
            self.is_trained = True
            print("Models loaded successfully!")
        else:
            raise FileNotFoundError("Model files not found. Please train the model first.")

if __name__ == "__main__":
    predictor = CropYieldPredictor()
    metrics = predictor.train()
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)
