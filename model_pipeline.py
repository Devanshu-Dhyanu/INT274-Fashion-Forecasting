import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor

def build_models():
    print("Building Forecasting Pipeline...")
    sales = pd.read_csv('sales_data.csv', parse_dates=['Date'])
    stores = pd.read_csv('stores_processed.csv')
    
    df = sales.merge(stores, on='Store')
    
    # Feature Engineering (Time Series)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 4 else 0)
    
    df = df.sort_values(['Store', 'Category', 'Date'])
    df['Demand_Lag_7'] = df.groupby(['Store', 'Category'])['Demand'].shift(7)
    df['Demand_Rolling_7'] = df.groupby(['Store', 'Category'])['Demand'].transform(lambda x: x.shift(1).rolling(7).mean())
    df = df.dropna()
    
    # Define Feature Groups
    cat_cols = ['Category', 'StoreType', 'Assortment', 'Region']
    num_cols = ['IsPromo', 'IsSeason', 'DayOfWeek', 'Month', 'IsWeekend', 'Cluster', 
                'Demand_Lag_7', 'Demand_Rolling_7', 'PCA1', 'PCA2']
    
    features = cat_cols + num_cols
    target = 'Demand'
    
    X = df[features]
    y = df[target]
    
    cutoff_date = df['Date'].max() - pd.Timedelta(days=30)
    X_train = X[df['Date'] <= cutoff_date]
    y_train = y[df['Date'] <= cutoff_date]
    X_test = X[df['Date'] > cutoff_date]
    y_test = y[df['Date'] > cutoff_date]
    
    # 1. Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols)
        ]
    )
    
    # 2. XGBoost Pipeline
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(n_estimators=300, learning_rate=0.07, max_depth=5, random_state=42))
    ])
    
    # 3. KNN Pipeline
    knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', KNeighborsRegressor(n_neighbors=5))
    ])
    
    # Train both
    print("Training XGBoost...")
    xgb_pipeline.fit(X_train, y_train)
    print("Training KNN...")
    knn_pipeline.fit(X_train, y_train)
    
    # Simple Evaluation
    y_pred_xgb = xgb_pipeline.predict(X_test)
    y_pred_knn = knn_pipeline.predict(X_test)
    
    mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb)
    mape_knn = mean_absolute_percentage_error(y_test, y_pred_knn)
    
    print(f"XGBoost MAPE: {mape_xgb:.4f}")
    print(f"KNN MAPE: {mape_knn:.4f}")
    
    # Save bundle
    model_data = {
        'xgb_pipeline': xgb_pipeline,
        'knn_pipeline': knn_pipeline,
        'features': features,
        'metrics': {'xgb': mape_xgb, 'knn': mape_knn}
    }
    joblib.dump(model_data, 'model_bundle.joblib')
    
    # Visualization
    plt.figure(figsize=(12, 6))
    sample_df = df[df['Date'] > cutoff_date].iloc[:100]
    plt.plot(sample_df['Date'], sample_df['Demand'], label='Actual', color='black', alpha=0.5)
    plt.plot(sample_df['Date'], xgb_pipeline.predict(sample_df[features]), label=f'XGBoost (MAPE: {mape_xgb:.2f})', color='green')
    plt.plot(sample_df['Date'], knn_pipeline.predict(sample_df[features]), label=f'KNN (MAPE: {mape_knn:.2f})', color='blue', linestyle='--')
    plt.title('Demand Forecasting: XGBoost vs KNN Performance')
    plt.legend()
    plt.savefig('forecast_results.png')
    
    print("Models Built and Saved Successfully.")
    return xgb_pipeline, knn_pipeline

if __name__ == "__main__":
    build_models()
