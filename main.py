import data_generation
import analysis
import model_pipeline
import os

def main():
    print("=== Demand Forecasting Project Main Orchestrator ===")
    
    # 1. Data Generation
    print("\n[Step 1/4] Generating synthetic data...")
    data_generation.generate_data(num_stores=10, num_days=365)
    
    # 2. Analysis & Clustering
    print("\n[Step 2/4] Performing Store Cluster Analysis, Encoding and PCA...")
    analysis.perform_analysis()
    
    # 3. Modeling
    print("\n[Step 3/4] Building Forecasting Models (KNN & XGBoost Pipeline)...")
    model_pipeline.build_models()
    
    # 4. Final Summary
    print("\n[Step 4/4] Project Execution Complete!")
    print("\nGenerated Files:")
    print("- stores_processed.csv: Final processed store data with clusters and PCA")
    print("- store_clusters_pca.png: Visualization of store clusters")
    print("- forecast_results.png: Visualization of XGBoost demand forecast vs actuals")
    
    print("\nTechniques used:")
    print("1. Store Cluster Analysis (K-Means)")
    print("2. Categorical Encoding (Label Encoding)")
    print("3. Dimensionality Reduction (PCA)")
    print("4. KNN Regression")
    print("5. XGBoost Time-Series Pipeline")

if __name__ == "__main__":
    main()
