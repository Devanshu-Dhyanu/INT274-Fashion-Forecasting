# Project Report: Bioscale - Ethical Fashion Intelligence
**Course Code:** INT274  
**Project Title:** Conscious Demand Forecasting & Store Profiling  

---

## 1. Abstract
The fashion industry is responsible for significant environmental waste due to overproduction. This project, **Bioscale**, introduces an AI-driven approach to demand forecasting specifically tailored for ethical fashion boutiques. By utilizing XGBoost for predictive modeling and K-Means clustering for store profiling, the system enables businesses to optimize inventory, reduce carbon footprints, and align production with actual consumer demand.

## 2. Problem Statement
Traditional inventory management often relies on manual intuition, leading to stockouts or excessive unsold inventory. In the context of ethical fashion, this waste contradicts the core values of sustainability. There is a critical need for a high-precision forecasting tool that accounts for seasonality, promotional impact, and regional variations.

## 3. Methodology

### 3.1 Data Generation & Preprocessing
Since real-world ethical fashion data is often proprietary, a synthetic dataset was generated simulating 12 international boutiques over 400 days. The data includes:
- **Categorical Data:** Store types, assortments (Organic, Eco-Basics), and regions.
- **Temporal Factors:** Seasonality (Peak vs. Off-peak), day-of-week effects.
- **Demand Drivers:** Sustainability ratings and promotional eco-incentives.

### 3.2 Feature Engineering & Analysis
- **Store Clustering:** Used **K-Means** to group boutiques into 3 distinct clusters based on performance metrics.
- **Dimensionality Reduction:** Applied **PCA** to visualize high-dimensional store vectors in a 2D space.
- **Lag Features:** Created 7-day demand lags and rolling averages to capture time-series trends.

### 3.3 Modeling
Two core algorithms were implemented within a scikit-learn pipeline:
1. **XGBoost Regressor:** An ensemble gradient boosting model for high-precision non-linear forecasting.
2. **K-Nearest Neighbors (KNN):** Used for similarity-based inference and comparative store analysis.

## 4. Implementation (Tech Stack)
- **Programming Language:** Python
- **Machine Learning:** Scikit-Learn, XGBoost
- **Data Visualization:** Plotly, Seaborn, Matplotlib
- **Web Framework:** Streamlit (Deployed on Streamlit Cloud)

## 5. Results
- **Forecasting Accuracy:** The XGBoost model achieved a significant reduction in Mean Absolute Percentage Error (MAPE) compared to baseline models.
- **Interactive Intelligence:** The Streamlit dashboard provides real-time "Scenario Simulation," allowing admins to test demand predictions under different promotional or seasonal conditions.

## 6. Conclusion
Bioscale demonstrates that AI can be a powerful ally in the transition to circular fashion. By predicting demand more accurately, the system directly contributes to reduced textile waste and more efficient resource allocation.
