import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
from data_generation import generate_data
from analysis import perform_analysis
from model_pipeline import build_models

# Page Configuration
st.set_page_config(page_title="Ethical Fashion Intelligence", layout="wide", page_icon="🌿")

# --- PREMIUM CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@100;300;400;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F8F9FA !important;
        font-family: 'Montserrat', sans-serif !important;
    }

    /* Hero Image Header */
    .hero-container {
        height: 350px;
        background-image: linear-gradient(rgba(26,26,26,0.6), rgba(26,26,26,0.6)), url('file:///C:/Users/devan/.gemini/antigravity/brain/c8255c33-17f8-45f2-ad67-937af6693d35/fashion_demand_hero_1777523468737.png');
        background-size: cover;
        background-position: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        border-radius: 20px;
        margin-bottom: 40px;
        color: white;
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    
    .hero-title {
        font-size: 4rem !important;
        font-weight: 700 !important;
        letter-spacing: -2px;
        text-transform: uppercase;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1.2rem !important;
        letter-spacing: 12px;
        font-weight: 300;
        text-transform: uppercase;
        margin-top: 10px;
    }

    /* Professional Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E0E0E0;
    }
    
    /* Metrics Fix */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #F0F0F0;
    }
    [data-testid="stMetricValue"] {
        color: #2D5A27 !important;
        font-weight: 700 !important;
    }

    /* Buttons */
    div.stButton > button {
        background: #2D5A27 !important;
        color: white !important;
        border-radius: 50px !important;
        border: none !important;
        padding: 0.8rem 3rem !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        background: #1A1A1A !important;
        transform: scale(1.05);
    }
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 40px;
        border-bottom: 2px solid #E0E0E0;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        color: #666 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #2D5A27 !important;
        border-bottom-color: #2D5A27 !important;
    }
    /* Global Text Colors */
    h1, h2, h3, h4, h5, h6 {
        color: #1A1A1A !important;
    }
    
    [data-testid="stWidgetLabel"] p {
        color: #1A1A1A !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 0.9rem !important;
    }

    /* Radio Button & Selectbox Options Visibility */
    div[data-testid="stRadio"] label, 
    div[data-testid="stSelectbox"] label,
    .st-ae, .st-af, .st-ag, .st-ah { 
        color: #1A1A1A !important;
        font-weight: 500 !important;
    }

    /* Ensuring all labels within widgets are dark */
    label[data-testid="stWidgetCustomControlLabel"] {
        color: #1A1A1A !important;
    }

    .stMarkdown p {
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- APP COMPONENT ---

# Hero Header
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Bioscale</h1>
    <p class="hero-subtitle">Conscious Demand Forecaster</p>
</div>
""", unsafe_allow_html=True)

# Data Loading
@st.cache_data
def load_all():
    if not os.path.exists('stores_processed.csv'):
        generate_data()
        perform_analysis()
        build_models()
    return pd.read_csv('sales_data.csv', parse_dates=['Date']), pd.read_csv('stores_processed.csv')

sales, stores = load_all()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/2D5A27/leaf.png", width=60)
    st.title("Admin Console")
    if st.button("RECALIBRATE SYSTEM"):
        with st.spinner("Processing High-Dim Fashion Vectors..."):
            generate_data()
            perform_analysis()
            build_models()
        st.success("SYSTEM READY")
    
    st.markdown("---")
    st.caption("DEPLOYED BY: DEVANSHU (ME DEVELOPER)")
    st.caption("ALGO: XGBOOST ENSEMBLE + PCA")

# Top Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("ACTIVE BOUTIQUES", len(stores))
m2.metric("ECO-RATING AVG", f"{stores['SustainabilityRating'].mean():.2f}/5.0")
m3.metric("ITEM CATEGORIES", len(sales['Category'].unique()))
m4.metric("PEAK SEASON DEMAND", f"{sales[sales['IsSeason']==1]['Demand'].mean():.0f} units")

st.markdown("<br>", unsafe_allow_html=True)

# Pages/Tabs
t1, t2, t3, t4 = st.tabs(["📊 INTELLIGENCE", "🔮 FORECASTER", "🤖 MANUAL INFERENCE", "📂 ARCHIVES"])

with t1:
    st.subheader("Deep Learning Cluster Analysis")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        # Plotly Cluster Pie
        fig_pie = px.pie(stores, names='Cluster', hole=0.6, 
                         color_discrete_sequence=['#2D5A27', '#1A1A1A', '#A0A0A0'],
                         title="Store Categorization Distribution")
        fig_pie.update_layout(showlegend=True, font_family="Montserrat")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        # Plotly PCA Scatter
        fig_pca = px.scatter(stores, x='PCA1', y='PCA2', color='Cluster',
                            symbol='StoreType', hover_name='Region',
                            color_discrete_sequence=['#2D5A27', '#1A1A1A', '#A0A0A0'],
                            title="Principal Component Variance Projection")
        fig_pca.update_layout(plot_bgcolor='white', font_family="Montserrat")
        st.plotly_chart(fig_pca, use_container_width=True)

with t2:
    st.subheader("Category-wise Demand Trends")
    sel_cat = st.multiselect("Select Fashion Categories", sales['Category'].unique(), default=sales['Category'].unique()[0])
    
    trend_df = sales[sales['Category'].isin(sel_cat)].groupby(['Date', 'Category'])['Demand'].mean().reset_index()
    fig_line = px.line(trend_df, x='Date', y='Demand', color='Category', 
                       color_discrete_sequence=px.colors.sequential.Greens_r,
                       title="Historical Demand Trajectory")
    fig_line.update_layout(plot_bgcolor='rgba(0,0,0,0)', font_family="Montserrat")
    st.plotly_chart(fig_line, use_container_width=True)

with t3:
    st.subheader("Scenario Simulation (Comparative Analysis)")
    if os.path.exists('model_bundle.joblib'):
        bundle = joblib.load('model_bundle.joblib')
        xgb_model = bundle['xgb_pipeline']
        knn_model = bundle['knn_pipeline']
        feats = bundle['features']
        
        in1, in2 = st.columns(2)
        with in1:
            st_sel = st.selectbox("BOUTIQUE", stores['Store'].unique())
            cat_sel = st.selectbox("ITEM CATEGORY", sales['Category'].unique())
            model_choice = st.radio("SELECT ALGORITHM", ["XGBOOST", "KNN (K-Nearest)"], horizontal=True)
            promo_sel = st.radio("ECO-PROMO ACTIVE", [0, 1], horizontal=True)
        
        with in2:
            lag_val = st.number_input("DEMAND 7 DAYS AGO", value=300)
            roll_val = st.number_input("AVG DEMAND LAST WEEK", value=310)
            dow_sel = st.select_slider("DAY OF WEEK (MON-SUN)", range(7), 0)
            
        store_inf = stores[stores['Store'] == st_sel].iloc[0]
        
        if st.button("RUN COMPARATIVE INFERENCE"):
            input_vec = {
                'Store': st_sel, 
                'Category': cat_sel,
                'StoreType': store_inf['StoreType'],
                'Assortment': store_inf['Assortment'],
                'Region': store_inf['Region'],
                'IsPromo': promo_sel, 'IsSeason': 1, 'DayOfWeek': dow_sel, 'Month': 5,
                'IsWeekend': 1 if dow_sel >= 4 else 0, 'Cluster': store_inf['Cluster'],
                'Demand_Lag_7': lag_val, 'Demand_Rolling_7': roll_val,
                'PCA1': store_inf['PCA1'], 'PCA2': store_inf['PCA2']
            }
            input_df = pd.DataFrame([input_vec])
            
            selected_model = xgb_model if model_choice == "XGBOOST" else knn_model
            prediction = selected_model.predict(input_df)[0]
            
            # Show Similarity Analysis (KNN based)
            st.info(f"Insight: Store {st_sel} is most similar to Store {store_inf['Nearest_Store']} based on KNN analysis.")
            
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #1A1A1A, #2D5A27); color: white; padding: 40px; border-radius: 20px; text-align: center;">
                <p style="margin:0; letter-spacing: 5px; opacity:0.8;">{model_choice} ESTIMATED UNITS</p>
                <h1 style="margin:0; font-size: 5rem;">{prediction:,.0f}</h1>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Engine Offline. Recalibrate System.")

with t4:
    st.subheader("Global Inventory Repository")
    st.dataframe(sales, use_container_width=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #AAA; letter-spacing: 10px;">DEVANSHU | GLOBAL FASHION INTELLIGENCE</p>', unsafe_allow_html=True)
