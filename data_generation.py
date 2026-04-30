import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_data(num_stores=12, num_days=400):
    np.random.seed(42)
    
    # 1. Fashion Stores Data
    store_ids = range(1, num_stores + 1)
    store_types = ['Flagship', 'Outlet', 'Boutique', 'Online']
    assortments = ['Organic Essentials', 'Premium Ethical', 'Eco-Basics']
    regions = ['London', 'Paris', 'Milan', 'New York', 'Tokyo']
    
    stores = pd.DataFrame({
        'Store': store_ids,
        'StoreType': np.random.choice(store_types, num_stores),
        'Assortment': np.random.choice(assortments, num_stores),
        'Region': np.random.choice(regions, num_stores),
        'SustainabilityRating': np.random.uniform(4.0, 5.0, num_stores),
        'EcoTax_Incentive': np.random.choice([0, 1], num_stores)
    })
    
    # 2. Fashion Item Demand Data (instead of just generic sales)
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    all_demand = []
    categories = ['Recycled Denim', 'Organic Cotton Tees', 'Vegan Leather bags', 'Bamboo Activewear']
    
    for store_id in store_ids:
        # Base demand based on store type
        base_demand = np.random.randint(200, 500)
        
        for date in dates:
            for cat in categories:
                day_of_week = date.weekday()
                is_season = 1 if date.month in [11, 12, 3, 4] else 0 # Peak seasons
                is_promo = 1 if np.random.random() > 0.85 else 0
                
                # Factors
                weekly_factor = 1.3 if day_of_week >= 4 else 1.0
                season_factor = 1.5 if is_season else 0.8
                promo_factor = 1.8 if is_promo else 1.0
                noise = np.random.normal(1, 0.15)
                
                demand = int(base_demand * weekly_factor * season_factor * promo_factor * noise)
                
                all_demand.append({
                    'Date': date,
                    'Store': store_id,
                    'Category': cat,
                    'Demand': demand,
                    'IsPromo': is_promo,
                    'IsSeason': is_season,
                    'StockLevel': np.random.randint(100, 1000)
                })
            
    demand_df = pd.DataFrame(all_demand)
    
    # Save
    stores.to_csv('stores_data.csv', index=False)
    demand_df.to_csv('sales_data.csv', index=False) # Keep name same for compatibility or rename
    
    print(f"Fashion Intelligence Data generated: {num_stores} boutiques, {num_days} days.")
    return stores, demand_df

if __name__ == "__main__":
    generate_data()
