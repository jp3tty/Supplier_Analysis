import pandas as pd
from datetime import datetime

def main():
    # Load the data
    print("Loading data...")
    try:
        df = pd.read_csv('data/input/DataCoSupplyChainDataset.csv', encoding='utf-8')
    except UnicodeDecodeError:
        print("Trying alternative encoding...")
        df = pd.read_csv('data/input/DataCoSupplyChainDataset.csv', encoding='latin1')
    
    # Convert order date to datetime and extract temporal features
    print("Processing temporal features...")
    df['order_date'] = pd.to_datetime(df['order date (DateOrders)'])
    df['day_of_week'] = df['order_date'].dt.day_name()
    df['month'] = df['order_date'].dt.month_name()
    
    # Select and rename columns
    temporal_df = df[['Order Id', 'order date (DateOrders)', 'day_of_week', 'month']]
    temporal_df.columns = ['Order_ID', 'Order_Date', 'Day_of_Week', 'Month']
    
    # Sort by order date
    temporal_df = temporal_df.sort_values('Order_Date')
    
    # Save to CSV
    output_file = 'data/input/temporal_features.csv'
    temporal_df.to_csv(output_file, index=False)
    print(f"\nTemporal features saved to {output_file}")
    
    # Print sample of the data
    print("\nSample of the generated data:")
    print(temporal_df.head())

if __name__ == "__main__":
    main() 