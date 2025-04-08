import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better visualizations
plt.style.use('seaborn-v0_8')  # Using a built-in style
sns.set_theme()  # Set seaborn theme

def load_data():
    """Load the supply chain dataset and description."""
    data_path = Path('data')
    
    # Load main dataset with Latin-1 encoding
    df = pd.read_csv(data_path / 'DataCoSupplyChainDataset.csv', encoding='latin1')
    
    # Load description file
    description = pd.read_csv(data_path / 'DescriptionDataCoSupplyChain.csv', encoding='latin1')
    
    return df, description

def analyze_dataframe(df, description):
    """Perform initial analysis of the dataframe."""
    print("\n=== Dataset Overview ===")
    print(f"Number of rows: {df.shape[0]:,}")
    print(f"Number of columns: {df.shape[1]:,}")
    
    print("\n=== Column Information ===")
    # Create a DataFrame with column information
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': df.count(),
        'Dtype': df.dtypes
    })
    
    # Merge with description
    info_df = info_df.merge(
        description[['FIELDS', 'DESCRIPTION']],
        left_on='Column',
        right_on='FIELDS',
        how='left'
    )
    
    # Format and print the information with better alignment
    header = f"{'#':<3} {'Column':<35} {'Non-Null':<10} {'Dtype':<10} {'Description':<45}"
    separator = "-" * len(header)
    print("\n" + header)
    print(separator)
    
    for idx, row in info_df.iterrows():
        # Clean up the description by removing leading/trailing whitespace and colons
        desc = str(row['DESCRIPTION']).strip()
        if desc.startswith(':'):
            desc = desc[1:].strip()
        
        # Format the non-null count without commas
        non_null = f"{row['Non-Null Count']}"
        
        # Ensure consistent spacing
        print(f"{idx+1:<3} {row['Column']:<35} {non_null:<10} {str(row['Dtype']):<10} {desc[:45]:<45}")
    
    print("\n=== Missing Values ===")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percent
    })
    print(missing_df[missing_df['Missing Values'] > 0])

def plot_numerical_distributions(df):
    """Plot distributions of numerical columns."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns[:9]  # Limit to first 9 numerical columns
    
    # Create subplots for numerical columns
    n_cols = 3
    n_rows = (len(numerical_cols) + 2) // 3  # 3 columns per row
    
    plt.figure(figsize=(15, 5*n_rows))
    
    for idx, col in enumerate(numerical_cols, 1):
        plt.subplot(n_rows, 3, idx)
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('data/numerical_distributions.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    df, description = load_data()
    
    # Perform initial analysis
    analyze_dataframe(df, description)
    
    # Plot numerical distributions
    print("\nGenerating distribution plots...")
    plot_numerical_distributions(df)
    print("Plots saved to 'data/numerical_distributions.png'")
    
    # # Print description file contents
    # print("\n=== Dataset Description ===")
    # print(description.to_string())

if __name__ == "__main__":
    main() 