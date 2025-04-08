import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_theme()

def load_data():
    """Load the supply chain dataset and description."""
    data_path = Path('data')
    
    # Load main dataset with Latin-1 encoding
    df = pd.read_csv(data_path / 'DataCoSupplyChainDataset.csv', encoding='latin1')
    
    # Load description file
    description = pd.read_csv(data_path / 'DescriptionDataCoSupplyChain.csv', encoding='latin1')
    
    return df, description

def preprocess_data(df):
    """Preprocess the data for feature selection."""
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Drop columns with high percentage of missing values
    df_processed = df_processed.drop(['Product Description', 'Order Zipcode'], axis=1)
    
    # Fill remaining missing values
    df_processed['Customer Lname'] = df_processed['Customer Lname'].fillna('Unknown')
    df_processed['Customer Zipcode'] = df_processed['Customer Zipcode'].fillna(df_processed['Customer Zipcode'].median())
    
    # Convert categorical variables to numerical
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df_processed[col] = label_encoders[col].fit_transform(df_processed[col].astype(str))
    
    return df_processed, label_encoders

def perform_feature_selection(df, target_column='Late_delivery_risk'):
    """Perform feature selection using various methods."""
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Initialize feature selection methods
    k_best_anova = SelectKBest(score_func=f_classif, k='all')
    k_best_mutual = SelectKBest(score_func=mutual_info_classif, k='all')
    
    # Fit the selectors
    k_best_anova.fit(X, y)
    k_best_mutual.fit(X, y)
    
    # Get scores
    anova_scores = pd.DataFrame({
        'Feature': X.columns,
        'ANOVA_F_Score': k_best_anova.scores_,
        'Mutual_Info_Score': k_best_mutual.scores_
    })
    
    # Sort by ANOVA F-score
    anova_scores = anova_scores.sort_values('ANOVA_F_Score', ascending=False)
    
    return anova_scores

def plot_feature_importance(scores, top_n=20):
    """Plot feature importance scores."""
    plt.figure(figsize=(15, 10))
    
    # Plot ANOVA F-scores
    plt.subplot(2, 1, 1)
    sns.barplot(data=scores.head(top_n), x='ANOVA_F_Score', y='Feature')
    plt.title(f'Top {top_n} Features by ANOVA F-score')
    plt.xlabel('ANOVA F-score')
    
    # Plot Mutual Information scores
    plt.subplot(2, 1, 2)
    sns.barplot(data=scores.head(top_n), x='Mutual_Info_Score', y='Feature')
    plt.title(f'Top {top_n} Features by Mutual Information')
    plt.xlabel('Mutual Information Score')
    
    plt.tight_layout()
    output_dir = Path('data/feature_selection_v1')
    plt.savefig(output_dir / 'feature_importance_v1.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    print("Loading data...")
    df, description = load_data()
    
    print("\nPreprocessing data...")
    df_processed, label_encoders = preprocess_data(df)
    
    print("\nPerforming feature selection...")
    feature_scores = perform_feature_selection(df_processed)
    
    # Save results to CSV with versioned path
    output_dir = Path('data/feature_selection_v1')
    feature_scores.to_csv(output_dir / 'feature_scores_v1.csv', index=False)
    print("\nFeature scores saved to 'data/feature_selection_v1/feature_scores_v1.csv'")
    
    print("\nGenerating feature importance plots...")
    plot_feature_importance(feature_scores)
    print("Plots saved to 'data/feature_selection_v1/feature_importance_v1.png'")
    
    # Print top 10 features
    print("\nTop 10 Most Important Features:")
    print(feature_scores.head(10)[['Feature', 'ANOVA_F_Score', 'Mutual_Info_Score']].to_string(index=False))

if __name__ == "__main__":
    main() 