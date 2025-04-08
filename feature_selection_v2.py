import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
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
    """Preprocess the data for feature selection using get_dummies."""
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Drop columns with high percentage of missing values
    df_processed = df_processed.drop(['Product Description', 'Order Zipcode'], axis=1)
    
    # Fill remaining missing values
    df_processed['Customer Lname'] = df_processed['Customer Lname'].fillna('Unknown')
    df_processed['Customer Zipcode'] = df_processed['Customer Zipcode'].fillna(df_processed['Customer Zipcode'].median())
    
    # Handle datetime columns
    date_columns = ['order date (DateOrders)', 'shipping date (DateOrders)']
    for col in date_columns:
        df_processed[col] = pd.to_datetime(df_processed[col])
        df_processed[f'{col}_year'] = df_processed[col].dt.year
        df_processed[f'{col}_month'] = df_processed[col].dt.month
        df_processed[f'{col}_day'] = df_processed[col].dt.day
        df_processed[f'{col}_dayofweek'] = df_processed[col].dt.dayofweek
    df_processed = df_processed.drop(date_columns, axis=1)
    
    # Get categorical columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    # Analyze cardinality of categorical columns
    print("\nCategorical feature cardinality:")
    for col in categorical_cols:
        unique_count = df_processed[col].nunique()
        print(f"{col}: {unique_count} unique values")
    
    # Create dummy variables for categorical columns with reasonable cardinality
    df_dummies = pd.DataFrame()
    for col in categorical_cols:
        if df_processed[col].nunique() <= 20:  # Only create dummies for features with <= 20 categories
            dummies = pd.get_dummies(
                df_processed[col],
                prefix=col,
                prefix_sep='_',
                drop_first=True
            )
            df_dummies = pd.concat([df_dummies, dummies], axis=1)
        else:
            print(f"\nSkipping {col} due to high cardinality ({df_processed[col].nunique()} unique values)")
    
    # Get numeric columns
    numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    
    # Combine numeric columns with dummy variables
    df_final = pd.concat([
        df_processed[numeric_cols],
        df_dummies
    ], axis=1)
    
    # Print preprocessing summary
    print("\nFeature transformation summary:")
    print(f"Original features: {len(df.columns)}")
    print(f"After preprocessing: {len(df_final.columns)}")
    print("\nSample of dummy variables created:")
    dummy_examples = [col for col in df_final.columns if '_' in col][:5]
    print("\n".join(f"- {col}" for col in dummy_examples))
    
    return df_final

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
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'ANOVA_F_Score': k_best_anova.scores_,
        'Mutual_Info_Score': k_best_mutual.scores_
    })
    
    # Sort by ANOVA F-score
    feature_scores = feature_scores.sort_values('ANOVA_F_Score', ascending=False)
    
    return feature_scores

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
    output_dir = Path('data/feature_selection_v2')
    plt.savefig(output_dir / 'feature_importance_v2.png', bbox_inches='tight', dpi=300)
    plt.close()

def analyze_categorical_features(scores):
    """Analyze the importance of categorical features."""
    # Get categorical features (those with '_' in their names)
    categorical_scores = scores[scores['Feature'].str.contains('_')]
    
    print("\nTop 10 Categorical Features:")
    print(categorical_scores.head(10)[['Feature', 'ANOVA_F_Score', 'Mutual_Info_Score']].to_string(index=False))
    
    # Group by original feature name (before dummification)
    original_features = categorical_scores['Feature'].str.split('_').str[0]
    grouped_scores = categorical_scores.groupby(original_features)['ANOVA_F_Score'].mean().sort_values(ascending=False)
    
    print("\nAverage Importance by Original Categorical Feature:")
    print(grouped_scores.head(10).to_string())

def main():
    print("Loading data...")
    df, description = load_data()
    
    print("\nPreprocessing data...")
    df_processed = preprocess_data(df)
    
    print("\nPerforming feature selection...")
    feature_scores = perform_feature_selection(df_processed)
    
    # Save results to CSV with versioned path
    output_dir = Path('data/feature_selection_v2')
    feature_scores.to_csv(output_dir / 'feature_scores_v2.csv', index=False)
    print("\nFeature scores saved to 'data/feature_selection_v2/feature_scores_v2.csv'")
    
    print("\nGenerating feature importance plots...")
    plot_feature_importance(feature_scores)
    print("Plots saved to 'data/feature_selection_v2/feature_importance_v2.png'")
    
    # Print top 10 features
    print("\nTop 10 Most Important Features:")
    print(feature_scores.head(10)[['Feature', 'ANOVA_F_Score', 'Mutual_Info_Score']].to_string(index=False))
    
    # Analyze categorical features
    analyze_categorical_features(feature_scores)

if __name__ == "__main__":
    main() 