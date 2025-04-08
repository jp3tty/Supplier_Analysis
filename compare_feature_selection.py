import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

def load_feature_scores(version):
    """Load feature scores from CSV file."""
    file_path = f'data/feature_selection_{version}/feature_scores_{version}.csv'
    df = pd.read_csv(file_path)
    df['Version'] = version
    return df

def create_comparison_plot(v1_df, v2_df, output_path):
    """Create a comparison plot of top features from both versions."""
    # Combine dataframes
    combined_df = pd.concat([v1_df, v2_df])
    
    # Normalize scores for comparison
    for score_type in ['ANOVA_F_Score', 'Mutual_Info_Score']:
        combined_df[f'{score_type}_normalized'] = combined_df.groupby('Version')[score_type].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
    
    # Get top 10 features from each version
    top_v1 = v1_df.nlargest(10, 'ANOVA_F_Score')
    top_v2 = v2_df.nlargest(10, 'ANOVA_F_Score')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot ANOVA scores
    sns.barplot(data=combined_df[combined_df['Feature'].isin(top_v1['Feature'])], 
                x='Feature', y='ANOVA_F_Score_normalized', hue='Version', ax=ax1)
    ax1.set_title('Normalized ANOVA F-Scores Comparison (Top V1 Features)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_ylabel('Normalized Score')
    
    # Plot Mutual Information scores
    sns.barplot(data=combined_df[combined_df['Feature'].isin(top_v2['Feature'])], 
                x='Feature', y='Mutual_Info_Score_normalized', hue='Version', ax=ax2)
    ax2.set_title('Normalized Mutual Information Scores Comparison (Top V2 Features)')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_ylabel('Normalized Score')
    
    plt.tight_layout()
    plt.savefig(output_path / 'feature_comparison.png')
    plt.close()

def create_rank_comparison(v1_df, v2_df, output_path):
    """Create a rank comparison plot."""
    # Get ranks for each version
    v1_ranks = v1_df.nlargest(20, 'ANOVA_F_Score')[['Feature', 'ANOVA_F_Score']]
    v2_ranks = v2_df.nlargest(20, 'ANOVA_F_Score')[['Feature', 'ANOVA_F_Score']]
    
    # Create rank comparison
    v1_ranks['Rank'] = range(1, len(v1_ranks) + 1)
    v2_ranks['Rank'] = range(1, len(v2_ranks) + 1)
    
    # Merge ranks
    rank_comparison = pd.merge(v1_ranks, v2_ranks, on='Feature', suffixes=('_v1', '_v2'), how='outer')
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=rank_comparison, x='Rank_v1', y='Rank_v2')
    
    # Add labels for top features
    for i, row in rank_comparison.iterrows():
        if row['Rank_v1'] <= 10 or row['Rank_v2'] <= 10:
            plt.annotate(row['Feature'], (row['Rank_v1'], row['Rank_v2']), 
                        xytext=(5, 5), textcoords='offset points')
    
    plt.plot([0, 20], [0, 20], 'k--', alpha=0.3)  # Diagonal line
    plt.xlabel('Rank in Version 1')
    plt.ylabel('Rank in Version 2')
    plt.title('Feature Rank Comparison Between Versions')
    plt.tight_layout()
    plt.savefig(output_path / 'rank_comparison.png')
    plt.close()

def create_score_distribution(v1_df, v2_df, output_path):
    """Create distribution plots of scores."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot ANOVA score distributions
    sns.kdeplot(data=v1_df['ANOVA_F_Score'], label='Version 1', ax=ax1)
    sns.kdeplot(data=v2_df['ANOVA_F_Score'], label='Version 2', ax=ax1)
    ax1.set_title('Distribution of ANOVA F-Scores')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Density')
    ax1.legend()
    
    # Plot Mutual Information score distributions
    sns.kdeplot(data=v1_df['Mutual_Info_Score'], label='Version 1', ax=ax2)
    sns.kdeplot(data=v2_df['Mutual_Info_Score'], label='Version 2', ax=ax2)
    ax2.set_title('Distribution of Mutual Information Scores')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Density')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'score_distributions.png')
    plt.close()

def main():
    # Create output directory
    output_path = Path('data/feature_comparison')
    output_path.mkdir(exist_ok=True)
    
    # Load data
    v1_df = load_feature_scores('v1')
    v2_df = load_feature_scores('v2')
    
    # Create visualizations
    create_comparison_plot(v1_df, v2_df, output_path)
    create_rank_comparison(v1_df, v2_df, output_path)
    create_score_distribution(v1_df, v2_df, output_path)
    
    print("Comparative visualizations have been generated in the 'data/feature_comparison' directory.")

if __name__ == "__main__":
    main() 