import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def preprocess_data(df):
    """Preprocess the data for logistic regression."""
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Select features for the model
    features = [
        'Days for shipping (real)',
        'Days for shipment (scheduled)',
        'Order Item Quantity',
        'Order Item Total',
        'Market',
        'Customer Segment',
        'Shipping Mode'
    ]
    
    # Create dummy variables for categorical features
    categorical_features = ['Market', 'Customer Segment', 'Shipping Mode']
    df = pd.get_dummies(df, columns=categorical_features)
    
    # Update features list with dummy variables
    features = [col for col in df.columns if any(f in col for f in features)]
    
    # Scale numerical features
    numerical_features = [
        'Days for shipping (real)',
        'Days for shipment (scheduled)',
        'Order Item Quantity',
        'Order Item Total'
    ]
    
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df[features], df['Late_delivery_risk']

def train_model(X, y):
    """Train and evaluate the logistic regression model."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define hyperparameters to tune
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
        'penalty': ['l1', 'l2'],  # Regularization type
        'solver': ['liblinear'],  # Algorithm to use in optimization
        'max_iter': [100, 200, 500, 1000]  # Maximum number of iterations
    }
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        LogisticRegression(),
        param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='roc_auc',  # Use ROC AUC as the scoring metric
        n_jobs=-1,  # Use all available cores
        return_train_score=True  # Return training scores for each parameter combination
    )
    
    # Fit the grid search
    print("Performing hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    # Create results DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Select and rename relevant columns
    columns_to_keep = [
        'param_C',
        'param_penalty',
        'param_solver',
        'param_max_iter',
        'mean_train_score',
        'mean_test_score',
        'std_train_score',
        'std_test_score',
        'rank_test_score'
    ]
    
    results_df = results_df[columns_to_keep]
    results_df.columns = [
        'C',
        'Penalty',
        'Solver',
        'Max Iterations',
        'Mean Train Score',
        'Mean Test Score',
        'Std Train Score',
        'Std Test Score',
        'Rank'
    ]
    
    # Sort by rank
    results_df = results_df.sort_values('Rank')
    
    # Save results to CSV
    results_file = 'hyperparameter_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\nHyperparameter tuning results saved to {results_file}")
    
    # Get the best model
    model = grid_search.best_estimator_
    
    # Print best parameters
    print("\nBest hyperparameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), 
                annot=True, 
                fmt='d',
                cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    plt.close()
    
    return model, X_train.columns

def main():
    # Load the data
    print("Loading data...")
    try:
        df = pd.read_csv('data/DataCoSupplyChainDataset.csv', encoding='utf-8')
    except UnicodeDecodeError:
        print("Trying alternative encoding...")
        df = pd.read_csv('data/DataCoSupplyChainDataset.csv', encoding='latin1')
    
    # Preprocess the data
    print("Preprocessing data...")
    X, y = preprocess_data(df)
    
    # Train and evaluate the model
    print("Training model...")
    model, feature_names = train_model(X, y)
    
    # Print feature importance
    print("\nFeature Importance:")
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    })
    importance = importance.sort_values('Coefficient', ascending=False)
    print(importance)

if __name__ == "__main__":
    main() 