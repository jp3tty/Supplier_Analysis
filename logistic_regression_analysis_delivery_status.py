import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
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
from datetime import datetime

def preprocess_data(df):
    """Preprocess the data for logistic regression."""
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Convert order date to datetime and extract temporal features
    df['order_date'] = pd.to_datetime(df['order date (DateOrders)'])
    df['day_of_week'] = df['order_date'].dt.day_name()
    df['month'] = df['order_date'].dt.month_name()
    
    # Select features for the model
    features = [
        'Order Item Quantity',
        'Order Item Total',
        'Market',
        'Customer Segment',
        'Shipping Mode',
        'day_of_week',
        'month',
        'Days for shipping (real)',
        'Days for shipment (scheduled)'
    ]
    
    # Create dummy variables for categorical features
    categorical_features = ['Market', 'Customer Segment', 'Shipping Mode', 'day_of_week', 'month']
    df = pd.get_dummies(df, columns=categorical_features)
    
    # Update features list with dummy variables
    features = [col for col in df.columns if any(f in col for f in features)]
    
    # Scale numerical features
    numerical_features = [
        'Order Item Quantity',
        'Order Item Total',
        'Days for shipping (real)',
        'Days for shipment (scheduled)'
    ]
    
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df[features], df['Delivery Status']

def train_model(X, y):
    """Train and evaluate the logistic regression model."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define hyperparameters to tune
    param_grid = {
        'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'estimator__penalty': ['l1', 'l2'],
        'estimator__solver': ['liblinear'],
        'estimator__max_iter': [1000, 2000, 3000]
    }
    
    # Initialize GridSearchCV with OneVsRestClassifier
    base_estimator = LogisticRegression()
    ovr_classifier = OneVsRestClassifier(base_estimator)
    
    grid_search = GridSearchCV(
        ovr_classifier,
        param_grid,
        cv=5,
        scoring='roc_auc_ovr',
        n_jobs=-1,
        return_train_score=True
    )
    
    # Fit the grid search
    print("Performing hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    # Create results DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Select and rename relevant columns
    columns_to_keep = [
        'param_estimator__C',
        'param_estimator__penalty',
        'param_estimator__solver',
        'param_estimator__max_iter',
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
    results_file = 'data/output/delivery_status_hyperparameter_results.csv'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    results_df.to_csv(results_file, index=False)
    print(f"\nHyperparameter tuning results saved to {results_file}")
    
    # Get the best model
    model = grid_search.best_estimator_
    
    # Print best parameters
    print("\nBest hyperparameters found:")
    best_params = {k.replace('estimator__', ''): v for k, v in grid_search.best_params_.items()}
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Analyze feature importance
    print("\nFeature Importance Analysis:")
    # Get the unique classes
    classes = model.classes_
    feature_importance_list = []
    
    # Calculate feature importance for each class
    for i, class_label in enumerate(classes):
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Class': class_label,
            'Coefficient': model.estimators_[i].coef_[0],
            'Absolute_Coefficient': abs(model.estimators_[i].coef_[0])
        })
        feature_importance_list.append(feature_importance)
    
    # Combine all feature importances
    feature_importance = pd.concat(feature_importance_list, ignore_index=True)
    
    # Sort by absolute coefficient value
    feature_importance = feature_importance.sort_values('Absolute_Coefficient', ascending=False)
    
    # Calculate odds ratios
    feature_importance['Odds_Ratio'] = np.exp(feature_importance['Coefficient'])
    
    # Save feature importance to CSV
    feature_importance_file = 'data/output/delivery_status_feature_importance.csv'
    feature_importance.to_csv(feature_importance_file, index=False)
    
    # Print top 10 most important features for each class
    print("\nTop Factors Affecting Delivery Status:")
    for class_label in classes:
        print(f"\nClass: {class_label}")
        class_importance = feature_importance[feature_importance['Class'] == class_label].head(10)
        for idx, row in class_importance.iterrows():
            direction = "increases" if row['Coefficient'] > 0 else "decreases"
            print(f"{row['Feature']}:")
            print(f"  - {direction} the likelihood of {class_label}")
            print(f"  - Odds Ratio: {row['Odds_Ratio']:.4f}")
            print(f"  - Coefficient: {row['Coefficient']:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create output directory for plots
    os.makedirs('data/output/plots', exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix - Delivery Status')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('data/output/plots/delivery_status_confusion_matrix.png')
    plt.close()
    
    # Plot feature importance for each class
    plt.figure(figsize=(15, 5 * len(classes)))
    for i, class_label in enumerate(classes):
        plt.subplot(len(classes), 1, i+1)
        class_importance = feature_importance[feature_importance['Class'] == class_label].head(10)
        sns.barplot(x='Absolute_Coefficient', y='Feature', data=class_importance)
        plt.title(f'Top 10 Factors Affecting Delivery Status - {class_label}')
        plt.xlabel('Absolute Coefficient Value')
        plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('data/output/plots/delivery_status_feature_importance.png')
    plt.close()
    
    return model, X_train.columns

def main():
    # Load the data
    print("Loading data...")
    try:
        df = pd.read_csv('data/input/DataCoSupplyChainDataset.csv', encoding='utf-8')
    except UnicodeDecodeError:
        print("Trying alternative encoding...")
        df = pd.read_csv('data/input/DataCoSupplyChainDataset.csv', encoding='latin1')
    
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
        'Coefficient': model.estimators_[0].coef_[0]
    })
    importance = importance.sort_values('Coefficient', ascending=False)
    print(importance)

if __name__ == "__main__":
    main() 