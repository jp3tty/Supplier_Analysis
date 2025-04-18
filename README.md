# Supplier Analysis Project

This project focuses on analyzing supplier data to identify patterns and insights that can help optimize supply chain operations. The analysis includes feature selection and importance analysis to determine which factors most significantly impact delivery performance. The data was sourced from Kaggle (https://www.kaggle.com/code/hetul173757/notebook681ff23643/input).

## Project Structure

```
supplier_analysis/
├── data/                    # Data directory
│   ├── input/              # Input data files
│   │   ├── DataCoSupplyChainDataset.csv
│   │   ├── DescriptionDataCoSupplyChain.csv
│   │   ├── tokenized_access_logs.csv
│   │   └── temporal_features.csv
│   └── output/             # Output files
│       ├── plots/          # Visualization files
│       │   ├── confusion_matrix.png
│       │   ├── roc_curve.png
│       │   └── feature_importance.png
│       ├── feature_importance.csv
│       └── hyperparameter_results.csv
├── feature_selection_v1.py  # Initial feature selection implementation
├── feature_selection_v2.py  # Enhanced feature selection with improved preprocessing
├── logistic_regression_analysis.py  # Logistic regression analysis with hyperparameter tuning
├── csv_to_mysql.py         # Script to load CSV files into MySQL database
├── create_temporal_features.py  # Script to generate temporal features from order dates
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## Features

- **Data Loading and Preprocessing**: Handles loading of supply chain data with proper encoding and preprocessing steps
- **Feature Selection**: Implements multiple feature selection methods:
  - ANOVA F-test
  - Mutual Information
- **Logistic Regression Analysis**: Includes hyperparameter tuning and feature importance analysis
- **Temporal Feature Extraction**: Extracts and analyzes time-based features from order dates
- **Database Integration**: Loads processed data into MySQL database
- **Visualization**: Generates plots showing feature importance scores and model performance metrics
- **Categorical Feature Analysis**: Special handling and analysis of categorical variables

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd supplier_analysis
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up MySQL database (optional):
   - Configure database credentials in environment variables
   - Run `csv_to_mysql.py` to load data into the database

## Usage

The project contains several analysis scripts:

1. **Feature Selection**:
   - `feature_selection_v1.py`: Basic implementation with label encoding
   - `feature_selection_v2.py`: Enhanced version with one-hot encoding

2. **Logistic Regression Analysis** (`logistic_regression_analysis.py`):
   - Hyperparameter tuning
   - Feature importance analysis
   - Model performance evaluation
   - Generates visualizations in `data/output/plots/`

3. **Temporal Feature Generation** (`create_temporal_features.py`):
   - Extracts day of week and month from order dates
   - Creates temporal_features.csv in input directory

4. **Database Integration** (`csv_to_mysql.py`):
   - Loads CSV files into MySQL database
   - Handles multiple file encodings
   - Creates necessary database tables

To run any script:
```bash
python script_name.py
```

## Output

The scripts generate various outputs organized in the `data/output` directory:

- **CSV Files**:
  - Feature importance scores
  - Hyperparameter tuning results

- **Visualizations** (in `data/output/plots/`):
  - Confusion matrix
  - ROC curve
  - Feature importance plots

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pymysql
- sqlalchemy
- python-dotenv

All dependencies are listed in `requirements.txt`.

## Contributing

Feel free to submit issues and enhancement requests.
