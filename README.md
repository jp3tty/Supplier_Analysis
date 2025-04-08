# Supplier Analysis Project

This project focuses on analyzing supplier data to identify patterns and insights that can help optimize supply chain operations. The analysis includes feature selection and importance analysis to determine which factors most significantly impact delivery performance.

## Project Structure

```
supplier_analysis/
├── data/                    # Data directory
│   ├── DataCoSupplyChainDataset.csv
│   └── DescriptionDataCoSupplyChain.csv
├── feature_selection_v1.py  # Initial feature selection implementation
├── feature_selection_v2.py  # Enhanced feature selection with improved preprocessing
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## Features

- **Data Loading and Preprocessing**: Handles loading of supply chain data with proper encoding and preprocessing steps
- **Feature Selection**: Implements multiple feature selection methods:
  - ANOVA F-test
  - Mutual Information
- **Visualization**: Generates plots showing feature importance scores
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

## Usage

The project contains two versions of feature selection:

1. **Version 1** (`feature_selection_v1.py`):
   - Basic feature selection implementation
   - Uses label encoding for categorical variables
   - Generates feature importance plots

2. **Version 2** (`feature_selection_v2.py`):
   - Enhanced preprocessing with one-hot encoding
   - Additional datetime feature extraction
   - More detailed categorical feature analysis
   - Improved visualization and reporting

To run either version:
```bash
python feature_selection_v1.py  # For version 1
python feature_selection_v2.py  # For version 2
```

## Output

The scripts generate:
- Feature importance scores saved as CSV files
- Visualization plots showing top features
- Detailed analysis of categorical features (in v2)

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

All dependencies are listed in `requirements.txt`.

## Contributing

Feel free to submit issues and enhancement requests.
