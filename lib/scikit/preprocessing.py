import pandas as pd
from sklearn.impute import SimpleImputer

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in the dataset.
    
    Parameters:
    data (pd.DataFrame): The dataset.
    strategy (str): The imputation strategy - 'mean', 'median', or 'most_frequent'.
    
    Returns:
    pd.DataFrame: The dataset with missing values handled.
    """
    imputer = SimpleImputer(strategy=strategy)
    imputed_data = imputer.fit_transform(data)
    return pd.DataFrame(imputed_data, columns=data.columns)