import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def encode_categorical_features(data, columns=None):
    """
    Encode categorical features using one-hot encoding.
    
    Parameters:
    data (pd.DataFrame): The dataset.
    columns (list of str, optional): The columns to encode. If None, all string columns are encoded.
    
    Returns:
    pd.DataFrame: The dataset with categorical features encoded.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if columns is None:
        columns = data.select_dtypes(include=['object']).columns
    encoded_data = encoder.fit_transform(data[columns])
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(columns),
        index=data.index
    )
    return data.drop(columns, axis=1).join(encoded_df)