import pandas as pd

def create_mapping(df, column_key, column_value):
    """
    Create a mapping between two columns in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_key (str): The name of the column to use as the keys of the mapping.
    column_value (str): The name of the column to use as the values of the mapping.

    Returns:
    dict: A dictionary where the keys are the unique values in column_key, 
          and the values are the corresponding values in column_value.
    """
    if column_key not in df.columns or column_value not in df.columns:
        raise ValueError(f"Columns {column_key} or {column_value} not found in DataFrame")

    mapping = df.set_index(column_key)[column_value].to_dict()
    return mapping