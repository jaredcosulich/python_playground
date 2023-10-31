import pandas as pd
from .preprocessing import handle_missing_values

def test_handle_missing_values():
    data = pd.DataFrame({
        'A': [1, 2, None, 4],
        'B': [5, None, 7, 8]
    })
    imputed_data_mean = handle_missing_values(data)
    assert imputed_data_mean.equals(pd.DataFrame({
        'A': [1, 2, 2.3333333333333335, 4],
        'B': [5, 6.666666666666667, 7, 8]
    }))
    
    imputed_data_median = handle_missing_values(data, strategy='median')
    assert imputed_data_median.equals(pd.DataFrame({
        'A': [1, 2, 2.0, 4],
        'B': [5, 7.0, 7, 8]
    }))
    
    imputed_data_most_frequent = handle_missing_values(data, strategy='most_frequent')
    assert imputed_data_most_frequent.equals(pd.DataFrame({
        'A': [1, 2, 1.0, 4],
        'B': [5, 5.0, 7, 8]
    }))