import numpy as np
from .preprocessing import impute_missing_values

def test_impute_missing_values():
    data = np.array([[1, np.nan], [3, 4], [np.nan, 6]])
    imputed_data = impute_missing_values(data)
     # assuming 'mean' strategy
    assert imputed_data[0, 1] == 5 
    assert imputed_data[2, 0] == 2 
