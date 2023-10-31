import pandas as pd
from .encoders import encode_categorical_features

def test_encode_categorical_features():
    data = pd.DataFrame({
        'A': ['cat', 'dog', 'dog'],
        'B': ['red', 'blue', 'red']
    })
    encoded_data = encode_categorical_features(data)
    assert encoded_data.equals(pd.DataFrame({
        'A_cat': [1.0, 0.0, 0.0],
        'A_dog': [0.0, 1.0, 1.0],
        'B_blue': [0.0, 1.0, 0.0],
        'B_red': [1.0, 0.0, 1.0]
    }))