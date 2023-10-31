import numpy as np
from .feature_selection import select_features

def test_select_features():
    X = np.random.rand(10, 10)
    y = np.random.randint(2, size=10)
    selected_features = select_features(X, y, n_features=5)
    assert selected_features.shape[1] == 5