from sklearn.svm import SVC
import numpy as np
from .model_selection import perform_grid_search

def test_perform_grid_search():
    X = np.random.rand(10, 2)
    y = np.random.randint(2, size=10)
    model = SVC()
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    best_params = perform_grid_search(model, param_grid, X, y)
    assert 'C' in best_params
    assert 'kernel' in best_params