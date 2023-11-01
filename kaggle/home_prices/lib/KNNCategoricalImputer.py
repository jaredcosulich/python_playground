import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer

class KNNCategoricalImputer:
    def __init__(self, n_neighbors=5):
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
        self.categories_ = None

    def fit(self, X, y=None):
        # Encode categories to numbers
        X_encoded = self.encoder.fit_transform(X)
        self.imputer.fit(X_encoded)
        # Store the categories for inverse transform later
        self.categories_ = self.encoder.categories_
        return self

    def transform(self, X):
        X_encoded = self.encoder.transform(X)
        X_imputed = self.imputer.transform(X_encoded)
        # Round imputed values to ensure they are integers
        X_imputed_rounded = np.round(X_imputed)
        # Decode numbers back to categories
        return self.encoder.inverse_transform(X_imputed_rounded)
