import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from lib.KNNCategoricalImputer import KNNCategoricalImputer

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, data):
        self.numerical_features = data.select_dtypes(include=['number']).columns.tolist()
        self.categorical_features = data.select_dtypes(include=['object']).columns.tolist()

    def fit(self, X, y=None):
        num_preprocessor = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])

        cat_preprocessor = Pipeline([
            ('imputer', KNNCategoricalImputer(n_neighbors=5)),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_preprocessor, self.numerical_features),
                ('cat', cat_preprocessor, self.categorical_features)])
        
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)