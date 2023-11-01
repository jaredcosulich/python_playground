from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy='mean')
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        self.num_features = None
        self.cat_features = None

    def fit(self, X, y=None):
        self.num_features = X.select_dtypes(include=['number']).columns.tolist()
        self.cat_features = X.select_dtypes(include=['object']).columns.tolist()
        self.num_imputer.fit(X[self.num_features])
        self.cat_imputer.fit(X[self.cat_features])
        self.scaler.fit(self.num_imputer.transform(X[self.num_features]))
        self.encoder.fit(self.cat_imputer.transform(X[self.cat_features]))
        return self

    def transform(self, X, y=None):
        data_num_imputed = pd.DataFrame(self.num_imputer.transform(X[self.num_features]), columns=self.num_features)
        data_cat_imputed = pd.DataFrame(self.cat_imputer.transform(X[self.cat_features]), columns=self.cat_features)
        data_num_scaled = pd.DataFrame(self.scaler.transform(data_num_imputed), columns=self.num_features)
        data_cat_encoded = pd.DataFrame(self.encoder.transform(data_cat_imputed), columns=self.encoder.get_feature_names_out(self.cat_features))
        data_processed = pd.concat([data_num_scaled, data_cat_encoded], axis=1)
        return data_processed
