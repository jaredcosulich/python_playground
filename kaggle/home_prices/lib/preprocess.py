import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lib.KNNCategoricalImputer import KNNCategoricalImputer

def preprocess(data):
    data.drop('Id', axis=1, inplace=True) 
    # data['Age'] = data['YearBuilt'] - data['YrSold']

    # Assume that num_features and cat_features are lists of numerical and categorical feature names
    num_features = data.select_dtypes(include=['number']).columns.tolist()
    cat_features = data.select_dtypes(include=['object']).columns.tolist()

    # Remove 'SalePrice' from num_features as it is the target variable
    num_features.remove('SalePrice')

    # Right Skewed Features
    for feature in FEATURES_TO_ADJUST.get("rightSkewed"):
        data[feature] = np.log1p(data[feature])

    # Left Skewed Features
    for feature in FEATURES_TO_ADJUST.get("leftSkewed"):
        data[feature] = data[feature] ** 2

    # High Variability Features
    scaler = StandardScaler()
    for feature in FEATURES_TO_ADJUST.get("highVariability"):
        data[feature] = scaler.fit_transform(data[[feature]])

    # Define preprocessors for numerical and categorical data
    num_preprocessor = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])

    cat_preprocessor = Pipeline([
        ('imputer', KNNCategoricalImputer(n_neighbors=5)),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # We'll standardize numerical features and one-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_preprocessor, num_features),
            ('cat', cat_preprocessor, cat_features)])

    # Now we'll split the data into features and target variable
    X = data.drop('SalePrice', axis=1) 
    y = data['SalePrice']

    return (X, y, preprocessor)

FEATURES_TO_ADJUST = {
    "rightSkewed": [
        # "MSSubClass",
        # "LotFrontage",
        # "LotArea",
        # "OverallCond",
        # "MasVnrArea",
        # "BsmtFinSF1",
        # "BsmtFinSF2",
        # "BsmtUnfSF",
        # "TotalBsmtSF",
        # "1stFlrSF",
        # "2ndFlrSF",
        # "LowQualFinSF",
        # "GrLivArea",
        # "BsmtFullBath",
        # "BsmtHalfBath",
        # "HalfBath",
        # "KitchenAbvGr",
        # "TotRmsAbvGrd",
        # "Fireplaces",
        # "WoodDeckSF",
        # "OpenPorchSF",
        # "EnclosedPorch",
        # "3SsnPorch",
        # "ScreenPorch",
        # "PoolArea",
        # "MiscVal"
    ],
    "leftSkewed": [
        # "YearBuilt",
        # "YearRemodAdd",
        # "GarageYrBlt",
        # "Age"
    ],
    "highVariability": [
        # "LotArea",
        # "MasVnrArea",
        # "BsmtFinSF1",
        # "BsmtFinSF2",
        # "BsmtUnfSF",
        # "TotalBsmtSF",
        # "1stFlrSF",
        # "2ndFlrSF",
        # "GrLivArea",
        # "GarageArea",
        # "WoodDeckSF",
        # "OpenPorchSF",
        # "EnclosedPorch",
        # "MiscVal"
    ]
}