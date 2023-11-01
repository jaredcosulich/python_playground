from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess(data):
    data.drop('Id', axis=1, inplace=True) 
    # Assume that num_features and cat_features are lists of numerical and categorical feature names
    num_features = data.select_dtypes(include=['number']).columns.tolist()
    cat_features = data.select_dtypes(include=['object']).columns.tolist()

    # Remove 'SalePrice' from num_features as it is the target variable
    num_features.remove('SalePrice')

    # Define preprocessors for numerical and categorical data
    num_preprocessor = Pipeline([
        ('imputer', KNNImputer()),
        ('scaler', StandardScaler())
    ])

    cat_preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
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