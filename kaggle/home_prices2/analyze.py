import pandas as pd
from lib.preprocess import preprocess
from lib.gradient_boost import gradient_boost_regression
from lib.utils import feature_importance_rf

def analyze():   
    data = pd.read_csv('train.csv')
    
    (X, y, preprocessor) = preprocess(data)
    (_, _, score) = gradient_boost_regression(X, y, preprocessor)
    
    feature_importance = feature_importance_rf(data, 'SalePrice')
    
    for feature in feature_importance:
        if feature[1] < 0.01:
            print(feature[0])

analyze()