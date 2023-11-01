import pandas as pd
# from lib.analyze import analyze
from lib.preprocess import preprocess
from lib.linear_regression import linear_regression
from lib.ridge_regression import ridge_regression
from lib.elastic_net_regression import elastic_net_regression
from lib.random_forest_regression import random_forest_regression

def main():
    # Load the data
    data = pd.read_csv('train.csv')

    # Analyze the data
    # analyze(data_train)
    
    # Preprocess the training data
    (X, y, preprocessor) = preprocess(data)

    linear_regression(X, y, preprocessor)
    ridge_regression(X, y, preprocessor)
    elastic_net_regression(X, y, preprocessor)
    random_forest_regression(X, y, preprocessor)    


main()