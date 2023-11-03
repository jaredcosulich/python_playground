import pandas as pd
from lib.CustomPreprocessor import CustomPreprocessor
from lib.gradient_boost import gradient_boost_regression

def main():
    # Load the data
    data = pd.read_csv('train.csv')
    # test_data = pd.read_csv('test.csv')

    X = data.drop('SalePrice', axis=1) 
    y = data['SalePrice']

    preprocessor = CustomPreprocessor(data)
    (y_test_gb, y_pred_gb) = gradient_boost_regression(X, y, preprocessor)
    print(y_pred_gb)    

main()