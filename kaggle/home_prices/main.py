import pandas as pd
import matplotlib.pyplot as plt
# from lib.analyze import analyze
from lib.preprocess import preprocess
from lib.linear_regression import linear_regression
from lib.ridge_regression import ridge_regression
from lib.elastic_net_regression import elastic_net_regression
from lib.random_forest_regression import random_forest_regression
from lib.gradient_boost import gradient_boost_regression
from lib.deep_learning import deep_learning
from lib.plot import plot

def main():
    # Load the data
    data = pd.read_csv('train.csv')

    # Analyze the data
    # analyze(data_train)
    
    # Preprocess the training data
    (X, y, preprocessor) = preprocess(data)

    # linear_regression(X, y, preprocessor)
    # ridge_regression(X, y, preprocessor)
    # elastic_net_regression(X, y, preprocessor)
    # random_forest_regression(X, y, preprocessor)
    # random_forest_regression(X, y, preprocessor, with_grid_search=True)    
    # gradient_boost_regression(X, y, preprocessor)
    (y_test_nn, y_pred_nn) = deep_learning(X, y, preprocessor)

    plt.figure(figsize=(10, 5))
    plot(plt, 2, y_test_nn, y_pred_nn, 'Neural Network', 'red')
    
    plt.tight_layout()
    plt.show()

main()