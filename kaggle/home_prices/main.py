import pandas as pd
import matplotlib.pyplot as plt
# from lib.analyze import analyze
from lib.preprocess import preprocess
# from lib.linear_regression import linear_regression
# from lib.ridge_regression import ridge_regression
# from lib.elastic_net_regression import elastic_net_regression
# from lib.random_forest_regression import random_forest_regression
from lib.gradient_boost import gradient_boost_regression
from lib.gradient_boost_final import gradient_boost_regression_final
# from lib.deep_learning4 import deep_learning
from lib.plot import plot
from lib.utils import feature_importance_rf#, cross_validation_feature_selection#, analyze_feature_distributions#, preprocessing_recommendations

def main():
    plt.figure(figsize=(10, 5))

    # Load the data
    data = pd.read_csv('train.csv')
    # test_data = pd.read_csv('test.csv')

    # Analyze the data
    # analyze(data_train)
    
    # Preprocess the training data
    (X, y, preprocessor) = preprocess(data)

    # summary = analyze_feature_distributions(X)
    # recs = preprocessing_recommendations(summary)
    # print(recs)

    # linear_regression(X, y, preprocessor)
    # ridge_regression(X, y, preprocessor)
    # elastic_net_regression(X, y, preprocessor)
    # random_forest_regression(X, y, preprocessor)
    # random_forest_regression(X, y, preprocessor, with_grid_search=True)    
    (y_test_gb, y_pred_gb) = gradient_boost_regression(X, y, preprocessor)
    # (y_test_nn, y_pred_nn) = deep_learning(X, y, preprocessor)

    plot(plt, 1, y_test_gb, y_pred_gb, 'Gradient Boost', 'green')
    # plot(plt, 2, y_test_nn, y_pred_nn, 'Neural Network', 'red')
    
    plt.tight_layout()
    plt.show()
    # y_pred = gradient_boost_regression_final(X, y, test_data, preprocessor)
    # test_data['SalePrice'] = y_pred
    # selected_columns = test_data[['Id', 'SalePrice']]
    # selected_columns.to_csv('predictions.csv', index=False)


main()