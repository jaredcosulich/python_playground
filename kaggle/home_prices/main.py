import pandas as pd
# from lib.analyze import analyze
from lib.preprocess import preprocess
from lib.learn import learn

def main():
    # Load the data
    data = pd.read_csv('train.csv')
    # data_test = pd.read_csv('test.csv')

    # Analyze the data
    # analyze(data_train)
    
    # Preprocess the training data
    (X, y, preprocessor) = preprocess(data)

    # Preprocess the testing data
    # (X_test, y_test) = preprocess(data_test)

    learn(X, y, preprocessor)


main()