from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet

def elastic_net_regression(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    elastic_net_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', ElasticNet(alpha=1.0, l1_ratio=0.5))
    ])
    elastic_net_pipeline.fit(X_train, y_train)

    # Evaluate the model on the test data
    score = elastic_net_pipeline.score(X_test, y_test)
    print(f'Elastic Net Regression R^2 score: {score}')

