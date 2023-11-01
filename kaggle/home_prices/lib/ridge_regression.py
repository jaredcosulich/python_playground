from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

def ridge_regression(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ridge_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=1.0))
    ])
    ridge_pipeline.fit(X_train, y_train)

    # Evaluate the model on the test data
    score = ridge_pipeline.score(X_test, y_test)
    print(f'Ridge Regression R^2 score: {score}')

