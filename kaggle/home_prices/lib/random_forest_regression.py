from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def random_forest_regression(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    random_forest_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    random_forest_pipeline.fit(X_train, y_train)

    # Evaluate the model on the test data
    score = random_forest_pipeline.score(X_test, y_test)
    print(f'Random Forest Regression R^2 score: {score}')

