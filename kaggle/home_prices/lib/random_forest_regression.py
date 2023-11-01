from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def random_forest_regression(X, y, preprocessor, with_grid_search=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    random_forest_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    score = 0
    if with_grid_search:
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20, 30]
        }
        grid_search = GridSearchCV(estimator=random_forest_pipeline, param_grid=param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        score = grid_search.score(X_test, y_test)
    else:
        random_forest_pipeline.fit(X_train, y_train)
        score = random_forest_pipeline.score(X_test, y_test)

    print(f'Random Forest Regression {"(grid search)" if with_grid_search else ""} R^2 score: {score}')

