from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb

def gradient_boost_regression(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=100, random_state=42))
    ])
    xgb_pipeline.fit(X_train, y_train)

    # Evaluate the model on the test data
    score = xgb_pipeline.score(X_test, y_test)
    print(f'Gradient Boost Regression R^2 score: {score}')

    y_pred = xgb_pipeline.predict(X_test)
    return (y_test, y_pred)