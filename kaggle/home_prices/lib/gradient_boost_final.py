from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb

def gradient_boost_regression_final(X_train, y_train, X_test, preprocessor):
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=100, random_state=42))
    ])
    xgb_pipeline.fit(X_train, y_train)

    y_pred = xgb_pipeline.predict(X_test)
    return y_pred