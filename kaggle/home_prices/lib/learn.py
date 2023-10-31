from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

def learn(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with a linear regression model as a baseline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)

    # Evaluate the model on the test data
    score = pipeline.score(X_test, y_test)
    print(f'Model R^2 score: {score}')
