from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .pipeline import create_pipeline

def test_create_pipeline():
    preprocessors = [('scaler', StandardScaler())]
    model = LogisticRegression()
    pipeline = create_pipeline(preprocessors, model)
    assert len(pipeline.steps) == 2
