from sklearn.pipeline import Pipeline

def create_pipeline(preprocessors, model):
    pipeline = Pipeline(preprocessors + [('model', model)])
    return pipeline