from sklearn.feature_selection import SelectKBest

def select_features(X, y, n_features=5):
    selector = SelectKBest(k=n_features)
    selector.fit(X, y)
    return selector.transform(X)
