from sklearn.impute import SimpleImputer

def impute_missing_values(data, strategy='mean'):
    imputer = SimpleImputer(strategy=strategy)
    return imputer.fit_transform(data)
