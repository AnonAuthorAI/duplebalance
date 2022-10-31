from sklearn.tree import DecisionTreeClassifier

class ResampleClassifier():
    
    def __init__(self,
        base_estimator = DecisionTreeClassifier(),
        sampler = None,
    ):
        self.sampler = sampler
        self.base_estimator = base_estimator
        self.resampled_data = None
        
    def fit(self, X, y, sampler=None, fit_kwargs={}, sampler_kwargs={}):
        clf = self.base_estimator
        sampler = self.sampler if sampler is None else sampler
        if sampler is None:
            X_res, y_res = X, y
        else:
            X_res, y_res = sampler.fit_resample(X, y, **sampler_kwargs)
        clf.fit(X_res, y_res, **fit_kwargs)
        self.n_training_samples_ = X_res.shape[0]
        
        return self

    def predict(self, *args):
        return self.base_estimator.predict(*args)
    
    def predict_proba(self, *args):
        return self.base_estimator.predict_proba(*args)