from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted
from .codt_py import OptimalDecisionTreeClassifier as DT, SearchStrategyEnum

class OptimalDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth, strategy=SearchStrategyEnum.Dfs):
        self.tree = DT(max_depth=max_depth, strategy=strategy)
    def fit(self, X, y):
        # TODO check arguments
        self.tree.fit(X, y)
        # Trailing underscore necessary for detection of `check_is_fitted`
        self.dim_ = len(X[0])
    def predict(self, X):
        # TODO check argument is 2d float array of same size as self.dim_
        check_is_fitted(self)
        return self.tree.predict(X)
