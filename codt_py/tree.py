from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, check_is_fitted, validate_data
from .codt_py import OptimalDecisionTreeClassifier as OCT, OptimalDecisionTreeRegressor as ORT, SearchStrategyEnum
import numpy as np

class OptimalDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth, strategy=SearchStrategyEnum.Dfs):
        self.max_depth = max_depth
        self.strategy = strategy

    def fit(self, X, y):
        X, y = validate_data(self, X, y, ensure_min_samples=2, dtype=np.float64)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.tree_ = OCT(self.max_depth, self.strategy)
        self.tree_.fit(X, y)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return np.take(self.classes_, self.tree_.predict(X))

class OptimalDecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, max_depth, strategy=SearchStrategyEnum.Dfs):
        self.max_depth = max_depth
        self.strategy = strategy

    def fit(self, X, y):
        X, y = validate_data(self, X, y, ensure_min_samples=2, dtype=np.float64, y_numeric=True)
        self.tree_ = ORT(self.max_depth, self.strategy)
        self.tree_.fit(X, y)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return self.tree_.predict(X)
