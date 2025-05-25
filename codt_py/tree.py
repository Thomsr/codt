from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, check_is_fitted, validate_data
from .codt_py import OptimalDecisionTreeClassifier as OCT, OptimalDecisionTreeRegressor as ORT, search_strategy_from_string
import numpy as np

class OptimalDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=2, strategy="dfs-prio", complexity_cost=0.0, timeout=None, upperbound="for-remaining-interval", terminal_solver="left-right", intermediates=False):
        self.max_depth = max_depth
        self.strategy = strategy
        self.complexity_cost = complexity_cost
        self.timeout = timeout
        self.upperbound = upperbound
        self.terminal_solver = terminal_solver
        self.intermediates = intermediates

    def fit(self, X, y):
        X, y = validate_data(self, X, y, ensure_min_samples=2, dtype=np.float64)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.tree_ = OCT(self.max_depth, search_strategy_from_string(self.strategy), self.complexity_cost)
        self.tree_.fit(X, y)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return np.take(self.classes_, self.tree_.predict(X))
    
    def get_tree(self):
        check_is_fitted(self)
        return self.tree_.tree()
    
    def intermediate_lbs(self):
        check_is_fitted(self)
        return self.tree_.intermediate_lbs()
    
    def intermediate_ubs(self):
        check_is_fitted(self)
        return self.tree_.intermediate_ubs()


class OptimalDecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, max_depth=2, strategy="dfs-prio", complexity_cost=0.0, timeout=None, upperbound="for-remaining-interval", terminal_solver="left-right", intermediates=False):
        self.max_depth = max_depth
        self.strategy = strategy
        self.complexity_cost = complexity_cost
        self.timeout = timeout
        self.upperbound = upperbound
        self.terminal_solver = terminal_solver
        self.intermediates = intermediates

    def fit(self, X, y):
        X, y = validate_data(self, X, y, ensure_min_samples=2, dtype=np.float64, y_numeric=True)
        self.tree_ = ORT(self.max_depth, search_strategy_from_string(self.strategy), self.complexity_cost)
        self.tree_.fit(X, y)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return self.tree_.predict(X)
    
    def get_tree(self):
        check_is_fitted(self)
        return self.tree_.tree()
    
    def intermediate_lbs(self):
        check_is_fitted(self)
        return self.tree_.intermediate_lbs()
    
    def intermediate_ubs(self):
        check_is_fitted(self)
        return self.tree_.intermediate_ubs()
