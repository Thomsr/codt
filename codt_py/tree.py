from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, check_is_fitted, validate_data
from .codt_py import (
    OptimalDecisionTreeClassifier as OCT,
    OptimalDecisionTreeRegressor as ORT,
)
import numpy as np

class _BaseOptimalDecisionTree(BaseEstimator, ABC):
    _rust_class = None  # Set in subclass

    def __init__(
        self,
        max_depth=2,
        strategy="dfs-prio",
        complexity_cost=0.0,
        timeout=None,
        upperbound="for-remaining-interval",
        terminal_solver="left-right",
        intermediates=False,
        node_lowerbound=True,
        memory_limit=None,
    ):
        self.max_depth = max_depth
        self.strategy = strategy
        self.complexity_cost = complexity_cost
        self.timeout = timeout
        self.upperbound = upperbound
        self.terminal_solver = terminal_solver
        self.intermediates = intermediates
        self.node_lowerbound = node_lowerbound
        self.memory_limit = memory_limit

    def _init_rust_class(self):
        return self._rust_class(
            self.max_depth,
            self.strategy,
            self.complexity_cost,
            self.timeout,
            self.upperbound,
            self.terminal_solver,
            self.intermediates,
            self.node_lowerbound,
            self.memory_limit,
        )

    @abstractmethod
    def fit(self, X, y):
        pass

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return self._predict_impl(X)

    def _predict_impl(self, X):
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

    def expansions(self):
        check_is_fitted(self)
        return self.tree_.expansions()


class OptimalDecisionTreeClassifier(_BaseOptimalDecisionTree, ClassifierMixin):
    _rust_class = OCT

    def fit(self, X, y):
        X, y = validate_data(self, X, y, ensure_min_samples=2, dtype=np.float64)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.tree_ = self._init_rust_class()
        self.tree_.fit(X, y)
        return self

    def _predict_impl(self, X):
        return np.take(self.classes_, self.tree_.predict(X))


class OptimalDecisionTreeRegressor(_BaseOptimalDecisionTree, RegressorMixin):
    _rust_class = ORT

    def fit(self, X, y):
        X, y = validate_data(self, X, y, ensure_min_samples=2, dtype=np.float64, y_numeric=True)
        self.tree_ = self._init_rust_class()
        self.tree_.fit(X, y)
        return self
