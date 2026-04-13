from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from .codt_py import (
    OptimalDecisionTreeClassifier as OCT,
)
import numpy as np

class _BaseOptimalDecisionTree(BaseEstimator, ABC):
    _rust_class = None  # Set in subclass

    def __init__(
        self,
        strategy="dfs-prio",
        timeout=None,
        lowerbound="class-count",
        upperbound="for-remaining-interval",
        cart_upperbound="disabled",
        intermediates=False,
        memory_limit=None,
    ):
        self.strategy = strategy
        self.timeout = timeout
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.cart_upperbound = cart_upperbound
        self.intermediates = intermediates
        self.memory_limit = memory_limit

    def _init_rust_class(self):
        lowerbound = self.lowerbound
        if isinstance(lowerbound, (list, tuple, set)):
            lowerbound = ",".join(lowerbound)

        return self._rust_class(
            self.strategy,
            self.timeout,
            lowerbound,
            self.upperbound,
            self.cart_upperbound,
            self.intermediates,
            self.memory_limit,
        )

    def fit(self, X, y):
        X, y = self._validate_data_odt(X, y)
        self.tree_ = self._init_rust_class()
        self.tree_.fit(X, y)
        return self

    @abstractmethod
    def _validate_data_odt(self, X, y):
        pass

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return self._predict_impl(X)

    def _predict_impl(self, X):
        return self.tree_.predict(X)

    def tree(self):
        check_is_fitted(self)
        return self.tree_.tree()

    def status(self):
        check_is_fitted(self)
        return self.tree_.status()

    def is_perfect(self):
        check_is_fitted(self)
        return self.tree_.is_perfect()

    def branch_count(self):
        check_is_fitted(self)
        return self.tree_.branch_count()

    def tree_size(self):
        check_is_fitted(self)
        return self.tree_.tree_size()

    def intermediate_lbs(self):
        check_is_fitted(self)
        return self.tree_.intermediate_lbs()

    def intermediate_ubs(self):
        check_is_fitted(self)
        return self.tree_.intermediate_ubs()

    def expansions(self):
        check_is_fitted(self)
        return self.tree_.expansions()

    def memory_usage_bytes(self):
        check_is_fitted(self)
        return self.tree_.memory_usage_bytes()


class OptimalDecisionTreeClassifier(_BaseOptimalDecisionTree, ClassifierMixin):
    _rust_class = OCT

    def _validate_data_odt(self, X, y):
        X, y = validate_data(self, X, y, ensure_min_samples=2, dtype=np.float64)
        self.classes_, y = np.unique(y, return_inverse=True)
        return X, y

    def _predict_impl(self, X):
        return np.take(self.classes_, self.tree_.predict(X))
