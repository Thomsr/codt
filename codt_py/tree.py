from sklearn.base import BaseEstimator, ClassifierMixin
from .codt_py import OptimalDecisionTreeClassifier as DT

class OptimalDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth, max_nodes):
        self.tree = DT(max_depth=max_depth, max_nodes=max_nodes)
        print("wrapper")
    def fit(self, X, y):
        self.tree.fit(X, y)
    def predict(self, X):
        return self.tree.predict(X)
