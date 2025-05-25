from .tree import OptimalDecisionTreeClassifier, OptimalDecisionTreeRegressor

def all_search_strategies():
    return [
        "dfs",
        "dfs-prio",
        "and-or",
        "bfs-lb",
        "bfs-curiosity",
        "bfs-gosdt",
    ]

def all_upperbounds():
    return [
        "solutions-only",
        "tight-from-sibling",
        "for-remaining-interval",
    ]

def all_terminal_solvers():
    return [
        "leaf",
        "left-right",
        "d2",
    ]