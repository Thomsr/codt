import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

data = np.loadtxt("./data/sampled/contraceptive_0.5_4.txt")
y = data[:, 0]
X = data[:, 1:]

clf = DecisionTreeClassifier(
    criterion="gini",
    min_samples_leaf=1,
    min_samples_split=2,
    max_depth=None,
    random_state=42,
)
clf.fit(X, y)

pred = clf.predict(X)
misclassifications = np.sum(pred != y)
print(f"Number of misclassifications: {misclassifications}")
assert misclassifications == 0, "Tree did not perfectly fit the data!"
print("CART tree perfectly fits the data.")
print(f"Tree size (number of nodes): {clf.tree_.node_count}")

tree_rules = export_text(clf, feature_names=[f"x{i}" for i in range(X.shape[1])])
print(tree_rules)

print("Root feature:", clf.tree_.feature[0])
print("Root threshold:", clf.tree_.threshold[0])
print("Root impurity:", clf.tree_.impurity[0])
print("Left child impurity:", clf.tree_.impurity[clf.tree_.children_left[0]])
print("Right child impurity:", clf.tree_.impurity[clf.tree_.children_right[0]])
print("Left n_samples:", clf.tree_.n_node_samples[clf.tree_.children_left[0]])
print("Right n_samples:", clf.tree_.n_node_samples[clf.tree_.children_right[0]])
