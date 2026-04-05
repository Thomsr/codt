from sklearn.metrics import accuracy_score
from codt_py import OptimalDecisionTreeClassifier
import pandas as pd

df = pd.read_csv("./data/sampled/backache_0.2_1.txt", sep=" ", header=None)

X = df[df.columns[1:]].to_numpy()
y = df[0].to_numpy()

classifier = OptimalDecisionTreeClassifier()
classifier.fit(X, y)
y_pred = classifier.predict(X)
print(y_pred, y)
print(classifier.get_tree())
print(classifier.status())
print(classifier.branch_count())
print(accuracy_score(y, y_pred))
