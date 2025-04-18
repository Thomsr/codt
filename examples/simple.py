from sklearn.metrics import accuracy_score
from codt_py import OptimalDecisionTreeClassifier
import pandas as pd

df = pd.read_csv("../contree/datasets/bank.txt", sep=" ", header=None)

X = df[df.columns[1:]].to_numpy()
y = df[0].to_numpy()

classifier = OptimalDecisionTreeClassifier(max_depth=2)
classifier.fit(X, y)
y_pred = classifier.predict(X)
print(accuracy_score(y, y_pred))
