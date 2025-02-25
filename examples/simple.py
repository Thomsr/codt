from codt_py import OptimalDecisionTreeClassifier
import pandas as pd

df = pd.read_csv("../contree/datasets/bank.txt", sep=" ", header=None)

X = df[df.columns[1:]].to_numpy()
y = df[0].to_numpy()

classifier = OptimalDecisionTreeClassifier(max_depth=2, max_nodes=3)
classifier.fit(X, y)
print(classifier.predict(X))
