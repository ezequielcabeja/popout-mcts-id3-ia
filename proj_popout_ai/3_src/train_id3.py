import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def train():
    df = pd.read_csv("data/dataset.csv")

    X = df.drop("label", axis=1)
    y = df["label"]

    model = DecisionTreeClassifier(criterion="entropy")  # ID3-like
    model.fit(X, y)

    return model