import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np

def phase(status):
    # read train test data
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    if status=="production":
        return X_train, y_train, X_test, "Empty"
    elif status == "testing":
        X_example_train, X_example_test, y_example_train, y_example_test = train_test_split(X_train, y_train,
                                                                                            test_size=0.33)
        return X_example_train, X_example_test, y_example_train, y_example_test