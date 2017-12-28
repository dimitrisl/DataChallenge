from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def get_prediction(X_train, X_test, y_train, y_test):

    logreg = LogisticRegression()
    X_train.drop(['user_id','order_id'], axis=1, inplace=True)
    X_test.drop(['user_id','order_id'], axis=1, inplace=True)
    y_train.drop('order_id', axis=1, inplace=True)
    logreg.fit(X_train, y_train["category"])
    y_pred = logreg.predict(X_test)
    print "the logistic regression result is", y_pred
    if y_test is not str:
        print "The accuracy achieved is", accuracy_score(y_test["category"], y_pred)
