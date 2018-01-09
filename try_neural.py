from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from keras.optimizers import Adam
import keras
import numpy
import pandas as pd
numpy.random.seed(7)

name = "with_20_features_x_"
X_train = pd.read_csv("x_trains/%strain.csv"%name)
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("x_tests/%stest.csv"%name)
y_train.drop('order_id', axis=1, inplace=True)
# # #

phase = "production"

if phase == "production":
    model = Sequential()
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1],
                    init=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=2000, batch_size=X_train.shape[0])
    predictions = model.predict(X_test)
    rounded = pd.DataFrame([int(round(x[0])) for x in predictions])
    rounded.columns = ["category"]
    x_test_labels = pd.read_csv('data/X_test.csv')
    submission = pd.concat([x_test_labels["order_id"], rounded["category"]], axis=1, keys=['order_id', 'category'])
    submission.to_csv("%ssubmission.csv"%name,index=False)

else:
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33)
    try:
        model = Sequential()
        model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1],
                        init=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                        activation='tanh'))
        model.add(Dense(X_train.shape[1], activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10000, batch_size=X_train.shape[0])
    except KeyboardInterrupt:
        scores = model.evaluate(X_test, y_test)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
