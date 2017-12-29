from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy
import pandas as pd
from initial_data import phase
numpy.random.seed(7)

X_train = pd.read_csv("new2.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("new_x_test.csv")
y_train.drop('order_id', axis=1, inplace=True)
#
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,test_size=0.33)

model = Sequential()
model.add(Dense(12, input_dim=11, activation='tanh'))
model.add(Dense(11, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=5)


# evaluate the model
# scores = model.evaluate(X_test, y_test)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(X_test)
# round predictions
rounded = pd.DataFrame([round(x[0]) for x in predictions])
rounded.columns = ["category"]
x_test_labels = pd.read_csv('data/X_test.csv')
submission = pd.concat([x_test_labels["order_id"], rounded["category"]].astype(int), axis=1, keys=['order_id', 'category'])
submission.to_csv("sample_submission.csv",index=False)
# print(rounded)
