from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy
import pandas as pd
from initial_data import phase
numpy.random.seed(7)

X_train = pd.read_csv("tismnstomn_x_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("best_x_test.csv")
y_train.drop('order_id', axis=1, inplace=True)
X_test.drop('category', axis=1, inplace=True)
# #
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,test_size=0.33)
# X_train["orders_user"] = X_train.orders_user/X_train.orders_user.max()
# # days_since_prior_order
# X_test["orders_user"] = X_test.orders_user/X_test.orders_user.max()
# X_train["days_since_prior_order"] = X_train.days_since_prior_order/X_train.days_since_prior_order.max()
# # days_since_prior_order
# X_test["days_since_prior_order"] = X_test.days_since_prior_order/X_test.days_since_prior_order.max()
# X_train["order_hour_of_day"] = X_train.order_hour_of_day/X_train.order_hour_of_day.max()
# # days_since_prior_order
# X_test["order_hour_of_day"] = X_test.order_hour_of_day/X_test.order_hour_of_day.max()
# X_train["order_number"] = X_train.order_number/X_train.order_number.max()
# # days_since_prior_order
# X_test["order_number"] = X_test.order_number/X_test.order_number.max()
# X_train["order_dow"] = X_train.order_dow/X_train.order_dow.max()
# # days_since_prior_order
# X_test["order_dow"] = X_test.order_dow/X_test.order_dow.max()


model = Sequential()
model.add(Dense(12, input_dim=10, activation='tanh'))
model.add(Dense(24, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5000, batch_size=X_train.shape[0])


#evaluate the model
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#predictions = model.predict(X_test)
# round predictions
# rounded = pd.DataFrame([round(x[0]) for x in predictions])
# rounded.columns = ["category"]
# x_test_labels = pd.read_csv('data/X_test.csv')
# submission = pd.concat([x_test_labels["order_id"], rounded["category"]].astype(int), axis=1, keys=['order_id', 'category'])
# submission.to_csv("sample_submission.csv",index=False)
# # print(rounded)
