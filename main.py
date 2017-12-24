# first feature has to be the ratio of beverages in each order.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np

X_train=pd.read_csv("data/X_train.csv")#the data that will be used for our classifier
y_train=pd.read_csv("data/y_train.csv")#contains the true labels of the order_ids that will be used as training dataset
X_test=pd.read_csv("data/X_test.csv")

print "x_train is \n",X_train.head(1)
print "y_train is \n",y_train.head(1)

X_example_train, X_example_test, y_example_train, y_example_test = train_test_split(X_train, y_train, test_size=0.33)

logreg = LogisticRegression()

logreg.fit(X_example_train, y_example_train["category"])
y_pred = logreg.predict(X_example_test)
print("logreg",accuracy_score(y_example_test["category"], y_pred))

# submission instructions
# X_test['category']=logreg.predict(X_test)
# submission=X_test[['order_id','category']]
# submission.to_csv("sample_submission.csv",index=False)

#read data
orders_df=pd.read_csv("data/orders.csv")
products_df=pd.read_csv("data/products.csv")
#departments_df=pd.read_csv("data/departments.csv")
#aisles_df = pd.read_csv("data/aisles.csv")
order_products_prior_df = pd.read_csv("data/order_products__prior.csv")
#keep products ids from department 7
products_dep_seven = products_df[products_df.department_id == 7]
products_dep_seven_id = list(products_dep_seven.product_id.values)

#take order prior ids with products from department 7
order_products_prior_df = order_products_prior_df[order_products_prior_df.product_id.isin(products_dep_seven_id)]
orders_prior_id = list(order_products_prior_df.order_id.values)

#count orders per user
orders_df= orders_df[orders_df.order_id.isin(orders_prior_id)]
orders_per_user = orders_df.groupby('user_id').count()

from matplotlib import pyplot as plt
#%matplotlib inline
orders_per_user.order_id.hist()
plt.title('Distribution of orders')
plt.xlabel('Number of orders')
plt.show()

print "variance: ",orders_per_user.order_id.var()
print "std: ",orders_per_user.order_id.std()
print "mean: ",orders_per_user.order_id.mean()

#Logistic regression
logreg = LogisticRegression()
print logreg.fit(X_train, y_train["category"])
y_pred = logreg.predict(X_test)
print X_train.head(4)

#print y_pred.shape
#print("logreg",accuracy_score(y_test["category"], y_pred))