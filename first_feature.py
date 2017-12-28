from initial_data import phase
from prediction import get_prediction # when we are ready to test it.
import pandas as pd
import numpy as np

X_train, X_test, y_train, y_test = phase("testing")
order_products_prior_df = pd.read_csv("data/order_products__prior.csv")
products_df = pd.read_csv("data/products.csv")
orders_df = pd.read_csv("data/orders.csv")
products_df=pd.read_csv("data/products.csv")


how= 'left'
key= 'product_id'
columns= ['aisle_id', 'product_id','product_name']

get_items7 = order_products_prior_df.merge(products_df, how=how, on=key)
get_items7.drop(columns, inplace=True, axis=1) # vriskoume ta antikeimena apo 7

fun = lambda x: 1 if x == 7 else 0
get_items7["department_id"] = get_items7['department_id'].apply(fun) # i turn every non department 7 id to 0 and every 7 to 1

to_shop = get_items7.copy()

all_items_7 = get_items7.groupby("order_id", as_index = False).department_id.sum()
all_items_bought = to_shop.groupby("order_id", as_index = False).add_to_cart_order.max()
mid = all_items_7.merge(all_items_bought, how="left", on="order_id")
mid = mid.rename(columns={"department_id":"beverages"})
ratios = mid["beverages"] / mid["add_to_cart_order"]
ratios.columns = ["ratio"]

new_feature = pd.concat([mid["order_id"], ratios], axis=1, keys=['rat', 'column'])
new_feature.columns = ["order_id","ratio"]

feature1 = orders_df.merge(new_feature, how="left", on="order_id")
feature1.drop(['user_id',"order_number","order_dow","order_hour_of_day","days_since_prior_order"], inplace = True, axis =1)
feature1.fillna(0,inplace=True)
su = orders_df.merge(feature1, how="left",on="order_id")
su.drop(['order_id','order_number', 'order_dow','order_hour_of_day','days_since_prior_order'], inplace = True, axis =1)
su = su.groupby("user_id",as_index=False).ratio.mean()
X_train = X_train.merge(su, how='inner', on=['user_id'])
X_test = X_test.merge(su, how='inner', on=['user_id'])
get_prediction(X_train, X_test, y_train, y_test)

