from feature_selector import join
from initial_data import phase
from prediction import get_prediction # when we are ready to test it.
import pandas as pd
import numpy as np

X_train, X_test, y_train, y_test = phase("testing")
order_products_priod_df = pd.read_csv("data/order_products__prior.csv")
products_df = pd.read_csv("data/products.csv")
orders_df = pd.read_csv("data/orders.csv")
products_df=pd.read_csv("data/products.csv")

params = {
    'how': 'left',
    'key': 'product_id',
    'columns': ['aisle_id', 'product_id','product_name'],
    'name': None
}

feature = join(order_products_priod_df, products_df, params)
fun = lambda x: 1 if x == 7 else 0
feature["department_id"] = feature['department_id'].apply(fun) # i turn every non department 7 id to 0 and every 7 to 1

pf = feature.copy()

feature = feature.groupby("order_id", as_index = False).department_id.sum()
pf = pf.groupby("order_id", as_index = False).add_to_cart_order.max()
rat = feature.merge(pf, how="left", on="order_id")
rat = rat.rename(columns={"department_id":"beverages"})
column = rat["beverages"]/rat["add_to_cart_order"]
column.columns = ["ratio"]
column = column.replace(0,np.nan)
column=column.dropna(how='all',axis=0)
column=column.replace(np.nan,0)
# drop the ratio of 0

new_feature = pd.concat([rat["order_id"], column],axis=1,keys=['rat','column'])
new_feature.columns = ["order_id","ratio"]
feature1 = orders_df.merge(new_feature, how="left", on="order_id")
feature1.drop(['user_id',"order_number","order_dow","order_hour_of_day","days_since_prior_order"], inplace = True, axis =1)
feature1.fillna(0,inplace=True)
su = orders_df.merge(feature1,how="left",on="order_id")
su.drop(['order_number', 'order_dow','order_hour_of_day','days_since_prior_order'], inplace = True, axis =1)
X_train = X_train.merge(su, how='left', on=['order_id','user_id'])
X_train.fillna(0, inplace=True)
print X_train.head(10)

X_test = X_test.merge(su, how='left', on=['order_id','user_id'])
X_test.fillna(0, inplace = True)

get_prediction(X_train, X_test, y_train, y_test)


