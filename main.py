from feature_selector import join
from initial_data import phase
from prediction import get_prediction # when we are ready to test it.
import pandas as pd
import numpy as np

X_train, X_test, y_train, y_test = phase("testing")
order_products_priod_df = pd.read_csv("data/order_products__prior.csv")
products_df = pd.read_csv("data/products.csv")

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
print feature.head(5)

pf = pf.groupby("order_id", as_index = False).add_to_cart_order.max()
print pf.head(5)

rat = feature.merge(pf, how="left", on="order_id")

rat = rat.rename(columns={"department_id":"beverages"})
print rat.head(15)

# x = raw_input(" ")
#
# print feature.head(6)
# X_train = X_train.merge(feature, how='left', on='order_id')
# X_train.fillna(0, inplace=True)
# print X_train.head(6)
#
# X_test = X_test.merge(feature, how='left', on='order_id')
# X_test.fillna(0, inplace = True)
#
# get_prediction(X_train, X_test, y_train, y_test)
# #clear the memory
# del products_df
# ###
#
