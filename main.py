from feature_selector import join
from initial_data import phase
from prediction import get_prediction # when we are ready to test it.
import pandas as pd

X_train, X_test, y_train, y_test = phase("testing")
order_products_priod_df = pd.read_csv("data/order_products__prior.csv")
products_df = pd.read_csv("data/products.csv")

params = {
    'how': 'left',
    'key': 'product_id',
    'columns': ['aisle_id', 'add_to_cart_order','product_id','product_name'],
    'name': None
}


feature = join(order_products_priod_df, products_df, params)


fun = lambda x: 1 if x == 7 else 0
feature["department_id"]= feature['department_id'].apply(fun) # i turn every non department 7 id to 0 and every 7 to 1

feature = feature[feature["department_id"]==1]


X_train = X_train.merge(feature, how='left', on='order_id')
X_train.fillna(0, inplace=True)

X_test = X_test.merge(feature, how='left', on='order_id')
X_test.fillna(0, inplace = True)
get_prediction(X_train, X_test,y_train,y_test)