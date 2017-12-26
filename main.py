from feature_selector import join
from initial_data import phase
from prediction import get_prediction # when we are ready to test it.
import pandas as pd

X_train, X_test, y_train, y_test = phase("testing")

order_products_priod_df = pd.read_csv("data/order_products__prior.csv")
products_df = pd.read_csv("data/products.csv")
how = 'left'
keys = 'product_id'

columns = ['product_name', 'aisle_id', 'add_to_cart_order','department_id','product_id']
feature = join(how, order_products_priod_df, products_df, columns, keys)

print feature.head(5)

X_train = X_train.merge(feature, how=how, on='order_id')

if X_train.isnull().values.any():
    X_train.fillna(0)
get_prediction(X_train, X_test, y_train, y_test)