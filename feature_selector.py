import pandas as pd


def join(how, left, right, name, *keys):
    how = how #left right inner outer.
    left = pd.read_csv(left)
    right = pd.read_csv(right)
    name = name
    keys = list(keys)

    new = left.merge(right, how=how, on=keys)
    if new.isnull().values.any():
        new.fillna(0)
    return new

left = "data/order_products__prior.csv"
right = "data/products.csv"
how = 'left'
name = 'quantities.csv'
keys = 'product_id'

feature = join(how, left, right, name, keys)
collumns = ['product_name', 'aisle_id', 'add_to_cart_order']
feature.drop(collumns, inplace=True, axis=1)
print feature.head(5)
