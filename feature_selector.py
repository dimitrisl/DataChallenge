import pandas as pd


def join(left, right, params):

    how = params.get('how')
    name = params.get('name')
    keys = params.get('keys')
    columns = params.get('columns')
    new = left.merge(right, how=how, on=keys)
    if new.isnull().values.any():
        new.fillna(0)
    new.drop(columns, inplace = True, axis =1)
    if name:
        new.to_csv(name,index = False)
    return new
