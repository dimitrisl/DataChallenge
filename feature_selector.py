import pandas as pd


def join(how, left, right, collumns, keys, name=None):
    new = left.merge(right, how=how, on=keys)
    if new.isnull().values.any():
        new.fillna(0)
    new.drop(collumns, inplace = True, axis =1)
    if name:
        new.to_csv(name,index = False)
    return new
