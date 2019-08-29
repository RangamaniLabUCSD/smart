# ====================================================
# General python
import pandas as pd

# pandas
class ref:
    """
    Pandas dataframe doesn't play nice with dolfin indexed functions since it will try
    to take the length but dolfin will not return anything. We use ref() to trick pandas
    into treating the dolfin indexed function as a normal object rather than something
    with an associated length
    """
    def __init__(self, obj): self.obj = obj
    def get(self):    return self.obj
    def set(self, obj):      self.obj = obj

def insert_dataframe_col(df, columnName, columnNumber=None, items=None):
    """
    pandas requires some weird manipulation to insert some things, e.g. dictionaries, as dataframe entries
    items is a list of things
    by default, will insert a column of empty dicts as the last column
    """
    if items == None:
        items = [{} for x in range(df.shape[0])]
    if columnNumber == None:
        columnNumber = df.columns.size
    df.insert(columnNumber, columnName, items)

def nan_to_none(df):
    return df.replace({pd.np.nan: None})


