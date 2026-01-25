import pandas as pd

df = pd.read_csv("/data/iris.csv")
print("Rows:", len(df))
print("Columns:", list(df.columns))
