# import pandas as pd
# df = pd.read_csv("data/raw/diabetes.csv")
# print(df.head())
# print(df.shape())
# print(df.columns.tolist())
import pandas as pd

# Load diabetes dataset
df = pd.read_csv("data/raw/diabetes.csv")

# Display first five records
print(df.head())

# Display number of rows and columns
print(df.shape)

# Display column names
print(df.columns.tolist())