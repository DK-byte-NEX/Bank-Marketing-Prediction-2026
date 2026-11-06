import pandas as pd

#Load dataset
df = pd.read_csv("data/bank-full.csv", sep=";")

print("=" * 50)
print("BANK MARKETING DATASET ANALYSIS")
print("=" * 50)


print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df["y"].value_counts())

print("\nDataset Information:")
print(df.info())