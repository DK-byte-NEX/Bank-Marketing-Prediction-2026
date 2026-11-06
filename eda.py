import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/bank-full.csv", sep=";")

#1. Statistical analysis
print("\nSTATISTICAL SUMMARY")
print(df.describe())

#2. Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

#3. Duplicate records
print("\nDUPLICATES")
print(df.duplicated().sum())

#4. Unknown values
print("\nUNKNOWN VALUES")
print((df == "unknown").sum())

#5. Target distribution
print("\nTARGET DISTRIBUTION")
print(df["y"].value_counts(normalize=True)*100)

#6. Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Customer Age Distribution")
plt.savefig("resultsage_distribution.png")
plt.close()

#7. Target distribution graph
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="y")
plt.title("Term Deposit Subscription")
plt.savefig("results/target_distribution.png")
plt.close()

print("\nEDA COMPLETED SUCCESSFULLY")