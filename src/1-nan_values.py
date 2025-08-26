import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/raw/train.csv")
df.info()


# print nan count for all columns
for col in df.columns:
    if df[col].isna().sum() > 0:
        print(f"{col}: {df[col].isna().sum()}")


# plot all numeric columns
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    plt.figure(figsize=(10, 5))
    plt.scatter(df[col], df["SalePrice"], alpha=0.5)
    plt.title(f"SalePrice vs {col}")
    plt.xlabel(col)
    plt.ylabel("SalePrice")
    plt.grid()
    plt.show()


def fill_nan_values(df):
    # Fill NaN values for categorical columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("don't have")

    # Fill NaN values for numerical columns
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        if 0 in df[col].unique():
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(df[col].median())

    return df


df = fill_nan_values(df)
df.info()

for col in df.select_dtypes(include=["int64", "float64"]).columns:
    plt.figure(figsize=(10, 5))
    plt.scatter(df[col], df["SalePrice"], alpha=0.5)
    plt.title(f"SalePrice vs {col}")
    plt.xlabel(col)
    plt.ylabel("SalePrice")
    plt.grid()
    plt.show()

df.to_pickle("../data/processed/01_no_nan.pkl")
