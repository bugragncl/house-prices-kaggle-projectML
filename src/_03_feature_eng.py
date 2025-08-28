import pandas as pd
import numpy as np

"""
df = pd.read_pickle("../data/processed/01_no_nan.pkl")
df
"""


# onehot ve ordinal encoder yap ordered ve nonordered columnlar için ve tüm yaptığın işlemleri fonksiyona çevir
def feature_engineering(dataframe):
    df = dataframe.copy()
    if "SalePrice" in df.columns:
        df["SalePrice_log"] = np.log1p(df["SalePrice"])
    df["MSSubClass"] = df["MSSubClass"].astype(str)

    ordered_cols = [
        "LotShape",
        "LandSlope",
        "ExterQual",
        "ExterCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "HeatingQC",
        "KitchenQual",
        "Functional",
        "FireplaceQu",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PavedDrive",
        "PoolQC",
    ]

    non_ordered_cols = (
        df.select_dtypes(include=["object"]).columns.difference(ordered_cols).tolist()
    )

    # create new features
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (
        df["FullBath"]
        + (0.5 * df["HalfBath"])
        + df["BsmtFullBath"]
        + (0.5 * df["BsmtHalfBath"])
    )
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
    )
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["TotalLivingArea"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["HasPool"] = df["PoolArea"].apply(lambda x: 1 if x > 0 else 0)
    df["Has2ndFloor"] = df["2ndFlrSF"].apply(lambda x: 1 if x > 0 else 0)
    df["HasGarage"] = df["GarageArea"].apply(lambda x: 1 if x > 0 else 0)
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    df["HasBsmt"] = df["TotalBsmtSF"].apply(lambda x: 1 if x > 0 else 0)
    df["HasFireplace"] = df["Fireplaces"].apply(lambda x: 1 if x > 0 else 0)

    # LotShape
    lotshape_map = {"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1, "don't have": 0}

    # LandSlope
    landslope_map = {"Gtl": 3, "Mod": 2, "Sev": 1, "don't have": 0}

    # ExterQual, ExterCond, BsmtQual, BsmtCond, HeatingQC, KitchenQual, FireplaceQu, GarageQual, GarageCond, PoolQC
    qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "don't have": 0}

    # BsmtExposure
    bsmtexposure_map = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "don't have": 0}

    # BsmtFinType1, BsmtFinType2
    bsmtfintype_map = {
        "GLQ": 6,
        "ALQ": 5,
        "BLQ": 4,
        "Rec": 3,
        "LwQ": 2,
        "Unf": 1,
        "don't have": 0,
    }

    # Functional
    functional_map = {
        "Typ": 7,
        "Min1": 6,
        "Min2": 5,
        "Mod": 4,
        "Maj1": 3,
        "Maj2": 2,
        "Sev": 1,
        "Sal": 0,
        "don't have": 0,
    }

    # GarageFinish
    garagefinish_map = {"Fin": 3, "RFn": 2, "Unf": 1, "don't have": 0}

    # PavedDrive
    paveddrive_map = {"Y": 3, "P": 2, "N": 1, "don't have": 0}

    for col in non_ordered_cols:
        df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)

    for col in ordered_cols:
        if col == "LotShape":
            df[col] = df[col].map(lotshape_map)
        elif col == "LandSlope":
            df[col] = df[col].map(landslope_map)
        elif col in [
            "ExterQual",
            "ExterCond",
            "BsmtQual",
            "BsmtCond",
            "HeatingQC",
            "KitchenQual",
            "FireplaceQu",
            "GarageQual",
            "GarageCond",
            "PoolQC",
        ]:
            df[col] = df[col].map(qual_map)
        elif col == "BsmtExposure":
            df[col] = df[col].map(bsmtexposure_map)
        elif col in ["BsmtFinType1", "BsmtFinType2"]:
            df[col] = df[col].map(bsmtfintype_map)
        elif col == "Functional":
            df[col] = df[col].map(functional_map)
        elif col == "GarageFinish":
            df[col] = df[col].map(garagefinish_map)
        elif col == "PavedDrive":
            df[col] = df[col].map(paveddrive_map)

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    if "SalePrice" in df.columns:
        df.drop("SalePrice", axis=1, inplace=True)

    return df


"""
df = feature_engineering(df)

df.info()

df.to_pickle("../data/processed/03_feature_eng.pkl")
"""
