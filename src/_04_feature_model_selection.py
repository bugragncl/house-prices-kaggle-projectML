import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import numpy as np

# df = pd.read_pickle("../data/processed/03_feature_eng.pkl")


def feature_selection(dataframe):
    df = dataframe.copy()

    X = df.drop("SalePrice_log", axis=1)
    y = df["SalePrice_log"]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    selected_features_rf = importances.head(30).index.tolist()

    return selected_features_rf


"""
selected_features = feature_selection(df)

selected_features
"""


def train_best_model_from_df(
    df,
    selected_features,
    target="SalePrice_log",
    cv=5,
    scoring="neg_root_mean_squared_error",
):
    """
    df: Dataframe
    selected_features: Önceden seçilmiş feature listesi
    target: hedef değişken
    cv: cross-validation fold sayısı
    scoring: skorlama metodu
    """

    X = df[selected_features]
    y = df[target]

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5),
        "RandomForest": RandomForestRegressor(
            n_estimators=500, max_depth=10, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42
        ),
    }

    results = []

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        rmse = -np.mean(scores)  # sklearn neg_root_mean_squared_error döndürür
        results.append({"Model": name, "RMSE": rmse})

    results_df = pd.DataFrame(results).sort_values("RMSE")

    # En iyi modeli seç
    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]

    # En iyi modeli tüm train set ile eğit
    return best_model
