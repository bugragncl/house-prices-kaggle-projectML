from _01_nan_values import fill_nan_values
from _03_feature_eng import feature_engineering
from _04_feature_model_selection import train_best_model_from_df, feature_selection
import pandas as pd
import numpy as np

df_train = pd.read_csv("../data/raw/train.csv")
df_test = pd.read_csv("../data/raw/test.csv")


def from_raw_to_predict(df_train, df_test):
    df_train = fill_nan_values(df_train)
    df_test = fill_nan_values(df_test)

    df_train = feature_engineering(df_train)
    df_test = feature_engineering(df_test)

    selected_features = feature_selection(df_train)

    best_model = train_best_model_from_df(df_train, selected_features)
    X = df_train[selected_features]
    y = df_train["SalePrice_log"]

    best_model.fit(X, y)

    predictions_log = best_model.predict(df_test[selected_features])

    predictions = np.expm1(predictions_log)

    return predictions


predictions = from_raw_to_predict(df_train, df_test)

submission = pd.DataFrame({"Id": df_test["Id"], "SalePrice": predictions})

submission.to_csv("submission01.csv", index=False)
