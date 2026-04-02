import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


def fetch_processor(cpu_name):
    if cpu_name in ("Intel Core i7", "Intel Core i5", "Intel Core i3"):
        return cpu_name
    if cpu_name.split()[0] == "Intel":
        return "Other Intel Processor"
    if cpu_name.split()[0] == "AMD":
        return "AMD Processor"
    return "Other Intel Processor"


def cat_os(text):
    if text in ("Windows 10", "Windows 7", "Windows 10 S"):
        return "Windows"
    if text in ("macOS", "Mac OS X"):
        return "Mac"
    return "Others/No OS/Linux"


def prepare_data(csv_path="laptop_data.csv"):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Unnamed: 0"]).copy()

    df["Ram"] = df["Ram"].str.replace("GB", "", regex=False).astype("int32")
    df["Weight"] = df["Weight"].str.replace("kg", "", regex=False).astype("float32")
    df["Touchscreen"] = df["ScreenResolution"].apply(lambda value: 1 if "Touchscreen" in value else 0)
    df["Ips"] = df["ScreenResolution"].apply(lambda value: 1 if "IPS" in value else 0)

    resolution = df["ScreenResolution"].str.split("x", n=1, expand=True)
    df["X_res"] = resolution[0].str.replace(",", "", regex=False).str.extract(r"(\d+\.?\d+)")[0].astype(int)
    df["Y_res"] = resolution[1].astype(int)
    df["ppi"] = (((df["X_res"] ** 2) + (df["Y_res"] ** 2)) ** 0.5 / df["Inches"]).astype(float)
    df = df.drop(columns=["ScreenResolution", "Inches", "X_res", "Y_res"])

    df["Cpu Name"] = df["Cpu"].apply(lambda value: " ".join(value.split()[0:3]))
    df["Cpu brand"] = df["Cpu Name"].apply(fetch_processor)
    df = df.drop(columns=["Cpu", "Cpu Name"])

    memory = df["Memory"].astype(str).replace(r"\.0", "", regex=True)
    memory = memory.str.replace("GB", "", regex=False)
    memory = memory.str.replace("TB", "000", regex=False)
    split_memory = memory.str.split("+", n=1, expand=True)

    df["first"] = split_memory[0]
    df["second"] = split_memory[1].fillna("0")

    df["Layer1HDD"] = df["first"].apply(lambda value: 1 if "HDD" in value else 0)
    df["Layer1SSD"] = df["first"].apply(lambda value: 1 if "SSD" in value else 0)
    df["Layer1Hybrid"] = df["first"].apply(lambda value: 1 if "Hybrid" in value else 0)
    df["Layer1Flash_Storage"] = df["first"].apply(lambda value: 1 if "Flash Storage" in value else 0)

    df["Layer2HDD"] = df["second"].apply(lambda value: 1 if "HDD" in value else 0)
    df["Layer2SSD"] = df["second"].apply(lambda value: 1 if "SSD" in value else 0)
    df["Layer2Hybrid"] = df["second"].apply(lambda value: 1 if "Hybrid" in value else 0)
    df["Layer2Flash_Storage"] = df["second"].apply(lambda value: 1 if "Flash Storage" in value else 0)

    df["first"] = df["first"].str.replace(r"\D", "", regex=True)
    df["second"] = df["second"].str.replace(r"\D", "", regex=True)
    df["first"] = df["first"].replace("", "0").astype(int)
    df["second"] = df["second"].replace("", "0").astype(int)

    df["HDD"] = (df["first"] * df["Layer1HDD"] + df["second"] * df["Layer2HDD"])
    df["SSD"] = (df["first"] * df["Layer1SSD"] + df["second"] * df["Layer2SSD"])
    df["Hybrid"] = (df["first"] * df["Layer1Hybrid"] + df["second"] * df["Layer2Hybrid"])
    df["Flash_Storage"] = (
        df["first"] * df["Layer1Flash_Storage"] + df["second"] * df["Layer2Flash_Storage"]
    )

    df = df.drop(
        columns=[
            "Memory",
            "first",
            "second",
            "Layer1HDD",
            "Layer1SSD",
            "Layer1Hybrid",
            "Layer1Flash_Storage",
            "Layer2HDD",
            "Layer2SSD",
            "Layer2Hybrid",
            "Layer2Flash_Storage",
            "Hybrid",
            "Flash_Storage",
        ]
    )

    df["Gpu brand"] = df["Gpu"].apply(lambda value: value.split()[0])
    df = df.drop(columns=["Gpu"])

    df["os"] = df["OpSys"].apply(cat_os)
    df = df.drop(columns=["OpSys"])

    return df


def build_pipeline():
    try:
        encoder = OneHotEncoder(sparse_output=False, drop="first")
    except TypeError:
        encoder = OneHotEncoder(sparse=False, drop="first")

    preprocessor = ColumnTransformer(
        transformers=[
            ("col_tnf", encoder, [0, 1, 7, 10, 11]),
        ],
        remainder="passthrough",
    )

    estimators = [
        (
            "rf",
            RandomForestRegressor(
                n_estimators=350,
                random_state=3,
                max_samples=0.5,
                max_features=0.75,
                max_depth=15,
            ),
        ),
        ("gbdt", GradientBoostingRegressor(n_estimators=100, max_features=0.5)),
        ("xgb", XGBRegressor(n_estimators=25, learning_rate=0.3, max_depth=5)),
    ]

    model = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100))

    return Pipeline(
        [
            ("step1", preprocessor),
            ("step2", model),
        ]
    )


def main():
    df = prepare_data()

    X = df.drop(columns=["Price"])
    y = np.log(df["Price"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    with open("df.pkl", "wb") as df_file:
        pickle.dump(df, df_file)

    with open("pipe.pkl", "wb") as pipe_file:
        pickle.dump(pipe, pipe_file)

    score = pipe.score(X_test, y_test)
    print(f"Training complete. Test R^2: {score:.4f}")


if __name__ == "__main__":
    main()
