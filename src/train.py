import argparse
import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump
from data_loader import load_csv
from preprocessing import Preprocessor

def train(data_path, model_path):
    df = load_csv(data_path)
    X = df["x"].values
    y = df["y"].values

    prep = Preprocessor()
    Xs, ys = prep.fit_transform(X, y)

    model = LinearRegression()
    model.fit(Xs, ys)

    dump({"model": model, "prep": prep}, model_path)
    print(f"Modelo salvo em {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    train(args.data, args.model)
