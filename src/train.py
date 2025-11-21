import pandas as pd
import yaml
import json
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib

def main():
    # Load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    input_data = params["input"]
    model_output = params["model_output"]
    metrics_output = params["metrics_output"]
    target = params["target"]

    print(f"📥 Loading processed dataset: {input_data}")
    df = pd.read_csv(input_data)

    # Remove non-informative identifiers
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    X = df.drop(columns=[target])
    y = df[target]

    # Detect categorical vs numerical features
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()
    num_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    model = LogisticRegression(
        solver=params["solver"],
        C=params["C"],
        max_iter=params["max_iter"]
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=42
    )

    mlflow.set_experiment("baseline-telco")

    with mlflow.start_run(run_name="logreg-baseline"):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.log_params(params)

        os.makedirs(os.path.dirname(model_output), exist_ok=True)
        joblib.dump(pipeline, model_output)
        mlflow.sklearn.log_model(pipeline, "model")

        with open(metrics_output, "w") as f:
            json.dump(metrics, f, indent=4)

        mlflow.log_artifact(metrics_output)

        print(f"📊 Metrics: {metrics}")
        print(f"💾 Model saved to: {model_output}")

if __name__ == "__main__":
    main()
