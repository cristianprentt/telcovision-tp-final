import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
import joblib
import os

def main():
    # Paths
    model_path = "models/model.pkl"
    data_path = "data/processed/telco_clean.csv"

    print("Loading model and data...")
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    # Predictions / probabilities
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Active", "Churn"])
    disp.plot()
    plt.title("Matriz de Confusión")
    plt.savefig("models/confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend()
    plt.savefig("models/roc_curve.png")
    plt.close()

    print(f"AUC: {roc_auc:.4f}")
    print("Archivos guardados en: models/")

if __name__ == "__main__":
    main()

