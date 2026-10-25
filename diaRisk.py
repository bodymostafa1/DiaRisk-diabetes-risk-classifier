
# DiaRisk: Diabetes Classification using Neural Networks

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. Load Dataset
# ==========================================

data_path = Path("diabetes.csv")

# Fallback for the uploaded file's environment
if not data_path.exists():
    data_path = Path("/mnt/data/diabetes.csv")

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nTarget distribution:")
print(df["Outcome"].value_counts())

# ==========================================
# 2. Prepare Features and Target
# ==========================================

X = df.drop(columns=["Outcome"]).copy()
y = df["Outcome"].astype(int)

# Validate the binary target
if set(y.unique()) != {0, 1}:
    raise ValueError("Outcome must contain both classes: 0 and 1.")

# In this dataset, zero can represent a missing measurement
# for these clinical variables. Zero pregnancies is valid.
zero_missing_cols = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in zero_missing_cols:
    if col in X.columns:
        X[col] = X[col].replace(0, np.nan)

print("\nFeatures:", X.columns.tolist())
print("Number of features:", X.shape[1])

# ==========================================
# 3. Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 4. Build Neural Network Pipeline
# ==========================================

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),

    ("scaler", StandardScaler()),

    ("classifier", MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=42
    ))
])

# ==========================================
# 5. Train Model
# ==========================================

model.fit(X_train, y_train)

print("\nModel training completed!")

# ==========================================
# 6. Make Predictions
# ==========================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# 7. Evaluate Model
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_prob)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {auc_score:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Diabetes", "Diabetes"],
        zero_division=0
    )
)

# ==========================================
# 8. Visualization 1: Training Loss Curve
# ==========================================

mlp = model.named_steps["classifier"]

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(mlp.loss_curve_) + 1),
    mlp.loss_curve_,
    marker="."
)

plt.title("Neural Network Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 9. Visualization 2: ROC Curve
# ==========================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 5))
plt.plot(
    fpr,
    tpr,
    label=f"MLP Classifier (AUC = {auc_score:.3f})"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 10. Visualization 3: Confusion Matrix
# ==========================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=[0, 1],
    display_labels=["No Diabetes", "Diabetes"],
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ==========================================
# 11. Save Trained Pipeline
# ==========================================

model_path = "diabetes_mlp_classifier.pkl"

joblib.dump(model, model_path)

print(f"\nModel saved successfully: {model_path}")
