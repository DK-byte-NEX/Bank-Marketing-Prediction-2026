
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# 1. CREATE OUTPUT DIRECTORIES

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 2. LOAD DATASET

df = pd.read_csv("data/bank-full.csv", sep=";")

print("\nBANK MARKETING PREDICTION")
print("=" * 50)
print("Original dataset shape:", df.shape)

# Remove duration to prevent data leakage
df = df.drop(columns=["duration"])

# 3. SEPARATE FEATURES AND TARGET

X = df.drop(columns=["y"])
y = df["y"].map({"no": 0, "yes": 1})

# Identify feature types
numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical features:", numerical_features)
print("Categorical features:", categorical_features)

# 4. SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 5. PREPROCESS DATA

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data
X_test_processed = preprocessor.transform(X_test)

# Save preprocessing pipeline
joblib.dump(preprocessor, "models/preprocessor.pkl")

print("\nPreprocessing completed!")
print("Encoded training shape:", X_train_processed.shape)
print("Encoded testing shape:", X_test_processed.shape)

# 6. DEFINE FIVE ML MODELS

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Naive Bayes": GaussianNB(),

    "SVM": SVC(
        probability=True,
        class_weight="balanced",
        random_state=42
    )
}

# 7. TRAIN AND EVALUATE MODELS

results = []

# GaussianNB requires dense input
X_train_nb = X_train_processed
X_test_nb = X_test_processed

for name, model in models.items():

    print(f"\nTraining {name}...")

    if name == "Naive Bayes":
        model.fit(X_train_nb, y_train)
        predictions = model.predict(X_test_nb)
        probabilities = model.predict_proba(X_test_nb)[:, 1]

    else:
        model.fit(X_train_processed, y_train)
        predictions = model.predict(X_test_processed)
        probabilities = model.predict_proba(X_test_processed)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, predictions)
    }

    results.append(metrics)

    # Save trained model
    filename = name.replace(" ", "_") + ".pkl"
    joblib.dump(model, f"models/{filename}")

    print(f"{name} completed!")

# 8. DISPLAY AND SAVE RESULTS

results_df = pd.DataFrame(results)

print("\nMODEL PERFORMANCE COMPARISON")
print("=" * 90)
print(results_df.to_string(index=False))

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nAll five models trained successfully!")
print("Models saved in the models folder.")
print("Comparison saved in results/model_comparison.csv")
