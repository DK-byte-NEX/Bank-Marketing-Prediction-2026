
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    roc_auc_score
)

os.makedirs("results", exist_ok=True)

# Load dataset
df = pd.read_csv("data/bank-full.csv", sep=";")

# Apply the same data preparation used during training
df = df.drop(columns=["duration"])

X = df.drop(columns=["y"])
y = df["y"].map({"no": 0, "yes": 1})

# Recreate the same train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load saved preprocessor
preprocessor = joblib.load("models/preprocessor.pkl")

# Transform test data
X_test_processed = preprocessor.transform(X_test)

# Define saved model paths
model_files = {
    "Logistic Regression": "models/Logistic_Regression.pkl",
    "Decision Tree": "models/Decision_Tree.pkl",
    "KNN": "models/KNN.pkl",
    "Naive Bayes": "models/Naive_Bayes.pkl",
    "SVM": "models/SVM.pkl"
}

results = []

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

plt.suptitle("Confusion Matrices - Bank Marketing", fontsize=16)

plt.figure(figsize=(10, 7))

# Evaluate each saved model
with open("results/classification_reports.txt", "w") as report_file:

    for index, (name, path) in enumerate(model_files.items()):

        print(f"\nEvaluating {name}...")

        model = joblib.load(path)

        # Naive Bayes requires dense input
        if name == "Naive Bayes":
            X_test_model = X_test_processed.toarray() if hasattr(
                X_test_processed, "toarray"
            ) else X_test_processed
        else:
            X_test_model = X_test_processed

        # Generate predictions
        predictions = model.predict(X_test_model)
        probabilities = model.predict_proba(X_test_model)[:, 1]

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, predictions)

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No", "Yes"]
        )

        display.plot(ax=axes[index], colorbar=False)
        axes[index].set_title(name)

        # Generate classification report
        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Subscription", "Subscription"],
            zero_division=0
        )

        print(report)

        report_file.write(f"\n{name}\n")
        report_file.write(report)
        report_file.write("\n" + "-" * 50 + "\n")

        # Generate ROC curve
        fpr, tpr, thresholds = roc_curve(
            y_test, probabilities
        )

        auc_score = roc_auc_score(y_test, probabilities)

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc_score:.3f})"
        )

        results.append({
            "Model": name,
            "AUC": auc_score
        })

        print(f"{name} evaluation completed!")

# Save confusion matrices
fig.delaxes(axes[5])
fig.tight_layout()
fig.savefig("results/confusion_matrices.png", dpi=300)
plt.close(fig)

# Complete ROC curve
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)

plt.savefig("results/roc_curves.png", dpi=300)
plt.close()

# Save evaluation results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/evaluation_results.csv",
    index=False
)

print("\nMODEL EVALUATION COMPLETED!")
print(results_df.to_string(index=False))

print("\nEvaluation files saved in results folder.")
