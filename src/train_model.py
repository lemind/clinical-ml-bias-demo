import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load synthetic clinical data
df = pd.read_csv("data/synthetic_patients.csv")

# Encode protected attribute for model input
df["sex_encoded"] = df["sex"].map({"male": 0, "female": 1})

# Features and target
X = df[["age", "sex_encoded", "severity"]]
y = df["outcome"]

# Train/test split for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train simple, interpretable model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Generate predictions on held-out data
X_test = X_test.copy()
X_test["pred"] = model.predict(X_test)

# Overall model performance
print("Overall accuracy:", accuracy_score(y_test, X_test["pred"]))

# Performance by sex to expose potential bias
for sex_value, label in [(0, "male"), (1, "female")]:
    subset = X_test[X_test["sex_encoded"] == sex_value]
    acc = accuracy_score(
        y_test.loc[subset.index],
        subset["pred"]
    )
    print(f"Accuracy ({label}): {acc:.3f}")

from sklearn.metrics import confusion_matrix


def false_negative_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp)


for sex_value, label in [(0, "male"), (1, "female")]:
    subset = X_test[X_test["sex_encoded"] == sex_value]
    y_true = y_test.loc[subset.index]
    y_pred = subset["pred"]

    fnr = false_negative_rate(y_true, y_pred)
    print(f"False Negative Rate ({label}): {fnr:.3f}")

