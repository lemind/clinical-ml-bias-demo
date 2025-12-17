import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Load synthetic clinical data
# -----------------------------
df = pd.read_csv("data/synthetic_patients.csv")

# Protected attribute encoding
# 0 = male, 1 = female
df["sex_encoded"] = df["sex"].map({"male": 0, "female": 1})

X = df[["age", "sex_encoded", "severity"]]
y = df["outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# Mitigation step 1: reweighting
# -----------------------------
# Give higher importance to female samples during training
weights = X_train["sex_encoded"].apply(
    lambda x: 2.0 if x == 1 else 1.0
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train, sample_weight=weights)

# -----------------------------
# Baseline predictions (0.5 threshold)
# -----------------------------
X_test = X_test.copy()
X_test["pred"] = model.predict(X_test)

print("=== Baseline (default threshold) ===")
print("Overall accuracy:", accuracy_score(y_test, X_test["pred"]))

for sex_value, label in [(0, "male"), (1, "female")]:
    subset = X_test[X_test["sex_encoded"] == sex_value]

    fn = ((subset["pred"] == 0) & (y_test.loc[subset.index] == 1)).sum()
    tp_fn = (y_test.loc[subset.index] == 1).sum()
    fnr = fn / tp_fn if tp_fn > 0 else 0

    print(f"False Negative Rate ({label}): {fnr:.3f}")

# -----------------------------
# Mitigation step 2: thresholding
# -----------------------------
# Use predicted probabilities to adjust decision threshold
probs = model.predict_proba(
    X_test[["age", "sex_encoded", "severity"]]
)[:, 1]
X_test["prob"] = probs

def apply_threshold(row):
    # Females: lower threshold to reduce missed positive cases
    if row["sex_encoded"] == 1:
        return int(row["prob"] >= 0.3)
    else:
        # Males: default threshold
        return int(row["prob"] >= 0.5)

X_test["pred_thresh"] = X_test.apply(apply_threshold, axis=1)

print("\n=== After threshold mitigation ===")
for sex_value, label in [(0, "male"), (1, "female")]:
    subset = X_test[X_test["sex_encoded"] == sex_value]

    fn = ((subset["pred_thresh"] == 0) & (y_test.loc[subset.index] == 1)).sum()
    tp_fn = (y_test.loc[subset.index] == 1).sum()
    fnr = fn / tp_fn if tp_fn > 0 else 0

    print(f"False Negative Rate ({label}): {fnr:.3f}")
