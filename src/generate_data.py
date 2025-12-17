"""
generate_data.py

This script generates a small synthetic clinical dataset
for an educational project on bias and fairness in
machine learning.

Clinical task (toy example):
Predict whether a patient develops a complication (binary).

Intentional bias:
Female patients are less likely to be labeled as high-risk
than male patients with the same clinical severity.
"""

import numpy as np
import pandas as pd


RANDOM_SEED = 42
N_SAMPLES = 1000
OUTPUT_FILE = "data/synthetic_patients.csv"


def generate_data(n_samples: int) -> pd.DataFrame:
    np.random.seed(RANDOM_SEED)

    sex = np.random.choice(["male", "female"], size=n_samples)
    age = np.clip(np.random.normal(60, 10, n_samples), 30, 90)
    severity = np.random.normal(0, 1, n_samples)

    base_risk = 0.3 * severity + 0.02 * (age - 60)

    # Intentional bias: females are under-labeled as high risk
    bias = np.where(sex == "female", -0.6, 0.0)

    prob = 1 / (1 + np.exp(-(base_risk + bias)))
    outcome = np.random.binomial(1, prob)

    return pd.DataFrame({
        "age": age,
        "sex": sex,
        "severity": severity,
        "outcome": outcome,
    })


if __name__ == "__main__":
    df = generate_data(N_SAMPLES)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved dataset to {OUTPUT_FILE}")
