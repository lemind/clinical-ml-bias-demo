# clinical-ml-bias-demo

A **minimal, reproducible demo** of how bias can appear — and be mitigated — in a simple clinical machine-learning pipeline.  
Uses **synthetic patient data**, **logistic regression**, and **post-hoc fairness mitigation** to show real trade-offs.

Inspired by **Ntoutsi et al. (2020)** and built as a hands-on learning exercise.

📄 **Article:**  
https://dev.to/lemind/demonstrating-bias-and-mitigation-in-a-simple-clinical-ml-pipeline-3pc3


## What this repo shows

- How bias propagates even in a **simple, interpretable model**
- Why accuracy alone hides clinically relevant harm
- How mitigation can **reduce group-specific error** without changing the data
- Why mitigation is **not cheating**, but a policy choice


## Core experiment (numbers)

**Baseline (default threshold = 0.5):**
- Overall accuracy: **0.593**
- False Negative Rate (male): **0.468**
- False Negative Rate (female): **0.925**

**After mitigation (reweighting + group-specific threshold):**
- False Negative Rate (male): **0.468**
- False Negative Rate (female): **0.094**

> The model becomes *less likely to miss positive outcomes for females*, at the cost of different decision thresholds.


## Mitigation strategy

1. **Reweighting during training**  
   Higher sample weight for the disadvantaged group (females)

2. **Threshold mitigation at inference**  
   Different decision thresholds per group to reduce false negatives  
   (model stays the same, only decision policy changes)


## Tools & stack

- Python
- pandas
- scikit-learn
- Logistic Regression (intentionally simple)


## Setup & usage

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python src/mitigate_bias.py
