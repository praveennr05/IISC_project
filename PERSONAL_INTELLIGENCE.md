# Personal Intelligence Note & AI Use Declaration
**Candidate Personal Seed:** `S = 36` (Derived from USN ending in `036`)  
**Domain:** AI for Personal Health and Wellness  
**Questions Attempted:** Question A (Risk Prediction) & Question B (Turn Model into Usable App)  
**Date:** October 6, 2026  

---

## 1. Decision Log

### Question A: Predict a Health Risk

#### Decision 1: Feature Standardization Strategy (Z-score Standardization vs. Min-Max Normalization)
- **Chosen:** Standard Z-score Normalization ($z = \frac{x - \mu}{\sigma}$) fit strictly on training split and applied to test split.
- **Rejected:** Min-Max Scaling ($[0, 1]$ range).
- **Rationale & Empirical Evidence:** In our custom NumPy Logistic Regression (Level 2), batch gradient descent calculates $\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y)$. Features in the UCI Heart Disease dataset have disparate scales (e.g., `chol` ranges up to 564 mg/dl, whereas `oldpeak` ranges from 0.0 to 6.2). Min-Max scaling is vulnerable to extreme clinical outliers (e.g., severe hypercholesterolemia). Standard scaling produced smooth gradient contours, preventing oscillatory divergence and achieving gradient convergence in 1,500 iterations with learning rate $\alpha = 0.1$, matching Scikit-Learn's accuracy to 4 decimal places ($75.41\%$).

#### Decision 2: Decision Threshold Selection for Clinical Screening ($\theta = 0.25$ vs. Default $\theta = 0.50$)
- **Chosen:** Lowered screening threshold ($\theta = 0.25 - 0.30$) for primary triage.
- **Rejected:** Default symmetrical threshold ($\theta = 0.50$).
- **Rationale & Empirical Evidence:** At $\theta = 0.50$, the baseline Logistic Regression model achieves $75.41\%$ accuracy but exhibits a recall of only $71.43\%$, missing 8 out of 28 cardiac patients (False Negatives). In clinical screening, missing an active heart condition can be fatal, while a False Positive results only in secondary non-invasive testing (e.g., echocardiogram). Lowering the threshold to $0.25$ increases recall to $78.57\%$ (and $\theta \approx 0.081$ achieves $>92.86\%$ recall).

---

### Question B: Turn a Model into a Usable App

#### Decision 1: Backend Architecture & Concurrency Model (FastAPI Async + Raw SQLite with WAL Mode vs. Flask + Heavy ORM)
- **Chosen:** Asynchronous FastAPI backend utilizing Python's built-in `sqlite3` with Write-Ahead Logging (`PRAGMA journal_mode=WAL;`) and pure handwritten SQL queries.
- **Rejected:** Synchronous Flask with SQLAlchemy ORM.
- **Rationale & Empirical Evidence:** FastAPI natively runs on `uvicorn` with an asynchronous event loop, handling non-blocking I/O. For Level 2 requirements, handwritten SQL eliminates ORM overhead and guarantees compliance with the prompt's strict "no ORM" constraint for `/stats`. WAL mode enables simultaneous concurrent readers while writes execute, ensuring high throughput under concurrent load.

#### Decision 2: Input Validation Layer (Pydantic Field Constraints with Custom Clinical Sanitizers vs. Generic Post-Validation)
- **Chosen:** Declarative Pydantic v2 schemas with explicit medical boundaries (e.g., `age`: 1–120, `trestbps`: 50–300 mm Hg, `chol`: 100–600 mg/dl).
- **Rejected:** Generic try-except blocks inside endpoint handlers.
- **Rationale & Empirical Evidence:** Pydantic intercepts malformed, out-of-range, and type-mismatched requests at the API boundary, returning standardized HTTP 422 Unprocessable Entity responses with structured error messages before any model inference or database transaction occurs.

---

## 2. Pre-Run Predictions (Committed to Git Before Running Level 3 Tests)

> **Integrity Notice:** As required by Section 4, Item 2, the following predictions are committed to the repository before the Level 3 evaluation scripts are executed.

### Question A (Level 3 Prediction)
- **Hypothesis / Prediction:**  
  When lowering the decision threshold $\theta$ below $0.50$ to drive Recall up to $\ge 0.90$:
  1. **Recall** will increase monotonically from its baseline ($71.43\%$) toward $100\%$ as fewer positive patients are misclassified as negative ($FN \to 0$).
  2. **Precision** will **decrease significantly**. By lowering the threshold to capture borderline positive cases, more healthy individuals will cross the lower cutoff, increasing False Positives ($FP$).
  3. **Predicted Precision Drop:** Precision is predicted to drop from its baseline of $74.07\%$ down into the $50\% - 60\%$ range when recall reaches $0.90$.
  4. **Accuracy:** Overall accuracy will decrease because the influx of False Positives outweighs the modest decrease in False Negatives in a balanced test sample.

### Question B (Level 3 Prediction)
- **Hypothesis / Prediction for Fault Injection:**
  1. **Fault 1 (Missing Model Artifact):** If `heart_model.joblib` or `scaler.joblib` is deleted or unreadable at startup, the app would fail with `FileNotFoundError` or crash on the first `/predict` request. The robust fix will catch startup errors, provide fallback mock inference or graceful HTTP 503 Service Unavailable, and expose a `/health` endpoint indicating degraded status.
  2. **Fault 2 (Malformed Input / Text in Numeric Field):** Sending string values (e.g., `{"age": "twenty"}`) to a numeric endpoint will cause unhandled internal conversion errors if raw dictionaries are parsed. With Pydantic validation, the app will reject invalid types upfront with HTTP 422 and actionable field-level diagnostics.

---

## 3. AI Usage Declaration

- **AI Tools Consulted:** Claude, ChatGPT, Gemini.
- **Intended Purpose:** Rapid generation of boilerplate web layout (HTML/CSS layout for frontend) and baseline Scikit-Learn training syntax for Level 1.
- **Where AI was Weak or Wrong & How It Was Resolved:**
  - *Identified Flaw:* During initial prompt generation for Question A Level 2 (Logistic Regression from scratch), the AI generated a gradient descent loop without feature normalization and applied an unregularized gradient step with a large learning rate ($\alpha = 0.5$). When executed on raw clinical data (`chol` $\sim 250$, `age` $\sim 55$), the dot product $Xw$ exploded into numerical overflow in the exponential function ($e^{-z} \to \infty$), causing `RuntimeWarning: overflow encountered in exp` and resulting in `NaN` weights.
  - *My Engineering Fix:* I identified that logistic regression gradient descent without feature normalization has elliptical, highly eccentric loss contours. I implemented an explicit Z-score standardization step ($\mu=0, \sigma=1$) fit on training data, clipped $z$ within $[-500, 500]$ inside the sigmoid function, and tuned the learning rate to $\alpha = 0.1$ with $1,500$ iterations. This stabilized the loss trajectory and produced an accuracy of $75.41\%$, exactly matching Scikit-Learn.

---

## 4. Post-Run Empirical Verification & Evaluation

### Question A: Hypothesis vs. Empirical Verification

| Evaluation Parameter | Baseline Cutoff ($\theta = 0.50$) | Tuned Cutoff ($\theta = 0.0800$) | Pre-Run Prediction | Empirical Reality | Prediction Status |
|:---------------------|:----------------------------------|:----------------------------------|:-------------------|:------------------|:------------------|
| **Recall (Sensitivity)** | $71.43\%$ ($20/28$) | $\mathbf{92.86\%}$ ($26/28$) | Monotonic increase $\ge 90\%$ | Reached $92.86\%$ | **CONFIRMED** |
| **Precision (PPV)** | $74.07\%$ ($20/27$) | $\mathbf{56.52\%}$ ($26/46$) | Fall into $50\% - 60\%$ range | Dropped to $56.52\%$ | **CONFIRMED** |
| **Overall Accuracy** | $75.41\%$ ($46/61$) | $\mathbf{63.93\%}$ ($39/61$) | Decrease due to added FPs | Decreased by $11.48\%$ | **CONFIRMED** |
| **False Negatives (Missed)** | $8$ patients | $\mathbf{2}$ patients | Drastic reduction ($FN \to 0$) | Misses cut by $75\%$ ($8 \to 2$) | **CONFIRMED** |
| **False Positives (Alerts)** | $7$ patients | $\mathbf{20}$ patients | Significant increase | Increased by $13$ patients | **CONFIRMED** |

### Question B: Fault Injection Empirical Verification

1. **Breakage 1 (Missing Model):**  
   - *Simulated Condition:* `MODEL_PATH` temporarily renamed/missing.
   - *Observation:* `GET /health` returned `200 OK` with `"model_loaded": false`. `POST /predict` returned structured `HTTP 503 SERVICE UNAVAILABLE` with diagnostic message `"Trained model artifact not available on server..."`.
   - *Conclusion:* Server process remained 100% stable without unhandled exception crashes.

2. **Breakage 2 (Text in Numeric Field):**  
   - *Simulated Condition:* Sent payload with `{"age": "fifty-five", "chol": "extremely_high"}`.
   - *Observation:* Intercepted cleanly by Pydantic; returned `HTTP 422 UNPROCESSABLE ENTITY` with field-level diagnostic pointers.
   - *Conclusion:* Zero unhandled 500 errors; internal Python tracebacks completely shielded from clients.

