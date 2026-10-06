# Demo Video Guide & Walkthrough Script (3 – 5 Minutes)

**Target Email:** `mnaveennk@iisc.ac.in`  
**Email Subject:** `B.E. Assignment – <Full Name> – <USN>`  
**File Name:** `<FullName>_Assignment_Demo.mp4`  
**Candidate Seed:** `S = 36` (USN ending in `036`)  
**Sharing Setting:** Anyone with the link can view (Google Drive)  

---

## 1. Video Recording Setup Checklist

- [ ] **Microphone & Screen:** Ensure microphone is clear. Record screen at 1080p.
- [ ] **Windows / Tabs Ready:**
  1. Terminal / VS Code showing project files.
  2. Browser open at `http://127.0.0.1:8000` (FastAPI Web App).
  3. Second terminal tab ready to run test commands.

---

## 2. Minute-by-Minute Speaking Script

### Minute 0:00 – 0:45 | Introduction & Candidate Configuration
**What to show:** Terminal with `config.py` and `git log`.  
**What to say:**  
> *"Hello! My name is [Full Name], USN [Your USN]. My personal random seed is S = 36, derived from the last three digits of my USN (036). I have answered Question A: Predict a Health Risk, and Question B: Turn a Model into a Usable App. As seen in my Git history, I committed my Level 3 pre-test predictions before running evaluations, and I have five structured commits spread across development."*

---

### Minute 0:45 – 2:00 | Question A: Predict a Health Risk (Focus on Levels 2 & 3)
**What to show:** Run `python question_a/level2_scratch.py`, then `python question_a/level3_reason.py`.  
**What to say:**  
> *"In Question A Level 1, using the UCI Heart Disease dataset with seed 36, Scikit-Learn Logistic Regression achieved 75.41% accuracy and 71.43% recall.*  
> 
> *In Level 2, I coded Logistic Regression completely from scratch using pure NumPy: implementing the numerically stable sigmoid function, binary cross-entropy loss, and batch gradient descent without any scikit-learn. I also wrote a custom confusion matrix from scratch.*  
> 
> *Here is the key result: My scratch NumPy model achieves exactly 75.41% accuracy, perfectly matching Scikit-Learn to 4 decimal places. Furthermore, the top three features by weight in both models are identical: first 'ca' (fluoroscopy vessels, weight 1.06 vs 0.97), second 'sex' (0.96 vs 0.86), and third 'cp' (chest pain type, 0.91 vs 0.84).*  
> 
> *In Level 3, I hypothesized that lowering the decision threshold to reach 90% recall would cause precision to drop from ~74% down to the 50-60% range. Running the threshold sweep confirmed this: at threshold theta = 0.08, recall reaches 92.86%, while precision drops to 56.52% due to increased false positives.*  
> 
> *In a clinical screening tool, this lower threshold is essential because a False Negative means an undiagnosed patient could suffer a fatal cardiac event, while a False Positive only requires routine secondary tests. Accuracy alone misleads because 0-1 loss weighs false alarms and lethal misses equally."*

---

### Minute 2:00 – 3:30 | Question B: Full-Stack App, Raw SQL, & Tests (Focus on Levels 1 & 2)
**What to show:** Browser at `http://127.0.0.1:8000` and terminal running `pytest question_b/tests/test_api.py -v`.  
**What to say:**  
> *"For Question B, I built a production FastAPI application that serves the trained heart model.*  
> 
> *On the frontend, users can enter clinical parameters or click quick presets. For example, clicking 'High-Risk Patient' instantly predicts an 86% cardiovascular risk, showing a red danger gauge, plain-language clinical interpretation, and emergency lifestyle warnings.*  
> 
> *In Level 2, every request is saved in SQLite. Notice our Database Statistics section: it queries the `/stats` endpoint, which uses hand-written SQL with COUNT, AVG, and CASE statements—strictly without an ORM. Total requests, average risk, and high-risk proportion update dynamically.*  
> 
> *Next, let's look at validation: entering an invalid age like -5 or text returns an immediate HTTP 422 with a clear medical error message: 'Age must be between 1 and 120'.*  
> 
> *Running pytest demonstrates 8 passing automated tests, verifying valid prediction, SQL stats calculation, and bad input rejection."*

---

### Minute 3:30 – 4:45 | Question B Level 3: Fault Injection & 100-User Concurrency
**What to show:** Run `python question_b/level3_reason.py`.  
**What to say:**  
> *"In Level 3, I intentionally broke my app in two ways to demonstrate resilience.*  
> 
> *In Breakage 1, I simulated a missing model file on disk. Before my fix, this causes an unhandled 500 crash or process termination. After my fix, the app catches the missing state, marks `/health` as degraded, and returns a clean HTTP 503 Service Unavailable.*  
> 
> *In Breakage 2, I injected text strings into numeric fields. Pydantic schema validation intercepts this at the gateway, returning structured 422 errors instead of unhandled 500 exceptions.*  
> 
> *To scale this app safely for 100 concurrent users: First, we run FastAPI with Uvicorn worker processes behind Gunicorn to parallelize CPU inference across cores. Second, we enabled Write-Ahead Logging in SQLite with busy timeout so concurrent reads and writes do not lock the database. For enterprise scale, this connects to a PostgreSQL connection pool. Third, keeping the service completely stateless allows seamless horizontal scaling behind a reverse proxy."*

---

### Minute 4:45 – 5:00 | Conclusion & Verification
**What to show:** Terminal running `python run_all.py`.  
**What to say:**  
> *"Running our master verification script `run_all.py` executes all five levels and pytest in under 7 seconds with zero errors. All code, datasets, decision logs, and pre-run predictions are available in the repository. Thank you!"*
