# UCI Heart Disease Dataset Documentation

## Provenance
- **Source:** UCI Machine Learning Repository
- **Original Donor:** V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
- **Repository URL:** `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`
- **Total Records:** 303 patient records
- **Total Features:** 13 clinical predictors + 1 binary target attribute
- **Random Seed Used Across Project:** `S = 36` (Candidate USN ending in `036`)

## Clinical Feature Dictionary

| Feature Name | Clinical Description | Data Type | Valid Clinical Range | Encoded Values |
|:-------------|:---------------------|:----------|:---------------------|:---------------|
| `age` | Patient age in years | Integer | 1 - 120 | Continuous integer (dataset range: 29 - 77) |
| `sex` | Biological sex | Integer | 0 or 1 | `0`: Female, `1`: Male |
| `cp` | Chest pain type | Integer | 1 - 4 | `1`: Typical angina<br>`2`: Atypical angina<br>`3`: Non-anginal pain<br>`4`: Asymptomatic |
| `trestbps` | Resting blood pressure (admission) | Integer | 50 - 300 mm Hg | Continuous (dataset range: 94 - 200 mm Hg) |
| `chol` | Serum cholesterol | Integer | 100 - 600 mg/dl | Continuous (dataset range: 126 - 564 mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl | Integer | 0 or 1 | `0`: False ($\le 120$ mg/dl)<br>`1`: True ($> 120$ mg/dl) |
| `restecg` | Resting electrocardiographic results | Integer | 0 - 2 | `0`: Normal<br>`1`: ST-T wave abnormality<br>`2`: Left ventricular hypertrophy |
| `thalach` | Maximum heart rate achieved | Integer | 50 - 250 bpm | Continuous (dataset range: 71 - 202 bpm) |
| `exang` | Exercise induced angina | Integer | 0 or 1 | `0`: No<br>`1`: Yes |
| `oldpeak` | ST depression induced by exercise relative to rest | Float | 0.0 - 10.0 mm | Continuous float (dataset range: 0.0 - 6.2 mm) |
| `slope` | Slope of the peak exercise ST segment | Integer | 1 - 3 | `1`: Upsloping<br>`2`: Flat<br>`3`: Downsloping |
| `ca` | Number of major vessels colored by fluoroscopy | Integer | 0 - 3 | `0`, `1`, `2`, `3` (Missing values imputed by mode 0) |
| `thal` | Thalassemia cardiac defect | Integer | 3, 6, 7 | `3`: Normal<br>`6`: Fixed defect<br>`7`: Reversible defect (Missing values imputed by mode 3) |
| `target` | Presence of heart disease ($>50\%$ diameter narrowing) | Integer | 0 or 1 | `0`: Absence of disease ($<50\%$ narrowing, $n=164$)<br>`1`: Presence of disease ($>50\%$ narrowing, $n=139$) |

## Cleaning & Preprocessing Rules
1. **Missing Data Imputation:** In the original Cleveland clinical data, `ca` contained 4 missing values (`?`) and `thal` contained 2 missing values (`?`). These were imputed using the domain modes (`ca=0`, `thal=3`).
2. **Target Binarization:** Original target values ranged from $0$ to $4$ indicating stenosis severity. For clinical risk screening, values $>0$ are binarized to $1$ (disease present) and $0$ remains $0$ (disease absent).
3. **Type Strictness:** All clinical categorical and discrete count columns are strictly cast to integer, and `oldpeak` is retained as a continuous float.
