# Handwritten Digit Classification — Multi-Model Benchmark with Streamlit

## a. Problem Statement

Given a 8×8 grayscale bitmap of a handwritten digit, predict which numeral
(0–9) it represents. This is a **multi-class classification** problem with ten
balanced classes.

The objective of this assignment is not simply to maximise accuracy on one
model, but to train five structurally different classifiers on the *same*
train/test split and compare them across six evaluation metrics — so that the
trade-offs between a linear model, a single tree, an instance-based learner, a
generative model and an ensemble become visible on identical data. The trained
models are then exposed through an interactive Streamlit application where any
user can upload a test CSV and re-score every model live.

## b. Dataset Description

**Name:** Optical Recognition of Handwritten Digits Data Set
**Source:** UCI Machine Learning Repository —
<https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits>
(bundled with scikit-learn as `sklearn.datasets.load_digits`, which ships the
1797-instance test partition of the original UCI archive)

| Property | Value |
|---|---|
| Instances | **1,797** (requirement: ≥ 500) |
| Features | **64** (requirement: ≥ 12) |
| Feature type | Integer, range 0–16 |
| Target | `digit_class` — 10 classes (0–9) |
| Missing values | None |
| Class balance | Near-uniform, ~177–183 instances per class |

**How the features were produced.** Each original scan is a 32×32 binary
bitmap. It is divided into non-overlapping 4×4 blocks, and the number of "on"
pixels inside each block is counted. This yields an 8×8 grid of integers in
the range 0–16, which is flattened into the 64 features `pixel_0_0` …
`pixel_7_7`. This block-averaging step gives the dataset useful properties: it
compresses the input by a factor of 16 and makes the representation tolerant to
small distortions in how a digit was written.

**Why this dataset was chosen.** It comfortably clears both the feature and
instance minimums, it is genuinely multi-class rather than binary (so the
metrics must be macro-averaged and the confusion matrix is informative rather
than a 2×2 box), all features share a single common scale, and there are no
missing values — which keeps the focus on model comparison rather than on
data cleaning.

**Preprocessing.**
- Stratified 70/30 train–test split (`random_state=42`) → 1,257 train / 540 test.
- `StandardScaler` applied **inside a pipeline** for Logistic Regression and
  kNN (the two distance/magnitude-sensitive models). Tree-based models and
  Gaussian Naive Bayes are fitted on the raw features, since scaling does not
  affect them.
- Scaling lives inside the `Pipeline` object rather than being applied
  globally, which prevents train-statistics leaking into the test fold and
  means the Streamlit app can accept a raw CSV with no manual preprocessing.

## c. GitHub Repository Link

`https://github.com/<your-username>/<your-repo-name>`

> Replace with your actual repository URL before submitting.

**Live Streamlit App:** `https://<your-app-name>.streamlit.app`

### Repository structure

```
project-folder/
│-- app.py                        # Streamlit front-end
│-- requirements.txt              # deployment dependencies
│-- README.md                     # this file
│-- test_data.csv                 # 540-row hold-out split used by the app
│-- model/
    │-- train_models.ipynb        # trains + persists all five models (executed, outputs embedded)
    │-- saved/
        │-- logistic_regression.joblib
        │-- decision_tree.joblib
        │-- knn.joblib
        │-- naive_bayes.joblib
        │-- random_forest.joblib
        │-- metrics.csv
        │-- feature_names.json
```

## d. Models Used

All five models were trained on the identical 1,257-row training split and
evaluated on the identical 540-row hold-out split. Because the task is
multi-class, Precision, Recall and F1 are **macro-averaged** (every digit
counts equally regardless of frequency) and AUC is computed
**one-vs-rest, macro-averaged** from predicted class probabilities.

### Hyperparameters

| Model | Configuration |
|---|---|
| Logistic Regression | `C=0.5`, `max_iter=2000`, L2 penalty, on scaled features |
| Decision Tree | `criterion='entropy'`, `max_depth=12`, `min_samples_leaf=3` |
| kNN | `n_neighbors=5`, `weights='distance'`, Euclidean, on scaled features |
| Naive Bayes | `GaussianNB`, `var_smoothing=1e-2` |
| Random Forest | `n_estimators=400`, unrestricted depth |

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.9796** | 0.9991 | **0.9797** | **0.9795** | **0.9796** | **0.9774** |
| Decision Tree | 0.8204 | 0.9240 | 0.8258 | 0.8201 | 0.8209 | 0.8008 |
| kNN | 0.9722 | 0.9937 | 0.9725 | 0.9720 | 0.9720 | 0.9692 |
| Naive Bayes | 0.9130 | 0.9887 | 0.9151 | 0.9130 | 0.9131 | 0.9035 |
| Random Forest (Ensemble) | 0.9685 | **0.9992** | 0.9698 | 0.9681 | 0.9682 | 0.9652 |

*Reproduce by running all cells of `model/train_models.ipynb`; results are
deterministic at `random_state=42`.*

### Stability check — 5-fold cross-validation

A single hold-out split separates the top models by only a handful of test
samples, so 5-fold cross-validation was run on the training set to check whether
those differences are real:

| ML Model Name | CV mean F1 | CV std | Hold-out F1 |
|---|---|---|---|
| Logistic Regression | 0.9664 | 0.0092 | 0.9796 |
| Decision Tree | 0.8501 | 0.0203 | 0.8209 |
| kNN | 0.9688 | 0.0169 | 0.9720 |
| Naive Bayes | 0.9162 | 0.0103 | 0.9131 |
| Random Forest (Ensemble) | **0.9736** | **0.0066** | 0.9682 |

This materially changes the interpretation. Logistic Regression topped the
single hold-out split, but under cross-validation **Random Forest has the
highest mean F1 and the lowest variance of any model** — its ±0.0066 spread is
less than half kNN's. Logistic Regression's hold-out score (0.9796) sits well
above its CV mean (0.9664), which indicates that particular split happened to
favour it. The three leading models remain within roughly one standard
deviation of one another.

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | The strongest model overall (F1 0.9796, MCC 0.9774), which is initially surprising for a linear classifier on an image task. The explanation is that after block-averaging, the 64 features are dense, on a common 0–16 scale, and the ten digit classes turn out to be very close to linearly separable in this space — so a model with only ~650 parameters has enough capacity without any room to overfit. Standardisation was essential; the `C=0.5` penalty adds mild regularisation. Its near-perfect AUC (0.9991) shows the probability estimates are well ranked, not just the hard labels. |
| **Decision Tree** | Clearly the weakest model (F1 0.8209), and the gap of ~16 points versus every other method is the most informative result in the table. A single tree splits on one pixel at a time, but no individual pixel is decisive for a digit — the signal is distributed across the whole 8×8 grid. Forcing axis-aligned cuts on such correlated features produces a brittle, high-variance model. Its AUC (0.9240) is also the lowest, reflecting coarse, poorly-calibrated leaf probabilities. It remains the only fully human-readable model here, so its value is interpretability, not accuracy. |
| **kNN** | Very strong (F1 0.9720), and essentially tied with Logistic Regression within noise. Digits of the same class produce near-identical pixel vectors, so Euclidean distance is a genuinely meaningful similarity measure here — exactly the condition under which kNN excels. `weights='distance'` gives closer neighbours more say, which helps on ambiguous digits. The costs are practical rather than statistical: it stores all 1,257 training rows and computes distances at prediction time, making it the slowest and heaviest model to serve. |
| **Naive Bayes** | Respectable but mid-pack (F1 0.9131), roughly 6–7 points behind the leaders. This is a textbook illustration of its conditional-independence assumption failing: neighbouring pixels in a digit are strongly correlated, so treating all 64 as independent given the class systematically over-counts evidence. `var_smoothing=1e-2` was necessary because many border pixels are near-constant zero within a class, giving near-zero variance and unstable Gaussian densities. Notably its AUC (0.9887) is far better than its accuracy would suggest — it ranks classes well but its probabilities are over-confident, so the arg-max decisions suffer. |
| **Random Forest (Ensemble)** | Nearly a 15-point F1 improvement over the single Decision Tree (0.9682 vs 0.8209), which is the clearest demonstration of the value of ensembling in this experiment. Bagging plus random feature subsetting decorrelates the 400 trees and averages away the variance that crippled the individual tree. It also achieves the best AUC in the entire table (0.9992), meaning its averaged vote proportions are the best-calibrated probability estimates here. It trails Logistic Regression only marginally on accuracy, while requiring no feature scaling and providing feature-importance output for free. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)**, on the weight of the evidence. Logistic Regression wins five of six metrics on the single hold-out split, and taken alone that table would name it the winner. But cross-validation contradicts that reading: Random Forest has the highest CV mean F1 (0.9736 vs 0.9664) *and* the lowest variance (±0.0066, less than half kNN's), while Logistic Regression's hold-out score sits well above its own CV mean — a sign that this particular split flattered it. Random Forest also posts the best AUC on the hold-out set, so its probability estimates are the best calibrated. The margins are genuinely small and the leading three sit within about one standard deviation of each other, so the defensible claim is that **Random Forest is the most reliable choice rather than a decisive winner**; Logistic Regression remains the better pick if inference speed or model size matters, being a fraction of the size on disk. What the experiment establishes unambiguously is the *tier ordering*: the leading three, then Naive Bayes, then the single Decision Tree well behind. |

## Streamlit App Features

| Requirement | Implementation |
|---|---|
| Dataset upload option (CSV) | Sidebar file uploader; schema is validated against the trained feature list and a clear error is shown on mismatch. Falls back to the bundled `test_data.csv` when nothing is uploaded. |
| Model selection dropdown | Sidebar `selectbox` listing all five trained models; predictions re-run instantly on selection. |
| Display of evaluation metrics | All six metrics (Accuracy, AUC, Precision, Recall, F1, MCC) shown as metric cards for the selected model, plus an optional all-models comparison table with best-in-column highlighting and a grouped bar chart. |
| Confusion matrix / classification report | Tabbed view: seaborn confusion-matrix heatmap, full per-class classification report, and a downloadable predictions table. |

Additional touches: cached model loading (`@st.cache_resource`), graceful
handling of CSVs with no target column (predictions-only mode), and CSV export
of predictions.

## How to Run Locally

```bash
git clone https://github.com/2025ac05349/ml-classification-benchmark.git
cd ml-classification-benchmark
pip install -r requirements.txt

jupyter notebook model/train_models.ipynb   # run all cells to regenerate model/saved/ and test_data.csv
streamlit run app.py             # opens on http://localhost:8501
```

## Deployment

Deployed on Streamlit Community Cloud from the `main` branch with `app.py` as
the entry point. `model/saved/*.joblib` is committed to the repository so the
cloud instance loads pre-trained models instead of retraining on cold start.

> **Note on version compatibility:** the `.joblib` files must be generated with
> the same scikit-learn version pinned in `requirements.txt`. If Streamlit
> Cloud logs show an unpickling warning, re-run `train_models.ipynb` locally under
> the pinned version and re-commit.
