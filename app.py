"""
app.py - Streamlit front-end for the handwritten-digit classification study.

Features
    * Upload your own test CSV (or fall back to the bundled test_data.csv)
    * Pick any of the five trained models from a dropdown
    * See the six required evaluation metrics for the selected model
    * Inspect the confusion matrix and full classification report
    * Compare every model side by side on the same uploaded data
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

SAVED_DIR = os.path.join("model", "saved")
DEFAULT_TEST_CSV = "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
}

st.set_page_config(page_title="Digit Classifier Benchmark",
                   page_icon="#", layout="wide")


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_models():
    loaded = {}
    for label, fname in MODEL_FILES.items():
        path = os.path.join(SAVED_DIR, fname)
        if os.path.exists(path):
            loaded[label] = joblib.load(path)
    return loaded


@st.cache_data(show_spinner=False)
def load_schema():
    with open(os.path.join(SAVED_DIR, "feature_names.json")) as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_default_test():
    return pd.read_csv(DEFAULT_TEST_CSV)


def compute_metrics(y_true, y_pred, y_proba):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": np.nan,
        "Precision": precision_score(y_true, y_pred, average="macro",
                                     zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro",
                               zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    try:
        if y_proba.shape[1] == 2:
            metrics["AUC"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            metrics["AUC"] = roc_auc_score(y_true, y_proba,
                                           multi_class="ovr", average="macro")
    except ValueError:
        # happens when the uploaded slice does not contain every class
        pass
    return metrics


# --------------------------------------------------------------------------- #
# Sidebar - data + model selection
# --------------------------------------------------------------------------- #
st.sidebar.header("1. Test data")
uploaded = st.sidebar.file_uploader(
    "Upload a test CSV (features + target column)", type=["csv"]
)

schema = load_schema()
FEATURES, TARGET = schema["features"], schema["target"]

if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.sidebar.success(f"Loaded {data.shape[0]} rows from upload.")
else:
    data = load_default_test()
    st.sidebar.info("Using bundled test_data.csv. Upload a file to override.")

st.sidebar.header("2. Model")
models = load_models()
if not models:
    st.error("No saved models found. Run all cells of `model/train_models.ipynb` first.")
    st.stop()

choice = st.sidebar.selectbox("Choose a classifier", list(models.keys()))
show_all = st.sidebar.checkbox("Also show all-model comparison", value=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dataset: UCI Optical Recognition of Handwritten Digits "
    "(1797 instances, 64 features, 10 classes)."
)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
st.title("Handwritten Digit Classification - Model Benchmark")
st.caption(
    "Five supervised classifiers trained on the same dataset, evaluated on a "
    "held-out test split. Upload your own CSV to re-score them live."
)

missing = [c for c in FEATURES if c not in data.columns]
if missing:
    st.error(
        f"The uploaded file is missing {len(missing)} expected feature "
        f"column(s), e.g. {missing[:5]}. Please upload data with the same "
        "schema as test_data.csv."
    )
    st.stop()

has_target = TARGET in data.columns
X = data[FEATURES]

with st.expander("Preview the test data", expanded=False):
    st.dataframe(data.head(20), width="stretch")
    st.write(f"Shape: {data.shape[0]} rows x {data.shape[1]} columns")

model = models[choice]
y_pred = model.predict(X)
y_proba = model.predict_proba(X)

if not has_target:
    st.warning(
        f"No `{TARGET}` column found, so metrics cannot be computed. "
        "Showing predictions only."
    )
    st.subheader(f"Predictions - {choice}")
    out = data.copy()
    out["prediction"] = y_pred
    st.dataframe(out.head(50), width="stretch")
    st.download_button("Download predictions",
                       out.to_csv(index=False).encode(),
                       "predictions.csv", "text/csv")
    st.stop()

y_true = data[TARGET]


# --------------------------------------------------------------------------- #
# Metrics for the selected model
# --------------------------------------------------------------------------- #
st.subheader(f"Evaluation metrics - {choice}")
scores = compute_metrics(y_true, y_pred, y_proba)

cols = st.columns(6)
for col, (name, value) in zip(cols, scores.items()):
    col.metric(name, "n/a" if np.isnan(value) else f"{value:.4f}")

tab_cm, tab_report, tab_preds = st.tabs(
    ["Confusion matrix", "Classification report", "Predictions"]
)

with tab_cm:
    labels = sorted(pd.unique(pd.concat([pd.Series(y_true),
                                         pd.Series(y_pred)])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={"size": 8})
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{choice} - confusion matrix")
    st.pyplot(fig, width="content")

with tab_report:
    report = classification_report(y_true, y_pred, output_dict=True,
                                   zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(4),
                 width="stretch")

with tab_preds:
    out = data.copy()
    out["prediction"] = y_pred
    out["correct"] = out["prediction"] == out[TARGET]
    st.dataframe(out[[TARGET, "prediction", "correct"]].head(100),
                 width="stretch")
    st.download_button("Download predictions",
                       out.to_csv(index=False).encode(),
                       "predictions.csv", "text/csv")


# --------------------------------------------------------------------------- #
# All-model comparison on the same data
# --------------------------------------------------------------------------- #
if show_all:
    st.markdown("---")
    st.subheader("All models on this test data")
    rows = []
    for label, mdl in models.items():
        preds = mdl.predict(X)
        proba = mdl.predict_proba(X)
        rows.append({"ML Model Name": label,
                     **compute_metrics(y_true, preds, proba)})
    table = pd.DataFrame(rows).round(4)

    st.dataframe(
        table.style.highlight_max(
            subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
            color="#1f6f4a"
        ),
        width="stretch",
    )

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    melted = table.melt(id_vars="ML Model Name", var_name="Metric",
                        value_name="Score")
    sns.barplot(data=melted, x="Metric", y="Score", hue="ML Model Name",
                ax=ax2)
    ax2.set_ylim(0.7, 1.005)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax2.set_title("Metric comparison across models")
    st.pyplot(fig2, width="stretch")

    best = table.loc[table["F1"].idxmax(), "ML Model Name"]
    st.success(f"Best macro-F1 on this data: **{best}**")
