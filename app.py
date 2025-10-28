# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import random

import preprocessing as pdata
import model as m  # exposes train_and_predict + the classifier classes

# --- Helpers (must be defined before first use) ---
def pct_to_fraction(pct: int) -> float:
    return max(0.01, min(0.99, pct / 100.0))

def compute_report_and_tables(y_true, y_pred):
    from sklearn.metrics import classification_report
    import pandas as pd
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_per_class = df_report.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    return df_report, df_per_class

def make_figures(y_true, y_pred, labels, feature_cols, model_has_importances, importances_series=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import pandas as pd
    figs = {}

    df_report, df_per_class = compute_report_and_tables(y_true, y_pred)
    fig1 = plt.figure(figsize=(10, 6))
    sns.barplot(data=df_per_class[['precision', 'recall', 'f1-score']])
    plt.title("Classification Metrics per Fast Food Company")
    plt.ylabel("Score"); plt.xlabel("Company"); plt.xticks(rotation=45); plt.ylim(0, 1)
    figs["per_class"] = fig1

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', xticks_rotation=45, ax=ax2, colorbar=False)
    ax2.set_title("Confusion Matrix: Fast Food Brand Prediction")
    figs["cm"] = fig2

    if model_has_importances and importances_series is not None:
        fig3 = plt.figure(figsize=(8, 6))
        importances_series.sort_values(ascending=True).plot(kind='barh')
        plt.title("Feature Importance in Predicting Fast Food Company")
        plt.xlabel("Importance")
        figs["fi"] = fig3

    return figs

st.set_page_config(page_title="Fast Food Brand Classifier", layout="wide")
st.title("NoshNarc© - A Fast Food Brand Classifier")
st.subheader("Can ML figure out what company fast food menu items come from?")

# ------------------ Session defaults ------------------
st.session_state.setdefault("seed_input", 42)             # pending seed (editable)
st.session_state.setdefault("test_size_pct_input", 30)    # pending test size %
st.session_state.setdefault("model_choice_input", "RandomForestClassifier")  # pending model choice
st.session_state.setdefault("last_run", None)             # stores the latest trained results dict
st.session_state.setdefault("history", [])                # list of past runs for reference
st.session_state.setdefault("bootstrapped", False)  # ← NEW

# ------------------ First-load bootstrap (runs once) ------------------
st.session_state.setdefault("bootstrapped", False)

if not st.session_state.bootstrapped and st.session_state.last_run is None:
    with st.spinner("Training default model…"):
        seed = int(st.session_state.seed_input)
        test_size_pct = int(st.session_state.test_size_pct_input)
        test_size = pct_to_fraction(test_size_pct)

        model_selection_name = st.session_state.model_choice_input
        model_cls = getattr(m, model_selection_name)

        # Load, train, predict
        X_train, X_test, y_train, y_test, feature_cols = pdata.load_data(seed=seed, test_size=test_size)
        model, y_pred = m.train_and_predict(X_train, y_train, X_test, seed=seed, model_selection=model_cls)

        # Results + figs
        df_report, df_per_class = compute_report_and_tables(y_test, y_pred)
        labels = getattr(model, "classes_", None)
        importances_series = None
        if hasattr(model, "feature_importances_"):
            importances_series = pd.Series(model.feature_importances_, index=feature_cols)
        figs = make_figures(
            y_true=y_test, y_pred=y_pred, labels=labels, feature_cols=feature_cols,
            model_has_importances=hasattr(model, "feature_importances_"),
            importances_series=importances_series
        )

        # Store as the current run
        st.session_state.last_run = dict(
            seed=seed,
            test_size_pct=test_size_pct,
            model_name=model_selection_name,
            model=model,
            df_report=df_report,
            df_per_class=df_per_class,
            labels=labels,
            figs=figs,
            has_importances=hasattr(model, "feature_importances_"),
            importances_series=importances_series,
            feature_cols=feature_cols,
        )

        # Optional: record in history
        try:
            acc = float(df_report.loc["accuracy", "precision"])
        except Exception:
            acc = None
        st.session_state.history.append({
            "seed": seed,
            "test_size_pct": test_size_pct,
            "model": model_selection_name,
            "accuracy": acc,
            "note": "initial load"
        })

        st.session_state.bootstrapped = True
        st.rerun()  # ← ensures the page shows results immediately on first load

# ------------------ Top bar controls ------------------
c1, c2, c3, c4, c5 = st.columns([1.2, 1, 3, 2.2, 1.2])

# 1) Randomize seed BEFORE rendering the input
with c2:
    random_clicked = st.button("🎲 Random seed", help="Fill the seed box with a random number (does not retrain).")
if random_clicked:
    st.session_state.seed_input = random.randrange(0, 10**9)
    st.rerun()

# 2) Pending inputs
with c1:
    st.number_input(
        "Seed",
        min_value=0, max_value=10**9, step=1, key="seed_input",
        help="Type a specific seed for reproducible splits/training (pending)."
    )

with c3:
    st.slider(
        "Test size (%)",
        min_value=1, max_value=99, step=1, key="test_size_pct_input",
        help="Evaluate stability as you increase the test set size (pending)."
    )

with c4:
    # NEW: model selection (pending)
    model_name = st.selectbox(
        "Model",
        options=["RandomForestClassifier", "HistGradientBoostingClassifier"],
        index=0 if st.session_state.model_choice_input == "RandomForestClassifier" else 1,
        help="Choose which estimator to train (pending)."
    )
    # keep it in session so it persists until retrain
    st.session_state.model_choice_input = model_name

# 3) Retrain button — only when clicked do we run the pipeline and update 'last_run'
with c5:
    retrain_clicked = st.button("🔁 Retrain", type="primary", help="Run split + training using the pending config.")

def pct_to_fraction(pct: int) -> float:
    return max(0.01, min(0.99, pct / 100.0))

def compute_report_and_tables(y_true, y_pred):
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_per_class = df_report.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    return df_report, df_per_class

def make_figures(y_true, y_pred, labels, feature_cols, model_has_importances, importances_series=None):
    figs = {}

    # Per-class metrics
    df_report, df_per_class = compute_report_and_tables(y_true, y_pred)
    fig1 = plt.figure(figsize=(10, 6))
    sns.barplot(data=df_per_class[['precision', 'recall', 'f1-score']])
    plt.title("Classification Metrics per Fast Food Company")
    plt.ylabel("Score"); plt.xlabel("Company"); plt.xticks(rotation=45); plt.ylim(0, 1)
    figs["per_class"] = fig1

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', xticks_rotation=45, ax=ax2, colorbar=False)
    ax2.set_title("Confusion Matrix: Fast Food Brand Prediction")
    figs["cm"] = fig2

    # Feature importance (RF exposes it; HGBT typically doesn't)
    if model_has_importances and importances_series is not None:
        fig3 = plt.figure(figsize=(8, 6))
        importances_series.sort_values(ascending=True).plot(kind='barh')
        plt.title("Feature Importance in Predicting Fast Food Company")
        plt.xlabel("Importance")
        figs["fi"] = fig3

    return figs

# ------------------ Train only when user clicks ------------------
if retrain_clicked:
    # Commit pending config as the "active" run
    seed = int(st.session_state.seed_input)
    test_size_pct = int(st.session_state.test_size_pct_input)
    test_size = pct_to_fraction(test_size_pct)

    # Map selection → class (classes are exposed via the imported `model` module)
    model_selection_name = st.session_state.model_choice_input
    model_cls = getattr(m, model_selection_name)   # e.g., m.RandomForestClassifier or m.HistGradientBoostingClassifier

    # Load, train, predict
    X_train, X_test, y_train, y_test, feature_cols = pdata.load_data(seed=seed, test_size=test_size)
    model, y_pred = m.train_and_predict(X_train, y_train, X_test, seed=seed, model_selection=model_cls)  # ← pass the class

    # Compute results + figures (store in session so they persist while you tweak inputs)
    df_report, df_per_class = compute_report_and_tables(y_test, y_pred)
    labels = getattr(model, "classes_", None)
    importances_series = None
    if hasattr(model, "feature_importances_"):
        importances_series = pd.Series(model.feature_importances_, index=feature_cols)
    figs = make_figures(
        y_true=y_test, y_pred=y_pred, labels=labels, feature_cols=feature_cols,
        model_has_importances=hasattr(model, "feature_importances_"),
        importances_series=importances_series
    )

    # Save the whole run so it remains visible across subsequent input tweaks
    st.session_state.last_run = dict(
        seed=seed,
        test_size_pct=test_size_pct,
        model_name=model_selection_name,
        model=model,  # ← add this so we can predict later
        df_report=df_report,
        df_per_class=df_per_class,
        labels=labels,
        figs=figs,
        has_importances=hasattr(model, "feature_importances_"),
        importances_series=importances_series,
        feature_cols=feature_cols,  # ← add this so we know the order for manual input
    )

    # Log lightweight history (seed, test size, accuracy, model)
    try:
        acc = float(df_report.loc["accuracy", "precision"])
    except Exception:
        acc = None
    st.session_state.history.append({
        "seed": seed,
        "test_size_pct": test_size_pct,
        "model": model_selection_name,
        "accuracy": acc
    })

# ------------------ Display ------------------
pending_caption = (
    f"Pending config → Model: {st.session_state.model_choice_input} • "
    f"Seed: {st.session_state.seed_input} • Test size: {st.session_state.test_size_pct_input}%"
)
st.caption(pending_caption)

if st.session_state.last_run is None:
    st.info("No run yet. Choose model/seed/test size, then click **🔁 Retrain**.")
else:
    run = st.session_state.last_run
    st.success(
        f"Showing last trained run → Model: {run['model_name']} • "
        f"Seed: {run['seed']} • Test size: {run['test_size_pct']}%"
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Overall report", "Confusion matrix", "Feature importance", "History"])

    with tab1:
        st.subheader("Classification report")
        st.dataframe(run["df_report"].style.format(precision=3), use_container_width=True)
        st.subheader("Per-class metrics")
        st.pyplot(run["figs"]["per_class"], use_container_width=True)

    with tab2:
        st.subheader("Confusion Matrix")
        st.pyplot(run["figs"]["cm"], use_container_width=True)

    with tab3:
        if run["has_importances"] and "fi" in run["figs"]:
            st.subheader("Feature importance")
            st.pyplot(run["figs"]["fi"], use_container_width=True)
        else:
            st.info("This model doesn’t expose feature importances.")

    with tab4:
        if st.session_state.history:
            df_hist = pd.DataFrame(st.session_state.history)
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.info("No past runs yet.")

# --- Add after your existing tabs block ---

# Helper to extract a single-row features DataFrame in correct order
def _build_features_df_from_row(row, feature_cols):
    return pd.DataFrame([row[feature_cols].values], columns=feature_cols)

def _predict_and_show(model, feature_cols, X_new):
    pred = model.predict(X_new)[0]
    st.markdown(f"### 🧠 Predicted company: **{pred}**")

    # Try to show probabilities if supported
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_new)[0]
            labels = getattr(model, "classes_", None)
            if labels is not None:
                probs_df = pd.DataFrame({"Company": labels, "Probability": proba})
                probs_df = probs_df.sort_values("Probability", ascending=False)
                st.subheader("Class probabilities")
                st.dataframe(probs_df.style.format({"Probability": "{:.3f}"}), use_container_width=True)
        except Exception:
            pass

# ------------------ Try your own item ------------------
try_tab = st.tabs(["Try your own item"])[0]

with try_tab:
    if st.session_state.last_run is None:
        st.info("Train once (🔁 Retrain) to enable live predictions with your current configuration.")
    else:
        run = st.session_state.last_run
        model = run["model"]
        feature_cols = run.get("feature_cols") or getattr(pdata, "feature_cols", None)

        if feature_cols is None:
            st.error("Feature columns not available. Retrain once to proceed.")
        else:
            st.markdown("### Test a single item using the **current model** (no retraining)")

            mode = st.radio(
                "Choose input method",
                ["Pick an existing dataset item", "Enter nutrition manually"],
                horizontal=True
            )

            if mode == "Pick an existing dataset item":
                # Build an easy-to-pick menu: "Item — Company (truth)"
                df_all = getattr(pdata, "df", None)
                if df_all is None:
                    st.error("Dataset not found in preprocessing. Can’t list items.")
                else:
                    # Display unique rows by (Item, Company) to avoid ambiguous names
                    display_rows = df_all[["Item", "Company"] + feature_cols].copy()
                    display_rows["__display__"] = display_rows.apply(
                        lambda r: f"{r['Item']} — {r['Company']}", axis=1
                    )
                    choice = st.selectbox(
                        "Choose an item from the dataset",
                        options=display_rows["__display__"].tolist()
                    )
                    if st.button("🔮 Predict selected item"):
                        row = display_rows.loc[display_rows["__display__"] == choice].iloc[0]
                        X_new = _build_features_df_from_row(row, feature_cols)
                        _predict_and_show(model, feature_cols, X_new)

                        st.caption("Using the item’s nutrition from your dataset. The shown 'truth' company is for reference only.")

            else:
                st.write("Enter values for each feature below:")
                cols = st.columns(2)
                # Pre-fill with dataset medians for convenience
                df_all = getattr(pdata, "df", None)
                med = df_all[feature_cols].median() if df_all is not None else pd.Series(0, index=feature_cols)

                inputs = {}
                for i, feat in enumerate(feature_cols):
                    with cols[i % 2]:
                        default_val = float(med.get(feat, 0))
                        # Let the user type any float; adjust bounds if you want validation
                        inputs[feat] = st.number_input(feat, value=default_val, step=1.0, format="%.6f")

                if st.button("🔮 Predict manual item"):
                    X_new = pd.DataFrame([inputs], columns=feature_cols)
                    _predict_and_show(model, feature_cols, X_new)

            st.caption(
                "Predictions use the **currently trained model** (Model/Seed/Test size shown above). "
                "Change the configuration and click **🔁 Retrain** to update the model used here."
            )