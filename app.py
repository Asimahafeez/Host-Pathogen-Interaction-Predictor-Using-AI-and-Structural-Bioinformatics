
import streamlit as st
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from utils_features import build_hp_features

st.set_page_config(page_title="Host-Pathogen Interaction Predictor", layout="wide")
st.title("🧫 Host–Pathogen Interaction Predictor (Pro)")
st.write("""Upload a CSV with columns: `host_seq`, `pathogen_seq`, optionally `host_id` and `pathogen_id`, and `label` (0/1) if training.
Or use the demo dataset.""")

# Sidebar
st.sidebar.header("Settings")
use_demo = st.sidebar.checkbox("Use demo data", value=True)
model_choice = st.sidebar.selectbox("Model", ["Random Forest"])
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
rs = st.sidebar.number_input("Random state", value=42, step=1)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if use_demo or uploaded is None:
    df = pd.read_csv("data_demo_pairs.csv")
    st.info("Using demo host-pathogen pairs.")
else:
    df = pd.read_csv(uploaded)

st.subheader("Data preview")
st.dataframe(df.head())

# Check required columns
if not {"host_seq","pathogen_seq"}.issubset(df.columns):
    st.error("CSV must contain 'host_seq' and 'pathogen_seq' columns.")
    st.stop()

# Feature engineering
X, feat_names, df_enriched = build_hp_features(df)
st.subheader("Feature preview (first 5)")
st.dataframe(X.head())

# Prepare labels
if "label" in df.columns:
    y = df["label"].astype(int)
else:
    st.info("No label column found — model will not be trained, use prediction mode with demo model.")
    y = None

# Train if labels exist
if y is not None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=rs, n_jobs=-1)
    model.fit(X_train_s, y_train)

    probs = model.predict_proba(X_test_s)[:,1]
    preds = (probs>=0.5).astype(int)
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    st.metric("ROC AUC", f"{auc:.3f}")
    st.metric("Accuracy", f"{acc:.3f}")

    # ROC plot
    fpr, tpr, _ = roc_curve(y_test, probs)
    fig, ax = plt.subplots()
    ax.plot(fpr,tpr,label=f"AUC={auc:.3f}")
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend()
    st.pyplot(fig)

    # Feature importance (permutation)
    try:
        r = permutation_importance(model, X_test_s, y_test, n_repeats=10, random_state=rs, n_jobs=-1)
        imp = pd.DataFrame({"feature": feat_names, "importance": r.importances_mean}).sort_values("importance", ascending=False)
        st.subheader("Global feature importance (permutation)")
        st.dataframe(imp.head(20))
    except Exception as e:
        st.warning(f"Permutation importance skipped: {e}")
else:
    st.info("No labels — skip training. Use the 'Predict' section below to score new pairs with a demo heuristic.")

# Prediction on new/uploaded pairs
st.subheader("Predict / Score pairs")
if y is None:
    # build a simple heuristic score from features for demo predictions
    X_s, _, enr = build_hp_features(df)
    # simple logistic-like score using blosum_mean and pid and hydro_diff
    score = (X_s["blosum_mean"].fillna(0) * 0.6) + (X_s["pid"].fillna(0) * 0.3) - (np.abs(X_s["hydro_diff"].fillna(0)) * 0.05)
    enr["interaction_score"] = score
    enr["pred_class"] = (score >= score.median()).astype(int)
    st.dataframe(enr[["host_id","pathogen_id","interaction_score","pred_class"]].head(30))
    # allow download
    enr.to_csv("hp_predictions.csv", index=False)
    with open("hp_predictions.csv","rb") as f:
        st.download_button("⬇ Download predictions.csv", f, file_name="hp_predictions.csv")
else:
    st.subheader("Predict on all pairs and show network of top interactions")
    # predict on full X using trained model
    X_all_s = scaler.transform(X)
    probs_all = model.predict_proba(X_all_s)[:,1]
    df_enriched["pred_prob"] = probs_all
    df_enriched["pred_class"] = (df_enriched["pred_prob"] >= 0.5).astype(int)
    st.dataframe(df_enriched[["host_id","pathogen_id","pred_prob","pred_class"]].head(50))
    # network of top interactions
    top = df_enriched.sort_values("pred_prob", ascending=False).head(40)
    G = nx.Graph()
    for _, r in top.iterrows():
        h = r.get("host_id", f"H_{_}")
        p = r.get("pathogen_id", f"P_{_}")
        score = float(r["pred_prob"])
        G.add_node(h, bipartite=0)
        G.add_node(p, bipartite=1)
        G.add_edge(h, p, weight=score)
    st.subheader("Interaction network (top predicted pairs)")
    pos = nx.spring_layout(G, seed=42, k=0.5)
    figg, axg = plt.subplots(figsize=(8,6))
    weights = [G[u][v]['weight']*3 for u,v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='skyblue', ax=axg)
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.8, ax=axg)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=axg)
    axg.set_axis_off()
    st.pyplot(figg)

st.markdown("---")
st.caption("Research prototype. Not for clinical use.")
