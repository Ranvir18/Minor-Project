import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Airport IoT Threat Detection", layout="wide")
st.title("Airport IoT Anomaly Detection Dashboard")

# ---------- Sidebar controls ----------
st.sidebar.header("Model settings")
contamination = st.sidebar.slider("Contamination", 0.001, 0.10, 0.01, 0.001)
n_estimators = st.sidebar.slider("n_estimators", 50, 500, 100, 50)

# ---------- Pipeline (same logic as detect_anomalies.py) ----------
def extract_features(df):
    f = df[["Time", "Length"]].copy()
    f["Time_Diff"] = f["Time"].diff().fillna(0)
    f["Packet_Rate"] = f["Time_Diff"].rolling(window=10, min_periods=1).mean()
    return f[["Length", "Time_Diff", "Packet_Rate"]]

@st.cache_data
def load_data():
    return (pd.read_csv("airport_datasets_excel2.csv"),
            pd.read_csv("airport_IoT_anomalies_dataset.csv"))

@st.cache_data
def run_detection(contamination, n_estimators):
    normal, attack = load_data()
    model = IsolationForest(n_estimators=n_estimators,
                            contamination=contamination, random_state=42)
    model.fit(extract_features(normal))
    X = extract_features(attack)
    out = attack.copy()
    out["Anomaly_Score"] = model.decision_function(X)
    out["Is_Anomaly"] = model.predict(X)
    out["Is_Anomaly"] = out["Is_Anomaly"].map({1: "Normal", -1: "Anomaly"})
    return out

df = run_detection(contamination, n_estimators)

# ---------- KPIs ----------
total = len(df)
flagged = (df["Is_Anomaly"] == "Anomaly").sum()
c1, c2, c3 = st.columns(3)
c1.metric("Packets analysed", f"{total:,}")
c2.metric("Anomalies flagged", f"{flagged:,}")
c3.metric("Anomaly rate", f"{flagged / total * 100:.2f}%")

# ---------- Charts ----------
colors = {"Normal": "#4C78A8", "Anomaly": "#E45756"}

left, right = st.columns(2)
with left:
    st.subheader("Traffic timeline")
    fig = px.scatter(df, x="Time", y="Length", color="Is_Anomaly",
                     color_discrete_map=colors, opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Anomaly score distribution")
    fig = px.histogram(df, x="Anomaly_Score", color="Is_Anomaly",
                       color_discrete_map=colors, nbins=60)
    st.plotly_chart(fig, use_container_width=True)

# Only shows if your Wireshark export has these columns
for col in ["Protocol", "Source", "Destination"]:
    if col in df.columns:
        st.subheader(f"Anomalies by {col}")
        top = (df[df["Is_Anomaly"] == "Anomaly"][col]
               .value_counts().head(10).reset_index())
        top.columns = [col, "count"]
        st.plotly_chart(px.bar(top, x=col, y="count"), use_container_width=True)

# ---------- Table + export ----------
st.subheader("Most suspicious packets")
worst = df.sort_values("Anomaly_Score").head(100)
st.dataframe(worst, use_container_width=True)
st.download_button("Download flagged packets (CSV)",
                   df[df["Is_Anomaly"] == "Anomaly"].to_csv(index=False),
                   "flagged_anomalies_output.csv")