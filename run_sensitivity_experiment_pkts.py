import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# 1. Load Datasets
df_normal = pd.read_csv('airport_datasets_excel2.csv')
df_attack = pd.read_csv('airport_IoT_anomalies_dataset.csv')

# 2. Extract Packet-Level Features
def extract_packet_features(df):
    df_feat = df[['Time', 'Length']].copy()
    df_feat['Time_Diff'] = df_feat['Time'].diff().fillna(0)
    df_feat['Packet_Rate'] = df_feat['Time_Diff'].rolling(window=10, min_periods=1).mean()
    return df_feat[['Length', 'Time_Diff', 'Packet_Rate']]

X_train = extract_packet_features(df_normal)
X_test = extract_packet_features(df_attack)

# 3. Sensitivity Thresholds
sensitivities = [0.01, 0.02, 0.025, 0.03, 0.05, 0.08, 0.10, 0.15]

# 4. Iterate over Packet Data
results_summary = []

for c in sensitivities:
    model = IsolationForest(n_estimators=100, contamination=c, random_state=42)
    model.fit(X_train)
    
    col_name = f'Is_Anomaly_{c*100}%'
    preds = model.predict(X_test)
    df_attack[col_name] = np.where(preds == -1, 'Anomaly', 'Normal')
    
    anomalies_count = (preds == -1).sum()
    pct_flagged = (anomalies_count / len(df_attack)) * 100
    
    results_summary.append({
        'Contamination (%)': f"{c*100}%",
        'Anomalous Packets Flagged': anomalies_count,
        'Total Packets Scanned': len(df_attack),
        '% Packets Flagged': f"{pct_flagged:.2f}%"
    })

# 5. Output Summary Table
summary_df = pd.DataFrame(results_summary)
print("--- Packet-Level Sensitivity Analysis ---")
print(summary_df.to_string(index=False))

# 6. Save Packet Output
df_attack.to_csv('packet_level_anomalies_output.csv', index=False)