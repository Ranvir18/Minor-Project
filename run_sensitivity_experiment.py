import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# 1. Load Datasets
df_normal = pd.read_csv('airport_datasets_excel2.csv')
df_attack = pd.read_csv('airport_IoT_anomalies_dataset.csv')

# 2. Extract Window-Level Features (1-second intervals)
def extract_window_features(df):
    df['time_sec'] = df['Time'].astype(int)
    grouped = df.groupby('time_sec').agg(
        packet_count=('Length', 'count'),
        total_bytes=('Length', 'sum'),
        mean_length=('Length', 'mean'),
        std_length=('Length', lambda x: x.std() if len(x) > 1 else 0)
    ).reset_index()
    return grouped

df_normal_win = extract_window_features(df_normal)
df_attack_win = extract_window_features(df_attack)

features = ['packet_count', 'total_bytes', 'mean_length', 'std_length']

X_train = df_normal_win[features]
X_test = df_attack_win[features]

# 3. Sensitivity Thresholds
sensitivities = [0.01, 0.02, 0.025, 0.03, 0.05, 0.08, 0.10, 0.15]

# 4. Train Model across Sensitivities
results_summary = []

for c in sensitivities:
    model = IsolationForest(n_estimators=100, contamination=c, random_state=42)
    model.fit(X_train)
    
    col_name = f'Is_Anomaly_{c*100}%'
    preds = model.predict(X_test)
    df_attack_win[col_name] = np.where(preds == -1, 'Anomaly', 'Normal')
    
    anomalies_count = (preds == -1).sum()
    pct_flagged = (anomalies_count / len(df_attack_win)) * 100
    
    results_summary.append({
        'Contamination (%)': f"{c*100}%",
        'Anomalous Seconds Flagged': anomalies_count,
        'Total Seconds Scanned': len(df_attack_win),
        '% Windows Flagged': f"{pct_flagged:.2f}%"
    })

# 5. Output Summary Table
summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))

# 6. Export Combined Results
df_attack_win.to_csv('multi_sensitivity_anomalies_output.csv', index=False)