import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# 1. Load Datasets
df_normal = pd.read_csv('airport_datasets_excel2.csv')
df_attack = pd.read_csv('airport_IoT_anomalies_dataset.csv')

def extract_features(df):
    """
    Extracts numerical ML features from Wireshark captures:
    - Length: Packet size in bytes
    - Time_Diff: Time gap between consecutive packets
    - Packet_Rate: Rolling count of packets per second
    """
    df_feat = df[['Time', 'Length']].copy()
    
    # Calculate time interval between packets
    df_feat['Time_Diff'] = df_feat['Time'].diff().fillna(0)
    
    # Approximate packet frequency (rolling window)
    df_feat['Packet_Rate'] = df_feat['Time_Diff'].rolling(window=10, min_periods=1).mean()
    
    return df_feat[['Length', 'Time_Diff', 'Packet_Rate']]

# 2. Extract Features
X_train = extract_features(df_normal)
X_test = extract_features(df_attack)

# 3. Train Isolation Forest Model on Normal Data
# 'contamination' represents the expected proportion of anomalies in training data
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(X_train)

# 4. Predict on Anomaly Dataset
# Returns -1 for anomalies, 1 for normal packets
df_attack['Anomaly_Score'] = model.decision_function(X_test)
df_attack['Is_Anomaly'] = model.predict(X_test)
df_attack['Is_Anomaly'] = df_attack['Is_Anomaly'].map({1: 'Normal', -1: 'Anomaly'})

# 5. Results Summary
anomaly_count = (df_attack['Is_Anomaly'] == 'Anomaly').sum()
total_packets = len(df_attack)

print(f"Total Packets Analyzed: {total_packets}")
print(f"Anomalies Flagged: {anomaly_count} ({(anomaly_count/total_packets)*100:.2f}%)")

# Save processed results to display on your dashboard
df_attack.to_csv('flagged_anomalies_output.csv', index=False)