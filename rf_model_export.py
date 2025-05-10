# rf_model_export.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load data
df = pd.read_csv("arrhythmia_dataset.csv")
features = ['0_pre-RR', '0_post-RR', '0_pPeak', '0_tPeak', '0_rPeak', '0_sPeak', '0_qPeak',
            '0_qrs_interval', '0_pq_interval', '0_qt_interval', '0_st_interval',
            '0_qrs_morph0', '0_qrs_morph1', '0_qrs_morph2', '0_qrs_morph3', '0_qrs_morph4',
            '1_pre-RR', '1_post-RR', '1_pPeak', '1_tPeak', '1_rPeak', '1_sPeak', '1_qPeak',
            '1_qrs_interval', '1_pq_interval', '1_qt_interval', '1_st_interval',
            '1_qrs_morph0', '1_qrs_morph1', '1_qrs_morph2', '1_qrs_morph3', '1_qrs_morph4']
target = "type"  

x = df[features]
y = df[target]

# 2. Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Split & train
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(x_train, y_train)

# 4. Save model and label encoder
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Optional: print accuracy
print("Test Accuracy:", accuracy_score(y_test, model.predict(x_test)))
