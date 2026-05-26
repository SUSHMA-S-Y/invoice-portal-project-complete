"""
================================================================
  MODEL TRAINING & BUILDING DEMO
  Dyashin Technosoft Pvt Ltd | ERP Finance Automation
================================================================
  
  HOW TO USE THIS FILE:
  
  Option A: OPEN & SHOW (for presentation)
  >>> Open this file in any text editor / IDE
  >>> Scroll through - code + output are BOTH visible
  >>> Show the training process, accuracy, and results
  
  Option B: RUN LIVE (if they ask to demonstrate)
  >>> python model_training_demo.py
  >>> It will execute and print all results in real-time
  
  Models covered:
    Model 1: Invoice Anomaly Detection Model (Isolation Forest)
    Model 2: Fraud Risk Classification Model (LR + DT + RF + GB + Ensemble)
================================================================
"""

print()
print("=" * 65)
print("  MODEL TRAINING & BUILDING - LIVE DEMO")
print("  Dyashin Technosoft Pvt Ltd | ERP Finance Automation")
print("=" * 65)
print()

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time, warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)
from sklearn.cluster import DBSCAN

warnings.filterwarnings('ignore')

# ==================== STEP 1: LOAD DATA ====================
print("=" * 65)
print("  STEP 1: LOADING DATASET")
print("=" * 65)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'data', 'fraud_dirty.csv')
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(SCRIPT_DIR, 'fraud_dirty.csv')

df = pd.read_csv(CSV_PATH)
print()
print(f"  Dataset: fraud_dirty.csv")
print(f"  Total Records: {df.shape[0]:,}")
print(f"  Total Columns: {df.shape[1]}")
print(f"  Columns: {', '.join(df.columns.tolist())}")
print()

# ==================== STEP 2: DATA CLEANING ====================
print("=" * 65)
print("  STEP 2: DATA CLEANING")
print("=" * 65)

df_clean = df.copy()
numeric_cols = ['Invoice_Amount', 'Tax_Amount', 'Total_Amount', 'Processing_Days',
                'Approval_Level', 'Vendor_Reliability_Score', 'Invoice_Frequency',
                'Avg_Invoice_Amount', 'Deviation_From_Avg']
for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

fraud_yes = ['yes', 'true', '1', 'high', 'fraud', 'y', 't', 'flagged', 'suspected', 'risk']
fraud_no = ['no', 'false', '0', 'low', 'legitimate', 'n', 'f', 'clean', 'ok', 'not fraud', 'safe']

def clean_flag(v):
    if pd.isna(v): return np.nan
    s = str(v).strip().lower()
    if s in fraud_yes: return 1
    if s in fraud_no: return 0
    return np.nan

df_clean['Fraud_Real'] = df_clean['Fraud_Flag'].apply(clean_flag)
raw_count = len(df_clean)
df_clean = df_clean.dropna(subset=['Fraud_Real'])

for col in numeric_cols:
    miss = df_clean[col].isnull().sum()
    if miss > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

df_clean = df_clean.drop_duplicates(subset=['Invoice_ID'], keep='first')
df_clean['Invoice_Amount'] = df_clean['Invoice_Amount'].abs()

print()
print("  Data Cleaning Steps:")
print(f"    1. Converted {len(numeric_cols)} columns to numeric")
print("    2. Standardized Fraud_Flag (yes/no/true/false -> 1/0)")
print(f"    3. Dropped {raw_count - len(df_clean):,} rows with invalid Fraud_Flag")
print("    4. Filled missing numeric values with median")
print("    5. Removed duplicate Invoice IDs")
print("    6. Fixed negative Invoice Amounts")
print()
print(f"  Result:")
print(f"    Raw:      {len(df):,} records")
print(f"    Cleaned:  {len(df_clean):,} records")
fraud_count = int((df_clean['Fraud_Real']==1).sum())
legit_count = int((df_clean['Fraud_Real']==0).sum())
print(f"    Fraud:    {fraud_count:,} ({fraud_count/len(df_clean)*100:.1f}%)")
print(f"    Legit:    {legit_count:,} ({legit_count/len(df_clean)*100:.1f}%)")
print()

# ==================== STEP 3: FEATURE ENGINEERING ====================
print("=" * 65)
print("  STEP 3: FEATURE ENGINEERING")
print("=" * 65)

mean_amt, std_amt = df_clean['Invoice_Amount'].mean(), df_clean['Invoice_Amount'].std()
df_clean['Amount_ZScore'] = (df_clean['Invoice_Amount'] - mean_amt) / std_amt
df_clean['Deviation_Abs'] = df_clean['Deviation_From_Avg'].abs()
mean_p, std_p = df_clean['Processing_Days'].mean(), df_clean['Processing_Days'].std()
df_clean['Processing_ZScore'] = (df_clean['Processing_Days'] - mean_p) / std_p
df_clean['Tax_Ratio'] = (df_clean['Tax_Amount'] / df_clean['Total_Amount'].replace(0, np.nan)).fillna(0)
df_clean['risk_composite'] = (
    df_clean['Amount_ZScore'].abs() * 0.2 +
    df_clean['Deviation_Abs'] * 0.015 * 0.2 +
    (10 - df_clean['Vendor_Reliability_Score']) / 10 * 0.2 +
    df_clean['Processing_ZScore'].abs() * 0.15 +
    df_clean['Approval_Level'] / 5 * 0.1 +
    (1 - df_clean['Vendor_Reliability_Score'] / 10) * 0.15
)

feature_cols = ['Invoice_Amount', 'Tax_Amount', 'Total_Amount', 'Processing_Days',
    'Approval_Level', 'Vendor_Reliability_Score', 'Invoice_Frequency',
    'Avg_Invoice_Amount', 'Deviation_From_Avg', 'Amount_ZScore',
    'Deviation_Abs', 'Processing_ZScore', 'risk_composite']

X = df_clean[feature_cols].fillna(0)
y = df_clean['Fraud_Real']

print()
print("  Engineered Features (New):")
print("    - Amount_ZScore:       How far invoice amount is from average (standard deviations)")
print("    - Deviation_Abs:       Absolute deviation from vendor's average invoice")
print("    - Processing_ZScore:   How far processing time is from average")
print("    - Tax_Ratio:           Tax as a proportion of total amount")
print("    - risk_composite:      Weighted composite risk score (0 to ~1)")
print()
print(f"  Total Features: {len(feature_cols)}")
print("  Feature List:")
for i, f in enumerate(feature_cols):
    print(f"    {i+1:2d}. {f}")
print()
print(f"  Feature Matrix Shape: {X.shape}")
print(f"  Target Distribution:  Fraud={int(y.sum())} | Legitimate={int((1-y).sum())}")
print()

# ==================== STEP 4: MODEL 1 - ANOMALY DETECTION ====================
print("=" * 65)
print("  STEP 4: MODEL 1 - INVOICE ANOMALY DETECTION MODEL")
print("  (Isolation Forest + DBSCAN + Z-Score)")
print("=" * 65)

scaler_all = StandardScaler()
X_scaled = scaler_all.fit_transform(X)

# 4a. Isolation Forest
print()
print("  --- 4a. Isolation Forest ---")
iso = IsolationForest(n_estimators=200, contamination=0.15, random_state=42, n_jobs=-1)
t0 = time.time()
iso_labels = iso.fit_predict(X_scaled)
iso_time = time.time() - t0
iso_scores = iso.decision_function(X_scaled)

df_clean['Anomaly_Label'] = np.where(iso_labels == -1, 1, 0)
df_clean['Anomaly_Score'] = iso_scores
anomaly_count = int((iso_labels == -1).sum())
total_fraud = int(y.sum())
tp_anom = int(((df_clean['Anomaly_Label']==1) & (y==1)).sum())

print()
print("  Training Parameters:")
print("    - n_estimators: 200 (number of isolation trees)")
print("    - contamination: 0.15 (expected anomaly ratio)")
print("    - random_state: 42")
print()
print("  Results:")
print(f"    Training Time:     {iso_time:.2f} seconds")
print(f"    Total Anomalies:   {anomaly_count:,} out of {len(df_clean):,} ({anomaly_count/len(df_clean)*100:.1f}%)")
print(f"    Normal Records:    {len(df_clean) - anomaly_count:,}")
print()
print("  Overlap with Actual Fraud:")
print(f"    Actual Fraud Cases:    {total_fraud:,}")
print(f"    Detected as Anomaly:   {tp_anom:,} ({tp_anom/total_fraud*100:.1f}% detection rate)")
print(f"    Missed (False Neg):    {total_fraud - tp_anom:,}")
print()

# 4b. DBSCAN
print("  --- 4b. DBSCAN Clustering ---")
db = DBSCAN(eps=2.0, min_samples=10)
t0 = time.time()
db_labels = db.fit_predict(X_scaled)
db_time = time.time() - t0
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = int((db_labels == -1).sum())

from sklearn.metrics import silhouette_score
if 1 < n_clusters < 50:
    sil = silhouette_score(X_scaled, db_labels)
    sil_str = f"{sil:.4f}"
else:
    sil_str = "N/A (clusters too few/many)"

print()
print("  Training Parameters:")
print("    - eps: 2.0 (neighborhood radius)")
print("    - min_samples: 10")
print()
print("  Results:")
print(f"    Training Time:     {db_time:.2f} seconds")
print(f"    Clusters Found:    {n_clusters}")
print(f"    Noise Points:      {n_noise:,} (considered outliers)")
print(f"    Silhouette Score:  {sil_str}")
print()

# 4c. Z-Score
print("  --- 4c. Statistical Z-Score ---")
z_out = (np.abs(X['Amount_ZScore']) > 2) | (np.abs(X['Processing_ZScore']) > 2) | (np.abs(X['Deviation_From_Avg']) > 50)
z_out_count = int(z_out.sum())
print()
print("  Thresholds:")
print("    - Amount Z-Score:      |Z| > 2")
print("    - Processing Z-Score:  |Z| > 2")
print("    - Deviation:           |deviation| > 50")
print()
print(f"  Outliers Found: {z_out_count:,} ({z_out.mean()*100:.1f}%)")
print()

# Combined
df_clean['Combined_Anomaly'] = (
    df_clean['Anomaly_Label'].astype(int) |
    (np.array(db_labels == -1)).astype(int) |
    z_out.astype(int)
)
combined_count = int(df_clean['Combined_Anomaly'].sum())
tp_combined = int(((df_clean['Combined_Anomaly']==1) & (y==1)).sum())

print("  --- Combined Anomaly Detection ---")
print(f"    Total Flagged:  {combined_count:,} ({combined_count/len(df_clean)*100:.1f}%)")
print(f"    Fraud Detected: {tp_combined}/{total_fraud} ({tp_combined/total_fraud*100:.1f}%)")
print()

# ==================== STEP 5: TRAIN-TEST SPLIT ====================
print("=" * 65)
print("  STEP 5: TRAIN-TEST SPLIT (80/20)")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print()
print("  Split Configuration:")
print("    Test Size:     20%")
print("    Random State:  42")
print("    Stratified:    Yes (maintains fraud ratio)")
print()
print(f"  Train Set:  {len(X_train):,} samples  (Fraud: {int(y_train.sum())}, Legit: {int((1-y_train).sum())})")
print(f"  Test Set:   {len(X_test):,} samples   (Fraud: {int(y_test.sum())}, Legit: {int((1-y_test).sum())})")
print()
print("  Scaling: StandardScaler (fit on train, transform on test)")
print()

# ==================== STEP 6: MODEL 2 - CLASSIFICATION ====================
print("=" * 65)
print("  STEP 6: MODEL 2 - FRAUD RISK CLASSIFICATION MODEL")
print("  (4 Classifiers + Ensemble Majority Voting)")
print("=" * 65)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=12, random_state=42, min_samples_split=5, min_samples_leaf=2),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def print_model_results(name, r):
    acc = r['accuracy']; prec = r['precision']; rec = r['recall']
    f1 = r['f1_score']; auc = r['roc_auc']
    cv_m = r['cv_mean']; cv_s2 = r['cv_std']; tt = r['train_time']
    print("  Results:")
    print(f"    Accuracy:      {acc:.4f} ({acc*100:.2f}%)")
    print(f"    Precision:     {prec:.4f}")
    print(f"    Recall:        {rec:.4f}")
    print(f"    F1-Score:      {f1:.4f}")
    print(f"    ROC-AUC:       {auc:.4f}")
    print(f"    CV (5-fold):   {cv_m:.4f} (+/- {cv_s2:.4f})")
    print(f"    Training Time: {tt:.2f}s")

# 6a. Logistic Regression
print()
print("  --- 6a. Logistic Regression ---")
model = models['Logistic Regression']
print("  Parameters: max_iter=1000, C=1.0 (default)")
t0 = time.time()
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]
cv_s = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='accuracy')
results['Logistic Regression'] = {
    'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
    'accuracy': accuracy_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred), 'f1_score': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_prob), 'confusion_matrix': confusion_matrix(y_test, y_pred),
    'cv_scores': cv_s, 'cv_mean': cv_s.mean(), 'cv_std': cv_s.std(), 'train_time': time.time()-t0
}
r = results['Logistic Regression']
print_model_results('Logistic Regression', r)

# 6b. Decision Tree
print()
print("  --- 6b. Decision Tree ---")
model = models['Decision Tree']
print("  Parameters: max_depth=12, min_samples_split=5, min_samples_leaf=2")
t0 = time.time()
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]
cv_s = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='accuracy')
results['Decision Tree'] = {
    'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
    'accuracy': accuracy_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred), 'f1_score': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_prob), 'confusion_matrix': confusion_matrix(y_test, y_pred),
    'cv_scores': cv_s, 'cv_mean': cv_s.mean(), 'cv_std': cv_s.std(), 'train_time': time.time()-t0
}
r = results['Decision Tree']
print_model_results('Decision Tree', r)

# 6c. Random Forest
print()
print("  --- 6c. Random Forest ---")
model = models['Random Forest']
print("  Parameters: n_estimators=200, max_depth=15")
t0 = time.time()
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]
cv_s = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='accuracy')
results['Random Forest'] = {
    'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
    'accuracy': accuracy_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred), 'f1_score': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_prob), 'confusion_matrix': confusion_matrix(y_test, y_pred),
    'cv_scores': cv_s, 'cv_mean': cv_s.mean(), 'cv_std': cv_s.std(), 'train_time': time.time()-t0
}
r = results['Random Forest']
print_model_results('Random Forest', r)

# 6d. Gradient Boosting
print()
print("  --- 6d. Gradient Boosting ---")
model = models['Gradient Boosting']
print("  Parameters: n_estimators=150, learning_rate=0.1, max_depth=5")
t0 = time.time()
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]
cv_s = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='accuracy')
results['Gradient Boosting'] = {
    'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
    'accuracy': accuracy_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred), 'f1_score': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_prob), 'confusion_matrix': confusion_matrix(y_test, y_pred),
    'cv_scores': cv_s, 'cv_mean': cv_s.mean(), 'cv_std': cv_s.std(), 'train_time': time.time()-t0
}
r = results['Gradient Boosting']
print_model_results('Gradient Boosting', r)

# ==================== STEP 7: ENSEMBLE ====================
print()
print("=" * 65)
print("  STEP 7: ENSEMBLE - MAJORITY VOTING")
print("=" * 65)

ens = np.zeros(len(X_test))
for r in results.values():
    ens += r['y_pred']
ens = (ens >= 2.5).astype(int)
ens_acc = accuracy_score(y_test, ens)
ens_f1 = f1_score(y_test, ens)
ens_prec = precision_score(y_test, ens)
ens_rec = recall_score(y_test, ens)

print()
print("  Method: Majority Voting (>= 2 out of 4 models must predict FRAUD)")
print()
print("  Individual Model Votes:")
for idx in range(min(10, len(X_test))):
    votes = []
    for name, r in results.items():
        votes.append('F' if r['y_pred'][idx]==1 else 'L')
    actual = 'FRAUD' if y_test.iloc[idx]==1 else 'LEGIT'
    final = 'FRAUD' if ens[idx]==1 else 'LEGIT'
    match = "OK" if final == actual else "MISS"
    print(f"    Test #{idx+1:3d}: {' | '.join(votes)} -> {final:7s} | Actual: {actual:7s} [{match}]")

print()
print("  Ensemble Performance:")
print(f"    Accuracy:  {ens_acc:.4f} ({ens_acc*100:.2f}%)")
print(f"    Precision: {ens_prec:.4f}")
print(f"    Recall:    {ens_rec:.4f}")
print(f"    F1-Score:  {ens_f1:.4f}")
print()

# ==================== STEP 8: FULL COMPARISON ====================
print("=" * 65)
print("  STEP 8: COMPLETE MODEL COMPARISON")
print("=" * 65)
print()
print("  +----------------------------+----------+----------+----------+----------+----------+----------+")
print("  | Model                      | Accuracy | Precision| Recall   | F1-Score | ROC-AUC  | CV Mean  |")
print("  +----------------------------+----------+----------+----------+----------+----------+----------+")

for name, r in results.items():
    a, p, rc, f, au, cv = r['accuracy'], r['precision'], r['recall'], r['f1_score'], r['roc_auc'], r['cv_mean']
    print(f"  | {name:<26} | {a:>8.4f} | {p:>8.4f} | {rc:>8.4f} | {f:>8.4f} | {au:>8.4f} | {cv:>8.4f} |")

print("  +----------------------------+----------+----------+----------+----------+----------+----------+")
print(f"  | Ensemble (Majority Voting) | {ens_acc:>8.4f} | {ens_prec:>8.4f} | {ens_rec:>8.4f} | {ens_f1:>8.4f} | {'N/A':>8} | {'N/A':>8} |")
print("  +----------------------------+----------+----------+----------+----------+----------+----------+")

best_name = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
best_f1 = results[best_name]['f1_score']
print()
print(f"  Best Individual Classifier: {best_name} (F1 = {best_f1:.4f})")
print(f"  Ensemble F1: {ens_f1:.4f}")
print()

# ==================== STEP 9: FEATURE IMPORTANCE ====================
print("=" * 65)
print("  STEP 9: FEATURE IMPORTANCE (Random Forest)")
print("=" * 65)

rf_model = results['Random Forest']['model']
importances = rf_model.feature_importances_
feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

print()
print(f"  {'Rank':<5} {'Feature':<32} {'Importance':>10}")
print("  " + "-" * 50)
for i, (feat, imp) in enumerate(feat_imp):
    bar = '#' * int(imp * 80)
    print(f"  {i+1:<5} {feat:<32} {imp:>10.4f}  {bar}")
print()

# ==================== STEP 10: CLASSIFICATION REPORT ====================
print("=" * 65)
print("  STEP 10: DETAILED CLASSIFICATION REPORT")
print("=" * 65)

print()
print(f"  --- Best Model: {best_name} ---")
y_pred_best = results[best_name]['y_pred']
print()
print(classification_report(y_test, y_pred_best, target_names=['Legitimate', 'Fraud']))

print(f"  --- Confusion Matrix: {best_name} ---")
cm = results[best_name]['confusion_matrix']
print()
print("                  Predicted")
print("                Legitimate   Fraud")
print("  Actual:")
print(f"    Legitimate    {cm[0][0]:>8}   {cm[0][1]:>5}")
print(f"    Fraud         {cm[1][0]:>8}   {cm[1][1]:>5}")
print()

# ==================== FINAL SUMMARY ====================
print("=" * 65)
print("  MODEL TRAINING & BUILDING - COMPLETE")
print("=" * 65)
print()
print("  Models Built:")
print()
print("    Model 1: Invoice Anomaly Detection Model")
print(f"      - Isolation Forest: {anomaly_count:,} anomalies ({tp_anom/total_fraud*100:.1f}% fraud detection)")
print(f"      - DBSCAN: {n_clusters} clusters, {n_noise:,} noise points")
print(f"      - Z-Score: {z_out_count:,} outliers")
print(f"      - Combined: {combined_count:,} flagged ({tp_combined/total_fraud*100:.1f}% fraud detection)")
print()
print("    Model 2: Fraud Risk Classification Model")

lr_acc = results['Logistic Regression']['accuracy']
dt_acc = results['Decision Tree']['accuracy']
rf_acc = results['Random Forest']['accuracy']
gb_acc = results['Gradient Boosting']['accuracy']
print(f"      - Logistic Regression:  Accuracy={lr_acc:.4f}")
print(f"      - Decision Tree:        Accuracy={dt_acc:.4f}")
print(f"      - Random Forest:        Accuracy={rf_acc:.4f}")
print(f"      - Gradient Boosting:    Accuracy={gb_acc:.4f}")
print(f"      - Ensemble Voting:      Accuracy={ens_acc:.4f}, F1={ens_f1:.4f}")
print()
print(f"  Dataset Used: {len(df_clean):,} cleaned records")
print(f"  Features Used: {len(feature_cols)}")
print(f"  Best Accuracy: {best_name} ({best_f1*100:.2f}%)")
print()
print("=" * 65)
