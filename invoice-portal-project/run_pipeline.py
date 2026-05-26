"""
============================================================
  Invoice Fraud Detection - ONE CLICK RUN PIPELINE
  Dyashin Technosoft Pvt Ltd | ERP Finance Automation
============================================================
  RUN THIS FILE ONLY ONCE - It generates ALL outputs:
    1. Clean dataset            -> output/clean_enriched_dataset.csv
    2. Flagged transactions     -> output/flagged_transactions.csv
    3. Model comparison         -> output/model_comparison.csv
    4. All charts (12 PNGs)     -> output/
    5. Screenshots (all in 1)   -> screenshots.pdf (project root)

  Models:
    Model 1: Invoice Anomaly Detection  (Isolation Forest + DBSCAN + Z-Score)
    Model 2: Fraud Risk Classification  (LR + DT + RF + GB + Ensemble)
============================================================
"""

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
                             roc_auc_score, confusion_matrix, roc_curve, silhouette_score)
from sklearn.cluster import DBSCAN

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(SCRIPT_DIR, 'data', 'fraud_dirty.csv')
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(SCRIPT_DIR, 'fraud_dirty.csv')

C = {'p': '#1a237e', 'f': '#e53935', 'l': '#43a047', 'w': '#fb8c00', 'g': '#e0e0e0'}
MC = {'Logistic Regression': '#1976d2', 'Decision Tree': '#388e3c',
      'Random Forest': '#f57c00', 'Gradient Boosting': '#7b1fa2'}

# ============================================================
# STEP 1: DATA LOADING
# ============================================================
print("=" * 70)
print("  INVOICE FRAUD DETECTION - ML PIPELINE")
print("  Dyashin Technosoft Pvt Ltd | ERP Finance Automation")
print("=" * 70)
print("\n[STEP 1/10] Loading Data...")

df = pd.read_csv(CSV_PATH)
print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Columns: {', '.join(df.columns.tolist())}")

# ============================================================
# STEP 2: DATA CLEANING
# ============================================================
print("\n[STEP 2/10] Data Cleaning...")

df_clean = df.copy()
numeric_cols = ['Invoice_Amount', 'Tax_Amount', 'Total_Amount', 'Processing_Days',
                'Approval_Level', 'Vendor_Reliability_Score', 'Invoice_Frequency',
                'Avg_Invoice_Amount', 'Deviation_From_Avg']
for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Clean Fraud_Flag
fraud_yes = ['yes', 'true', '1', 'high', 'fraud', 'y', 't', 'flagged', 'suspected', 'risk']
fraud_no = ['no', 'false', '0', 'low', 'legitimate', 'n', 'f', 'clean', 'ok', 'not fraud', 'safe']

def clean_flag(v):
    if pd.isna(v): return np.nan
    s = str(v).strip().lower()
    if s in fraud_yes: return 1
    if s in fraud_no: return 0
    return np.nan

df_clean['Fraud_Real'] = df_clean['Fraud_Flag'].apply(clean_flag)
df_clean = df_clean.dropna(subset=['Fraud_Real'])
print(f"  After flag cleaning: {len(df_clean):,} rows")

# Fill missing values
for col in numeric_cols:
    miss = df_clean[col].isnull().sum()
    if miss > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Remove duplicates
df_clean = df_clean.drop_duplicates(subset=['Invoice_ID'], keep='first')
# Fix negatives
df_clean['Invoice_Amount'] = df_clean['Invoice_Amount'].abs()
df_clean['Payment_Status'] = df_clean['Payment_Status'].astype(str).str.strip().str.title()
print(f"  Final cleaned: {len(df_clean):,} rows | Fraud: {(df_clean['Fraud_Real']==1).sum():,} | Legit: {(df_clean['Fraud_Real']==0).sum():,}")

# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================
print("\n[STEP 3/10] Feature Engineering...")

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
print(f"  Features: {len(feature_cols)} | X shape: {X.shape} | Fraud rate: {y.mean()*100:.1f}%")

# ============================================================
# STEP 4: OUTLIER DETECTION (Model 1 - Anomaly Detection)
# ============================================================
print("\n[STEP 4/10] Model 1: Invoice Anomaly Detection...")

scaler_all = StandardScaler()
X_scaled = scaler_all.fit_transform(X)

# Isolation Forest
iso = IsolationForest(n_estimators=200, contamination=0.15, random_state=42, n_jobs=-1)
t0 = time.time()
iso_labels = iso.fit_predict(X_scaled)
iso_scores = iso.decision_function(X_scaled)
print(f"  Isolation Forest: {time.time()-t0:.2f}s | Anomalies: {(iso_labels==-1).sum():,} ({(iso_labels==-1).mean()*100:.1f}%)")

df_clean['Anomaly_Label'] = np.where(iso_labels == -1, 1, 0)
df_clean['Anomaly_Score'] = iso_scores

# DBSCAN
db = DBSCAN(eps=2.0, min_samples=10)
db_labels = db.fit_predict(X_scaled)
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = (db_labels == -1).sum()
sil = silhouette_score(X_scaled, db_labels) if 1 < n_clusters < 50 else 0
print(f"  DBSCAN: {n_clusters} clusters, {n_noise:,} noise points, Silhouette={sil:.4f}")

# Z-Score
z_out = (np.abs(X['Amount_ZScore']) > 2) | (np.abs(X['Processing_ZScore']) > 2) | (np.abs(X['Deviation_From_Avg']) > 50)
print(f"  Z-Score outliers: {z_out.sum():,}")

# Combined anomaly
df_clean['Combined_Anomaly'] = (df_clean['Anomaly_Label'] == 1).astype(int) | (db_labels == -1).astype(int) | z_out.astype(int)
total_fraud = (y == 1).sum()
tp_anom = ((df_clean['Combined_Anomaly']==1) & (y==1)).sum()
print(f"  Combined anomalies: {df_clean['Combined_Anomaly'].sum():,} | Fraud detection rate: {tp_anom}/{total_fraud} ({tp_anom/total_fraud*100:.1f}%)")

# ============================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================
print("\n[STEP 5/10] Train-Test Split...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

# ============================================================
# STEP 6: FRAUD RISK CLASSIFICATION (Model 2)
# ============================================================
print("\n[STEP 6/10] Model 2: Fraud Risk Classification...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=12, random_state=42, min_samples_split=5, min_samples_leaf=2),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    t0 = time.time()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    cv_s = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='accuracy')
    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
        'accuracy': accuracy_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred), 'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob), 'confusion_matrix': confusion_matrix(y_test, y_pred),
        'cv_scores': cv_s, 'cv_mean': cv_s.mean(), 'cv_std': cv_s.std(), 'train_time': time.time()-t0
    }
    print(f"  {name}: Acc={results[name]['accuracy']:.4f} | F1={results[name]['f1_score']:.4f} | AUC={results[name]['roc_auc']:.4f} | CV={results[name]['cv_mean']:.4f} ({time.time()-t0:.1f}s)")

# Ensemble
ens = np.zeros(len(X_test))
for r in results.values():
    ens += r['y_pred']
ens = (ens >= 2.5).astype(int)
ens_acc = accuracy_score(y_test, ens)
ens_f1 = f1_score(y_test, ens)
print(f"  Ensemble (Majority Voting): Acc={ens_acc:.4f} | F1={ens_f1:.4f}")

# ============================================================
# STEP 7: FLAGGED TRANSACTIONS
# ============================================================
print("\n[STEP 7/10] Generating Flagged Transactions...")

fraud_mask = ens == 1
test_pos = np.where(fraud_mask)[0]
flagged_df = df_clean.loc[X_test.index[fraud_mask]].copy()
flagged_df['Risk_Score'] = [int(results['Random Forest']['y_prob'][i]*100) for i in test_pos]
flagged_df['Findings'] = [int(results['Random Forest']['y_prob'][i]*5) for i in test_pos]
flagged_df['Exposure'] = flagged_df['Invoice_Amount']
flagged_df['Anomaly_Detected'] = [df_clean.loc[X_test.index[i], 'Combined_Anomaly'] for i in test_pos]
flagged_df = flagged_df.sort_values('Risk_Score', ascending=False)
print(f"  Flagged: {len(flagged_df)} fraudulent transactions")

# ============================================================
# STEP 8: SAVE CSV OUTPUTS
# ============================================================
print("\n[STEP 8/10] Saving CSV Files...")

flagged_out = flagged_df[['Invoice_ID', 'Vendor_ID', 'Invoice_Date', 'Invoice_Amount',
    'Payment_Status', 'Vendor_Reliability_Score', 'Risk_Score', 'Findings', 'Anomaly_Detected']].head(50)
flagged_out.to_csv(f'{OUTPUT_DIR}/flagged_transactions.csv', index=False)
print(f"  [OK] flagged_transactions.csv")

df_clean.to_csv(f'{OUTPUT_DIR}/clean_enriched_dataset.csv', index=False)
print(f"  [OK] clean_enriched_dataset.csv ({len(df_clean):,} records)")

comparison = []
for name, r in results.items():
    comparison.append({'Model': name, 'Accuracy': round(r['accuracy'], 4), 'Precision': round(r['precision'], 4),
        'Recall': round(r['recall'], 4), 'F1-Score': round(r['f1_score'], 4),
        'ROC-AUC': round(r['roc_auc'], 4), 'CV Mean': round(r['cv_mean'], 4), 'CV Std': round(r['cv_std'], 4)})
comparison.append({'Model': 'Ensemble (Majority Voting)', 'Accuracy': round(ens_acc, 4), 'Precision': '-',
    'Recall': '-', 'F1-Score': round(ens_f1, 4), 'ROC-AUC': '-', 'CV Mean': '-', 'CV Std': '-'})
comparison.append({'Model': 'Isolation Forest (Anomaly)', 'Accuracy': '-', 'Precision': '-',
    'Recall': f'{tp_anom/total_fraud*100:.1f}%', 'F1-Score': '-', 'ROC-AUC': '-', 'CV Mean': '-', 'CV Std': '-'})
pd.DataFrame(comparison).to_csv(f'{OUTPUT_DIR}/model_comparison.csv', index=False)
print(f"  [OK] model_comparison.csv")

# ============================================================
# STEP 9: GENERATE ALL CHARTS
# ============================================================
print("\n[STEP 9/10] Generating Charts...")

# 1. Fraud Distribution
fig, ax = plt.subplots(figsize=(8, 6))
counts = y.value_counts()
wedges, texts, autotexts = ax.pie(counts.values, labels=['Legitimate', 'Fraud'],
    autopct='%1.1f%%', colors=[C['l'], C['f']], startangle=90, explode=(0.03, 0.08),
    shadow=True, textprops={'fontsize': 14, 'fontweight': 'bold'})
for t in autotexts: t.set_fontsize(13); t.set_fontweight('bold')
ax.set_title('Fraud Distribution in Dataset', fontsize=16, fontweight='bold', color=C['p'], pad=20)
ax.text(0, -1.35, f'Total: {len(y):,} records | Fraud: {sum(y==1):,} | Legitimate: {sum(y==0):,}',
        ha='center', fontsize=11, color='gray')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_fraud_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  [OK] 01_fraud_distribution.png")

# 2. Data Quality
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
miss_cols = df.isnull().sum(); miss_cols = miss_cols[miss_cols > 0].sort_values(ascending=True)
if len(miss_cols) > 0:
    axes[0].barh(range(len(miss_cols)), miss_cols.values, color=C['w'], edgecolor='white')
    axes[0].set_yticks(range(len(miss_cols))); axes[0].set_yticklabels(miss_cols.index, fontsize=8)
    axes[0].set_title('Missing Values by Column', fontsize=12, fontweight='bold', color=C['p'])
flag_vals = df['Fraud_Flag'].value_counts().head(12)
axes[1].barh(range(len(flag_vals)), flag_vals.values, color=C['p'], edgecolor='white')
axes[1].set_yticks(range(len(flag_vals))); axes[1].set_yticklabels(flag_vals.index.astype(str), fontsize=8)
axes[1].set_title('Fraud_Flag Values (Raw)', fontsize=12, fontweight='bold', color=C['p'])
clean_c = df_clean['Fraud_Real'].value_counts()
axes[2].bar(['Legitimate', 'Fraud'], clean_c.values, color=[C['l'], C['f']], edgecolor='white', width=0.5)
axes[2].set_title('After Data Cleaning', fontsize=12, fontweight='bold', color=C['p'])
for i, v in enumerate(clean_c.values):
    axes[2].text(i, v + 30, f'{v:,}', ha='center', fontweight='bold', fontsize=11)
plt.suptitle('Data Quality Assessment', fontsize=15, fontweight='bold', color=C['p'], y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_data_quality.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  [OK] 02_data_quality.png")

# 3. Feature Importance
rf_model = results['Random Forest']['model']
importances = rf_model.feature_importances_
feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
fig, ax = plt.subplots(figsize=(10, 7))
feats, imps = [f[0] for f in feat_imp], [f[1] for f in feat_imp]
colors_imp = [C['f'] if i < 3 else C['p'] for i in range(len(feats))]
bars = ax.barh(feats[::-1], imps[::-1], color=colors_imp[::-1], edgecolor='white', height=0.6)
for bar, val in zip(bars, imps[::-1]):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold', color=C['p'], pad=15)
ax.set_xlabel('Importance Score', fontsize=12); ax.set_xlim(0, max(imps)*1.25); ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_feature_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  [OK] 03_feature_importance.png")

# 4. Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax_i, (name, r) in enumerate(results.items()):
    ax = axes[ax_i // 2][ax_i % 2]
    cm = r['confusion_matrix']
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'{name}\nAccuracy: {r["accuracy"]*100:.1f}%', fontsize=12, fontweight='bold', color=C['p'])
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Legitimate', 'Fraud']); ax.set_yticklabels(['Legitimate', 'Fraud'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16, fontweight='bold', color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.suptitle('Confusion Matrices - All 4 Models', fontsize=15, fontweight='bold', color=C['p'], y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_confusion_matrices.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  [OK] 04_confusion_matrices.png")

# 5. ROC Curves
fig, ax = plt.subplots(figsize=(10, 7))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax.plot(fpr, tpr, color=MC[name], lw=2.5, label=f'{name} (AUC = {r["roc_auc"]:.4f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12); ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold', color=C['p'], pad=15)
ax.legend(loc='lower right', fontsize=10, framealpha=0.9); ax.grid(alpha=0.3)
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_roc_curves.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  [OK] 05_roc_curves.png")

# 6. Model Comparison Bar
fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics)); width = 0.18
for i, (name, r) in enumerate(results.items()):
    vals = [r['accuracy'], r['precision'], r['recall'], r['f1_score'], r['roc_auc']]
    ax.bar(x + i * width, vals, width, label=name, color=list(MC.values())[i], edgecolor='white')
ax.set_xlabel('Metrics', fontsize=12); ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', color=C['p'], pad=15)
ax.set_xticks(x + width * 1.5); ax.set_xticklabels(metrics, fontsize=11)
ax.legend(loc='best', fontsize=10); ax.set_ylim(0.85, 1.01); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_model_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  [OK] 06_model_comparison.png")

# 7. Amount Distribution
fig, ax = plt.subplots(figsize=(10, 6))
fa = df_clean[df_clean['Fraud_Real'] == 1]['Invoice_Amount']
la = df_clean[df_clean['Fraud_Real'] == 0]['Invoice_Amount']
ax.hist(la, bins=50, alpha=0.6, color=C['l'], label=f'Legitimate (n={len(la):,})', edgecolor='white')
ax.hist(fa, bins=50, alpha=0.6, color=C['f'], label=f'Fraud (n={len(fa):,})', edgecolor='white')
ax.set_xlabel('Invoice Amount ($)', fontsize=12); ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Invoice Amount Distribution: Fraud vs Legitimate', fontsize=14, fontweight='bold', color=C['p'], pad=15)
ax.legend(loc='best', fontsize=11); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_amount_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  [OK] 07_amount_distribution.png")

# 8. Cross Validation
fig, ax = plt.subplots(figsize=(10, 6))
cv_data = [r['cv_scores'] for r in results.values()]
labels_cv = [f"{name}\n(mean={r['cv_mean']:.4f})" for name, r in results.items()]
bp = ax.boxplot(cv_data, labels=labels_cv, patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], list(MC.values())):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_ylabel('Accuracy Score', fontsize=12)
ax.set_title('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold', color=C['p'], pad=15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_cross_validation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  [OK] 08_cross_validation.png")

# 9. Anomaly Detection
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
colors_sc = np.where(df_clean['Fraud_Real'] == 1, C['f'], C['l'])
ax.scatter(range(len(df_clean)), df_clean['Anomaly_Score'], c=colors_sc, alpha=0.3, s=5)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Record Index'); ax.set_ylabel('Anomaly Score')
ax.set_title('Isolation Forest - Anomaly Scores', fontsize=13, fontweight='bold', color=C['p'])

ax = axes[1]
tn = np.sum((df_clean['Anomaly_Label']==0) & (df_clean['Fraud_Real']==0))
fp = np.sum((df_clean['Anomaly_Label']==1) & (df_clean['Fraud_Real']==0))
fn = np.sum((df_clean['Anomaly_Label']==0) & (df_clean['Fraud_Real']==1))
tp = np.sum((df_clean['Anomaly_Label']==1) & (df_clean['Fraud_Real']==1))
cats = ['True Negative\n(Normal+Legit)', 'False Positive\n(Anomaly+Legit)',
        'False Negative\n(Normal+Fraud)', 'True Positive\n(Anomaly+Fraud)']
bars = ax.bar(cats, [tn, fp, fn, tp], color=[C['l'], C['w'], '#9e9e9e', C['f']], edgecolor='white', width=0.6)
for bar, val in zip(bars, [tn, fp, fn, tp]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20, f'{val:,}', ha='center', fontweight='bold', fontsize=11)
ax.set_ylabel('Count'); ax.set_title('Anomaly Detection vs Actual Fraud', fontsize=13, fontweight='bold', color=C['p'])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_anomaly_detection.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  [OK] 09_anomaly_detection.png")

# 10. Risk Score Distribution
fig, ax = plt.subplots(figsize=(10, 6))
rs = results['Random Forest']['y_prob'] * 100
ax.hist(rs[y_test.values == 0], bins=30, alpha=0.6, color=C['l'], label='Legitimate', edgecolor='white')
ax.hist(rs[y_test.values == 1], bins=30, alpha=0.6, color=C['f'], label='Fraud', edgecolor='white')
ax.axvline(x=50, color='black', linestyle='--', alpha=0.5, label='Threshold (50%)')
ax.set_xlabel('Risk Score (%)', fontsize=12); ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Predicted Risk Score Distribution', fontsize=14, fontweight='bold', color=C['p'], pad=15)
ax.legend(loc='best', fontsize=11); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/10_risk_score_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  [OK] 10_risk_score_distribution.png")

# 11. Data Cleaning Pipeline
fig, ax = plt.subplots(figsize=(10, 5))
stages = ['Raw Data', 'After Flag\nCleaning', 'After Missing\nFill', 'After\nDedup', 'Final\nClean']
svals = [len(df), len(df)-df['Fraud_Flag'].apply(clean_flag).isna().sum(), len(df_clean)+df_clean.shape[0]-len(df_clean), len(df_clean), len(df_clean)]
svals = [len(df), len(df_clean)+((df['Fraud_Flag'].apply(clean_flag).isna().sum())), len(df_clean), len(df_clean), len(df_clean)]
bars = ax.bar(stages, svals, color=[C['w'], '#ff9800', '#ffc107', C['p'], C['p']], edgecolor='white', width=0.6)
for bar, val in zip(bars, svals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50, f'{val:,}', ha='center', fontweight='bold', fontsize=12)
ax.set_ylabel('Records'); ax.set_title('Data Cleaning Pipeline', fontsize=14, fontweight='bold', color=C['p'], pad=15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/11_data_cleaning_pipeline.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  [OK] 11_data_cleaning_pipeline.png")

# 12. Transaction Logs Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
ax.hist(df_clean[df_clean['Fraud_Real']==0]['Processing_Days'], bins=40, alpha=0.6, color=C['l'], label='Legitimate', edgecolor='white')
ax.hist(df_clean[df_clean['Fraud_Real']==1]['Processing_Days'], bins=40, alpha=0.6, color=C['f'], label='Fraud', edgecolor='white')
ax.set_xlabel('Processing Days'); ax.set_ylabel('Frequency')
ax.set_title('Processing Time Distribution', fontsize=13, fontweight='bold', color=C['p'])
ax.legend(loc='best'); ax.grid(axis='y', alpha=0.3)

ax = axes[1]
all_lvls = sorted(df_clean['Approval_Level'].unique())
w = 0.35
ax.bar([l-w/2 for l in all_lvls], [df_clean[(df_clean['Fraud_Real']==0)&(df_clean['Approval_Level']==l)].shape[0] for l in all_lvls],
       w, color=C['l'], label='Legitimate', edgecolor='white')
ax.bar([l+w/2 for l in all_lvls], [df_clean[(df_clean['Fraud_Real']==1)&(df_clean['Approval_Level']==l)].shape[0] for l in all_lvls],
       w, color=C['f'], label='Fraud', edgecolor='white')
ax.set_xlabel('Approval Level'); ax.set_ylabel('Count'); ax.set_xticks(all_lvls)
ax.set_title('Fraud by Approval Level (Transaction Logs)', fontsize=13, fontweight='bold', color=C['p'])
ax.legend(loc='best'); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/12_transaction_logs_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  [OK] 12_transaction_logs_analysis.png")

# ============================================================
# STEP 10: COMBINE INTO SCREENSHOTS.PDF
# ============================================================
print("\n[STEP 10/10] Creating screenshots.pdf...")

try:
    from PIL import Image
    img_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
    if img_files:
        images = []
        for f in img_files:
            images.append(Image.open(os.path.join(OUTPUT_DIR, f)).convert('RGB'))
        screenshots_path = os.path.join(SCRIPT_DIR, 'screenshots.pdf')
        images[0].save(screenshots_path, save_all=True, append_images=images[1:], resolution=150)
        print(f"  [OK] screenshots.pdf ({len(img_files)} charts combined)")
except Exception as e:
    print(f"  [WARN] screenshots.pdf failed: {e}")

# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 70)
print("  ALL DONE! Pipeline completed in ONE run.")
print("=" * 70)
print(f"\n  Output files generated:")
print(f"    output/clean_enriched_dataset.csv  ({len(df_clean):,} records)")
print(f"    output/flagged_transactions.csv    ({len(flagged_out)} transactions)")
print(f"    output/model_comparison.csv        (6 models)")
print(f"    screenshots.pdf                    (12 charts in one file)")
print(f"\n  Models built:")
print(f"    Model 1: Invoice Anomaly Detection  (Isolation Forest + DBSCAN + Z-Score)")
print(f"    Model 2: Fraud Risk Classification  (LR + DT + RF + GB + Ensemble)")
print("=" * 70)
