import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, classification_report, roc_curve,
                             precision_recall_curve, f1_score, accuracy_score, average_precision_score)
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import os
import time
import shutil 
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- 核心优化：锁定数学库单线程，防止底层冲突 ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- 彻底解决字体警告：加入 DejaVu Sans 作为首选 ---
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']

TEMP_DIR = os.path.join(os.getcwd(), "joblib_temp_buffer")
if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)
os.environ['JOBLIB_TEMP_FOLDER'] = TEMP_DIR

OUTPUT_DIR_TRAIN = "step5_Hierarchical_Model_Reports"
MODEL_DIR = "model_assets"
TIER_FILE_PREFIX = 'step4.3'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = ['A1_ATAC', 'A1_CG', 'A1_CHG', 'A1_CHH', 'A1_H3K4me3', 'A1_H3K27me3',
                'A2_ATAC', 'A2_CG', 'A2_CHG', 'A2_CHH', 'A2_H3K4me3', 'A2_H3K27me3',
                'gc', 'log_dist']

TIER_ORDER = ["1_Common_Core", "2_Conserved_High", "3_Responsive_Mid", "4_Individual_All"]

for d in [OUTPUT_DIR_TRAIN, MODEL_DIR]:
    if not os.path.exists(d): os.makedirs(d)
for t in TIER_ORDER:
    os.makedirs(os.path.join(OUTPUT_DIR_TRAIN, t), exist_ok=True)

class LoopTransformer(nn.Module):
    def __init__(self, input_dim):
        super(LoopTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        return self.classifier(self.transformer(x).squeeze(1))

class LoopMLP(nn.Module):
    def __init__(self, input_dim):
        super(LoopMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def plot_global_feature_importance(model, name, output_path):
    plt.figure(figsize=(10, 8))
    if hasattr(model, 'feature_importances_'): importances = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'): importances = model.get_feature_importance()
    else: return
    
    importances = (importances / np.sum(importances)) * 100.0
    indices = np.argsort(importances)
    plt.title(f'Relative Contribution to Target Variance\nModel: {name}', fontweight='bold', fontsize=14, pad=15)
    bars = plt.barh(range(len(indices)), importances[indices], color='#3498db', align='center')
    plt.yticks(range(len(indices)), [FEATURE_COLS[i] for i in indices])
    for bar in bars:
        width = bar.get_width()
        plt.text(width + (max(importances)*0.01), bar.get_y() + bar.get_height()/2,
                 f'{width:.2f}%', va='center', fontsize=10, fontweight='bold')
    plt.xlabel('Relative Contribution (%)', fontweight='bold', fontsize=12)
    plt.tight_layout(); plt.savefig(output_path); plt.close()

def plot_tier_specific_importance(importances, model_name, tier_name, output_path):
    plt.figure(figsize=(10, 8))
    indices = np.argsort(importances)
    plt.title(f'Permutation Importance in {tier_name}\nModel: {model_name}', fontweight='bold')
    bars = plt.barh(range(len(indices)), importances[indices], color='#e74c3c', align='center')
    plt.yticks(range(len(indices)), [FEATURE_COLS[i] for i in indices])
    for bar in bars:
        width = bar.get_width()
        x_pos = width if width > 0 else 0
        plt.text(x_pos + (max(abs(importances))*0.01), bar.get_y() + bar.get_height()/2,
                 f'{width:.4f}', va='center', fontsize=9, fontweight='bold')
    plt.xlabel('Mean Accuracy/AUC Drop (Permutation)'); plt.tight_layout(); plt.savefig(output_path); plt.close()

def plot_tier_feature_heatmap(tier_imp_df, model_name, output_path):
    plt.figure(figsize=(10, 8))
    df_norm = tier_imp_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    sns.heatmap(df_norm, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(f"{model_name} Feature Importance Shift Across Tiers")
    plt.ylabel("Features"); plt.xlabel("Hierarchical Tiers")
    plt.tight_layout(); plt.savefig(output_path); plt.close()

def plot_global_roc_pr_grid(all_test_probs, test_df, output_path):
    rows = len(TIER_ORDER); cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(24, 5 * rows)) 
    for i, tier in enumerate(TIER_ORDER):
        mask = test_df['tier'] == tier
        if mask.sum() == 0: continue
        y_true = test_df.loc[mask, 'label']
        
        ax_roc = axes[i, 0]
        for m_name, probs in all_test_probs.items():
            y_score = probs[mask]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_v = roc_auc_score(y_true, y_score)
            ax_roc.plot(fpr, tpr, label=f'{m_name} (AUC={auc_v:.3f})', linewidth=1.5)
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax_roc.set_title(f"ROC: {tier}", fontweight='bold'); ax_roc.legend(loc='lower right', fontsize='x-small'); ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR')

        ax_pr = axes[i, 1]
        for m_name, probs in all_test_probs.items():
            y_score = probs[mask]
            pre, rec, _ = precision_recall_curve(y_true, y_score)
            ap_v = average_precision_score(y_true, y_score)
            ax_pr.plot(rec, pre, label=f'{m_name} (AP={ap_v:.3f})', linewidth=1.5)
        ax_pr.set_title(f"PR: {tier}", fontweight='bold'); ax_pr.legend(loc='lower left', fontsize='x-small'); ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')

        ax_acc = axes[i, 2]; thresholds = np.linspace(0, 1, 101)
        for m_name, probs in all_test_probs.items():
            y_score = probs[mask]; acc_scores = []
            for th in thresholds:
                preds = (y_score > th).astype(int); acc_scores.append(accuracy_score(y_true, preds))
            ax_acc.plot(thresholds, acc_scores, label=f'{m_name} (Max={max(acc_scores):.3f})', linewidth=1.5)
        ax_acc.set_title(f"Acc vs Threshold: {tier}", fontweight='bold'); ax_acc.set_xlabel('Threshold'); ax_acc.set_ylabel('Accuracy'); ax_acc.legend(loc='lower center', fontsize='x-small'); ax_acc.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(output_path); plt.close()

def plot_tier_radar_chart(tier_auc_matrix, output_path):
    categories = TIER_ORDER; N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]; angles += angles[:1]
    plt.figure(figsize=(9, 9)); ax = plt.subplot(111, polar=True)
    markers = ['o', 's', '^', 'v', 'D', 'X']
    for idx, (model_name, tier_scores) in enumerate(tier_auc_matrix.items()):
        values = [tier_scores.get(t, 0.5) for t in TIER_ORDER]; values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, marker=markers[idx % len(markers)])
        ax.fill(angles, values, alpha=0.05)
    plt.xticks(angles[:-1], [t.replace('_', '\n') for t in categories], fontsize=11, fontweight='bold')
    plt.title("Model AUC Robustness Across Hierarchical Tiers", y=1.08, fontsize=14)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1)); plt.savefig(output_path); plt.close()

def run_master_training():
    start_time = time.time()
    print(f"\n{'='*70}\n🚀 Step 5: Master Model Trainer (Evaluation & Metrics)\n{'='*70}")

    train_list = []; test_list = []
    for t_name in TIER_ORDER:
        tr_file = f"{TIER_FILE_PREFIX}_{t_name}_train.csv"; te_file = f"{TIER_FILE_PREFIX}_{t_name}_test.csv"
        if os.path.exists(tr_file) and os.path.exists(te_file):
            t_train = pd.read_csv(tr_file); t_test = pd.read_csv(te_file)
            t_train['tier'] = t_name; t_test['tier'] = t_name
            train_list.append(t_train); test_list.append(t_test)
    if not train_list: raise FileNotFoundError("No training data found! Run step 4.3 first.")
    
    train_df = pd.concat(train_list, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat(test_list, ignore_index=True)

    X_train, y_train = train_df[FEATURE_COLS], train_df['label']
    X_test, y_test = test_df[FEATURE_COLS], test_df['label']

    # --- 这里已经修复，metric_rows_list 初始化为列表 ---
    all_test_probs = {}; tier_auc_matrix = {}; metric_rows_list = []; all_models = {}

    print(f"\n[1/3] 🏋️ Training Machine Learning & Deep Learning Models...")
    ml_configs = {
        "XGBoost": xgb.XGBClassifier(n_estimators=400, max_depth=6, n_jobs=64),
        "LightGBM": lgb.LGBMClassifier(n_estimators=400, n_jobs=64, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=300, thread_count=64, verbose=False),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=64)
    }

    for name, m in tqdm(ml_configs.items(), desc="Training ML Models"):
        m.fit(X_train, y_train)
        all_test_probs[name] = m.predict_proba(X_test)[:, 1]; all_models[name] = m
        plot_global_feature_importance(m, name, f"{OUTPUT_DIR_TRAIN}/{name}_Feature_Relative_Contribution.pdf")
        joblib.dump(m, f"{MODEL_DIR}/step5_{name.lower()}.pkl")

    dl_configs = [("Transformer", LoopTransformer), ("MLP", LoopMLP)]
    for name, ModelClass in dl_configs:
        print(f"   -> Training DL: {name} (Input dim: {len(FEATURE_COLS)})...") 
        net = ModelClass(len(FEATURE_COLS)).to(DEVICE)
        loader = DataLoader(TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), 
                                          torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)), batch_size=4096, shuffle=True)
        opt = optim.Adam(net.parameters(), lr=0.001)
        for _ in tqdm(range(12), desc=f"DL {name}", leave=False):
            net.train()
            for bx, by in loader:
                opt.zero_grad(); nn.BCELoss()(net(bx.to(DEVICE)), by.to(DEVICE)).backward(); opt.step()
        net.eval(); all_models[name] = net
        with torch.no_grad():
            all_test_probs[name] = net(torch.tensor(X_test.values, dtype=torch.float32).to(DEVICE)).cpu().numpy().flatten()
        torch.save(net.state_dict(), f"{MODEL_DIR}/step5_{name.lower()}.pth")

    print(f"\n[2/3] 📊 Evaluating Models & Generating ROC/PR/Metric Charts...")
    summary_file = open(f"{OUTPUT_DIR_TRAIN}/hierarchical_model_summary.txt", "w")
    for t_name in TIER_ORDER:
        t_mask = test_df['tier'] == t_name
        if t_mask.sum() == 0: continue
        y_t = y_test[t_mask]; t_dir = os.path.join(OUTPUT_DIR_TRAIN, t_name)
        summary_file.write(f"--- Tier: {t_name} ---\n")
        plt.figure(figsize=(14, 6)); plt.subplot(1, 2, 1) 
        
        for m_name, probs in all_test_probs.items():
            p_t = probs[t_mask]; preds_t = (p_t > 0.5).astype(int)
            r2_val = (pearsonr(y_t, p_t)[0])**2  
            auc_v = roc_auc_score(y_t, p_t); acc_v = accuracy_score(y_t, preds_t); f1_v = f1_score(y_t, preds_t)
            
            metric_rows_list.append({"Model": m_name, "Tier": t_name, "AUC": auc_v, "Accuracy": acc_v, "F1": f1_v, "R2": r2_val})
            if m_name not in tier_auc_matrix: tier_auc_matrix[m_name] = {}
            tier_auc_matrix[m_name][t_name] = auc_v
            
            summary_file.write(f"[{m_name}] AUC: {auc_v:.4f} | R2: {r2_val:.4f} | Acc: {acc_v:.4f} | F1: {f1_v:.4f}\n")
            fpr, tpr, _ = roc_curve(y_t, p_t)
            plt.plot(fpr, tpr, label=f'{m_name} (AUC={auc_v:.3f})')
            
        plt.plot([0, 1], [0, 1], 'k--'); plt.legend(fontsize='x-small')
        plt.subplot(1, 2, 2)
        for m_name, probs in all_test_probs.items():
            pre, rec, _ = precision_recall_curve(y_t, probs[t_mask]); ap_s = average_precision_score(y_t, probs[t_mask])
            plt.plot(rec, pre, label=f'{m_name} (AP={ap_s:.3f})')
        plt.legend(fontsize='x-small'); plt.savefig(f"{t_dir}/Performance_{t_name}.pdf"); plt.close()
        summary_file.write("\n")
    summary_file.close()

    print(f"\n[3/3] 🧬 Calculating Permutation Importance & Global Visualizations...")
    for name in ["XGBoost", "RandomForest"]:
        if name in all_models:
            tier_imp_df = pd.DataFrame(index=FEATURE_COLS)
            for t_name in TIER_ORDER:
                t_mask = test_df['tier'] == t_name
                if t_mask.sum() < 50: continue
                X_eval = test_df.loc[t_mask, FEATURE_COLS].sample(min(t_mask.sum(), 5000), random_state=42)
                y_eval = test_df.loc[t_mask, 'label'].sample(min(t_mask.sum(), 5000), random_state=42)
                perm = permutation_importance(all_models[name], X_eval, y_eval, n_repeats=5, random_state=42, n_jobs=64) 
                tier_imp_df[t_name] = perm.importances_mean
                plot_tier_specific_importance(perm.importances_mean, name, t_name, f"{OUTPUT_DIR_TRAIN}/{t_name}/{name}_Permutation_Importance.pdf")
            plot_tier_feature_heatmap(tier_imp_df, name, f"{OUTPUT_DIR_TRAIN}/{name}_Tier_Feature_Heatmap.pdf")

    plot_global_roc_pr_grid(all_test_probs, test_df, f"{OUTPUT_DIR_TRAIN}/Global_ROC_PR_Grid.pdf")
    df_metrics = pd.DataFrame(metric_rows_list)
    df_metrics.to_csv(f"{OUTPUT_DIR_TRAIN}/Global_Metric_Summary.csv", index=False)
    
    for m in ["AUC", "R2", "Accuracy", "F1"]:
        plt.figure(figsize=(16, 8))
        ax = sns.barplot(data=df_metrics, x="Tier", y=m, hue="Model", palette="turbo")
        for p in ax.patches:
            if p.get_height() > 0: 
                ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points', 
                            fontsize=8, fontweight='bold', rotation=90)
        if m == "R2":
            plt.ylim(0, df_metrics['R2'].max() * 1.2)
            plt.title(f"Global Hierarchical Performance Summary ({m} - Phenotypic Variance Explained)", fontsize=16, fontweight='bold')
        else:
            plt.ylim(0.5, 1.05)
            plt.title(f"Global Hierarchical Performance Summary ({m})", fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.); plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR_TRAIN}/Metric_Bar_Chart_Labeled_{m}.pdf"); plt.close()

    plot_tier_radar_chart(tier_auc_matrix, f"{OUTPUT_DIR_TRAIN}/Model_Radar_Chart.pdf")

    try: shutil.rmtree(TEMP_DIR)
    except: pass
    print(f"\n✨ Step 5 Complete! Metrics and Plots saved. Total Time: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    run_master_training()
