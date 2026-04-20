import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_curve, auc
import os
import warnings

warnings.filterwarnings('ignore')

def run_merged_ml_pipeline(df, feature_cols, reg_target, class_target, title, out_prefix):
    """
    终极融合管道：
    1. 提取单特征的最佳生物学临界点 (Youden's J statistic)
    2. 运行随机森林回归 (计算表型解释率 R2 和贡献度)
    3. 运行随机森林分类 (计算联合 AUC)
    """
    valid_data = df[feature_cols + [reg_target, class_target]].dropna()
    if len(valid_data) < 100:
        print(f"  [警告] {title} 有效数据不足，跳过。")
        return

    X = valid_data[feature_cols]
    y_reg = valid_data[reg_target]
    y_class = valid_data[class_target]

    # ================= 1. 计算单特征独立 AUC 与临界点 (吸纳 Step 11 方法) =================
    feature_stats = []
    for col in feature_cols:
        x_val = valid_data[col].values
        y_val = y_class.values
        
        fpr, tpr, thresholds = roc_curve(y_val, x_val)
        roc_auc = auc(fpr, tpr)
        direction = ">" # 数值越大，越倾向于目标状态(1)
        
        # 方向自动校正 (例如 H3K27me3 是负相关的，数值越小越倾向于目标状态)
        if roc_auc < 0.5:
            fpr, tpr, thresholds = roc_curve(y_val, -x_val)
            roc_auc = auc(fpr, tpr)
            thresholds = -thresholds
            direction = "<"
            
        # 约登指数寻找最佳临界点
        J = tpr - fpr
        best_idx = np.argmax(J)
        best_thresh = thresholds[best_idx]
        
        clean_name = col.replace('Delta_', '').replace('_CK', '')
        feature_stats.append({
            'Epigenetic_Mark': clean_name,
            'Individual_AUC': roc_auc,
            'Optimal_Cutoff': best_thresh,
            'Direction': direction
        })
    df_stats = pd.DataFrame(feature_stats)

    # 划分训练集和测试集
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_class, test_size=0.2, random_state=42
    )

    # ================= 2. 运行回归模型 (获取 R2 和贡献度) =================
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, yr_train)
    
    yr_pred = rf_reg.predict(X_test)
    r2 = r2_score(yr_test, yr_pred)
    
    reg_importances = rf_reg.feature_importances_ * 100
    df_imp = pd.DataFrame({
        'Epigenetic_Mark': [f.replace('Delta_', '').replace('_CK', '') for f in feature_cols],
        'Contribution_Pct': reg_importances
    })
    
    # 合并贡献度与单特征统计信息
    df_final = pd.merge(df_imp, df_stats, on='Epigenetic_Mark')
    df_final = df_final.sort_values('Contribution_Pct', ascending=True).reset_index(drop=True)

    # 保存详细的数据表格供后续查阅
    df_final.sort_values('Contribution_Pct', ascending=False).to_csv(f"step6.{out_prefix}_Metrics.tsv", sep='\t', index=False)

    # ================= 3. 运行分类模型 (获取联合分类 AUC) =================
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
    rf_clf.fit(X_train, yc_train)
    
    yc_pred_prob = rf_clf.predict_proba(X_test)[:, 1]
    fpr_clf, tpr_clf, thresholds_clf = roc_curve(yc_test, yc_pred_prob)
    global_roc_auc = auc(fpr_clf, tpr_clf)

    print(f"  --> {title} | 回归 R2: {r2:.3f} | 联合分类 AUC: {global_roc_auc:.3f}")

    # ================= 4. 绘制终极 1x2 联合面板图 =================
    plt.rcParams['pdf.fonttype'] = 42
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- 左子图: 机器学习联合 ROC 曲线 ---
    ax1.plot(fpr_clf, tpr_clf, color='#E64B35', lw=2.5, label=f'Global RF Model (AUC = {global_roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title(f'Diagnostic Performance of Multi-Omics\n{title}', fontweight='bold', fontsize=14)
    ax1.legend(loc="lower right", fontsize=12, frameon=True, edgecolor='black')
    ax1.grid(alpha=0.3)

    # --- 右子图: 贡献度条形图 + 最佳物理临界点标注 ---
    bars = ax2.barh(df_final['Epigenetic_Mark'], df_final['Contribution_Pct'], color='#4DBBD5', edgecolor='black', linewidth=1.2)
    
    # 自定义 Y 轴标签，把临界点信息写上去
    new_y_labels = []
    for i, row in df_final.iterrows():
        mark = row['Epigenetic_Mark']
        direction = row['Direction']
        cutoff = row['Optimal_Cutoff']
        ind_auc = row['Individual_AUC']
        # 标签格式: ATAC (Cut: >0.52, AUC:0.75)
        new_y_labels.append(f"{mark}\n(Cut: {direction}{cutoff:.2f}, AUC:{ind_auc:.2f})")
    
    ax2.set_yticks(range(len(df_final)))
    ax2.set_yticklabels(new_y_labels, fontsize=10)

    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                 va='center', ha='left', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel("Relative Contribution to Target Variance (%)", fontsize=12)
    ax2.set_title(f'Feature Importance & Biological Thresholds\nPhenotypic Variance Explained ($R^2$) = {r2:.3f}', fontweight='bold', fontsize=14)
    ax2.set_xlim(0, max(df_final['Contribution_Pct']) * 1.3)

    plt.tight_layout(pad=3.0)
    pdf_out = f"step6.{out_prefix}_Merged_ML_Panel.pdf"
    plt.savefig(pdf_out)
    plt.close()

def run_merged_pipeline():
    samples = ['TM1', 'ZM113']
    mods = ['ATAC', 'H3K4me3', 'H3K27me3', 'CG', 'CHG', 'CHH']
    
    for s in samples:
        data_file = f"step3.{s}_Compartment_Transition_Details.tsv"
        if not os.path.exists(data_file):
            print(f"找不到文件 {data_file}，请先运行 step3。")
            continue
            
        print(f"\n========== 正在执行 {s} 终极融合机器学习与阈值图谱 ==========")
        df = pd.read_csv(data_file, sep='\t')

        # --- 任务 1：静态维持 ---
        print(f"▶ 分析 [静态维持] (回归 E1_CK / 分类 A vs B / 提取截断值)...")
        features_ck = [f"{m}_CK" for m in mods]
        if all(c in df.columns for c in features_ck + ['E1_CK', 'comp_CK']):
            df_static = df.copy()
            df_static['Target_A_vs_B'] = df_static['comp_CK'].map({'A': 1, 'B': 0})
            
            run_merged_ml_pipeline(
                df=df_static, 
                feature_cols=features_ck, 
                reg_target='E1_CK', 
                class_target='Target_A_vs_B', 
                title=f"{s} Compartment (CK state)",
                out_prefix=f"{s}_Static"
            )

        # --- 任务 2：动态翻转 ---
        print(f"▶ 分析 [动态翻转] (回归 Delta_E1 / 分类 B-to-A 激活 / 提取突变截断值)...")
        features_delta = [f"Delta_{m}" for m in mods]
        if all(c in df.columns for c in features_delta + ['Delta_E1', 'Transition']):
            df_dynamic = df[df['comp_CK'] == 'B'].copy()
            df_dynamic['Target_Activation'] = df_dynamic['Transition'].map({'B-to-A': 1, 'B-to-B': 0})
            
            run_merged_ml_pipeline(
                df=df_dynamic, 
                feature_cols=features_delta, 
                reg_target='Delta_E1', 
                class_target='Target_Activation', 
                title=f"{s} Compartment Activation (Cold Transition)",
                out_prefix=f"{s}_Dynamic"
            )

    print("\n✨ 终极方法融合版运行结束！")

if __name__ == "__main__":
    run_merged_pipeline()
