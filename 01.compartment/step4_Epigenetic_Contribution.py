import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import warnings

warnings.filterwarnings('ignore')

def calculate_feature_importance(df, feature_cols, target_col, title, out_prefix):
    """使用随机森林计算纯表观特征的贡献度"""
    valid_data = df[feature_cols + [target_col]].dropna()
    if len(valid_data) < 100:
        print(f"  [警告] {title} 有效数据不足，跳过。")
        return None

    X = valid_data[feature_cols]
    y = valid_data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"  --> {title} 模型解释力 (R2 Score): {r2:.3f}")

    importances = rf.feature_importances_ * 100 
    
    clean_features = [f.replace('Delta_', '').replace('_CK', '') for f in feature_cols]
    
    df_imp = pd.DataFrame({
        'Epigenetic_Mark': clean_features,
        'Contribution_Pct': importances
    }).sort_values('Contribution_Pct', ascending=True)

    plt.figure(figsize=(8, 6))
    bars = plt.barh(df_imp['Epigenetic_Mark'], df_imp['Contribution_Pct'], color='#4DBBD5', edgecolor='black')
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                 va='center', ha='left', fontsize=10, fontweight='bold')

    plt.title(f"{title}\nTotal Variance Explained (R²) = {r2:.2f}", fontweight='bold', fontsize=14)
    plt.xlabel("Relative Contribution to Compartment E1 (%)", fontsize=12)
    plt.ylabel("Epigenetic Features (Upstream Drivers)", fontsize=12)
    plt.xlim(0, max(df_imp['Contribution_Pct']) * 1.2) 
    plt.tight_layout()
    
    # 输出文件统一更新为 step4 前缀
    pdf_out = f"step4.{out_prefix}_Contribution_Barplot.pdf"
    plt.savefig(pdf_out)
    plt.close()
    
    return df_imp

def run_contribution_analysis():
    samples = ['TM1', 'ZM113']
    # 纯表观特征作为驱动力，不包含 RNA
    mods = ['ATAC', 'H3K4me3', 'H3K27me3', 'CG', 'CHG', 'CHH']
    
    for s in samples:
        # 读取 step3 的输出矩阵
        data_file = f"step3.{s}_Compartment_Transition_Details.tsv"
        if not os.path.exists(data_file):
            print(f"找不到文件 {data_file}，请先运行 step3。")
            continue
            
        print(f"\n========== 正在量化 {s} 纯表观修饰对三维结构的驱动贡献 ==========")
        df = pd.read_csv(data_file, sep='\t')

        # --- 任务 1：静态贡献 ---
        print(f"▶ 分析 [静态维持] 贡献度 (CK状态)...")
        features_ck = [f"{m}_CK" for m in mods]
        if all(c in df.columns for c in features_ck + ['E1_CK']):
            calculate_feature_importance(
                df, features_ck, 'E1_CK', 
                title=f"{s} (CK): Epigenetic Contribution to Static E1",
                out_prefix=f"{s}_Static_CK"
            )

        # --- 任务 2：动态贡献 ---
        print(f"▶ 分析 [动态翻转] 贡献度 (冷诱导状态)...")
        features_delta = [f"Delta_{m}" for m in mods]
        if all(c in df.columns for c in features_delta + ['Delta_E1']):
            calculate_feature_importance(
                df, features_delta, 'Delta_E1', 
                title=f"{s} (Transition): Epigenetic Drivers of ΔE1",
                out_prefix=f"{s}_Dynamic_Delta"
            )

    print("\n✨ Step 4 驱动力分析完成！")

if __name__ == "__main__":
    run_contribution_analysis()
