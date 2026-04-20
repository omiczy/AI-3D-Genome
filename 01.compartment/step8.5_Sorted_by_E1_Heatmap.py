import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
import os

def plot_e1_sorted_heatmap():
    # --- 1. 数据加载与精简 ---
    input_file = "Switch_compartment_with_Signals_and_Genes.tsv"
    output_pdf = "step8.5_Sorted_by_E1_Heatmap.pdf"
    
    if not os.path.exists(input_file):
        print(f"❌ 找不到数据文件: {input_file}")
        return

    df_raw = pd.read_csv(input_file, sep='\t')
    target_groups = ['T_CK_vs_T_ET', 'Z_CK_vs_Z_ET']
    df_filtered = df_raw[df_raw['Group'].isin(target_groups)].copy()
    
    # 🌟 移除 RNA，只保留纯表观特征
    marks = ["ATAC", "H3K4me3", "H3K27me3", "CG", "CHG", "CHH"]
    
    print("🛠️ 正在计算 Delta 信号并按照 Delta_E1 排序...")
    delta_list = []
    for _, row in df_filtered.iterrows():
        pref = "TM1" if row['Group'] == 'T_CK_vs_T_ET' else "ZM113"
        res = {'Type': row['compartment_switch'], 
               'Group': row['Group'],
               'Delta_E1': row['E1_sample2'] - row['E1_sample1']}
        for m in marks:
            res[m] = row[f"{pref}_ET_{m}"] - row[f"{pref}_CK_{m}"]
        delta_list.append(res)
    
    df_delta = pd.DataFrame(delta_list)
    # 过滤掉变化微弱的区间 (|Delta_E1| > 0.05)
    df_plot_base = df_delta[df_delta['Delta_E1'].abs() > 0.05].copy()

    # --- 2. 归一化与排序逻辑 ---
    # 🌟 强制 Delta_E1 处于第一列
    plot_cols = ["Delta_E1"] + marks
    plot_data = df_plot_base[plot_cols].copy()

    # 每列独立进行 Robust 缩放以增强对比
    for col in plot_data.columns:
        plot_data[col] = RobustScaler().fit_transform(plot_data[col].values.reshape(-1, 1))

    # --- 3. 按照 Delta_E1 降序排列 (B2A 在上，A2B 在下) ---
    def get_sorted_matrix(group_name):
        sub_df = plot_data[df_plot_base['Group'] == group_name]
        # 🌟 核心：严格按第一列 Delta_E1 排序
        return sub_df.sort_values(by="Delta_E1", ascending=False)

    tm1_matrix = get_sorted_matrix('T_CK_vs_T_ET')
    zm_matrix = get_sorted_matrix('Z_CK_vs_Z_ET')

    # --- 4. 绘图 ---
    print(f"🎨 正在绘制排序热图...")
    plt.rcParams['pdf.fonttype'] = 42
    
    h_tm1, h_zm = len(tm1_matrix), len(zm_matrix)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14), 
                                   gridspec_kw={'height_ratios': [h_tm1, h_zm]})

    cmap = "RdBu_r"
    v_min, v_max = -2, 2

    # A. 绘制 TM1 面板
    sns.heatmap(tm1_matrix, cmap=cmap, center=0, vmin=v_min, vmax=v_max, 
                ax=ax1, cbar=False, yticklabels=False)
    ax1.set_title(f"TM1: Epigenetic Response ordered by $\Delta E1$ (n={h_tm1})", fontweight='bold', fontsize=14)
    ax1.set_ylabel("Bins (Sorted by $\Delta E1$)", fontweight='bold')
    ax1.set_xticklabels([])

    # B. 绘制 ZM113 面板
    sns.heatmap(zm_matrix, cmap=cmap, center=0, vmin=v_min, vmax=v_max, 
                ax=ax2, cbar_kws={'label': 'Scaled Delta Score', 'orientation': 'horizontal', 'pad': 0.08}, 
                yticklabels=False)
    ax2.set_title(f"ZM113: Epigenetic Response ordered by $\Delta E1$ (n={h_zm})", fontweight='bold', fontsize=14)
    ax2.set_ylabel("Bins (Sorted by $\Delta E1$)", fontweight='bold')
    ax2.set_xticklabels(plot_cols, rotation=45, fontweight='bold')

    # 在 B2A 和 A2B 之间画一条分界线（Delta_E1 从正变负的位置）
    def add_split_line(ax, matrix):
        # 找到第一个 Delta_E1 < 0 的索引位置
        # 由于已经标准化，寻找原始值对应的位置或直接找 0 点
        split_idx = (matrix["Delta_E1"] < 0).values.argmax()
        if split_idx > 0:
            ax.axhline(y=split_idx, color='yellow', lw=2, linestyle='--', alpha=0.8)

    add_split_line(ax1, tm1_matrix)
    add_split_line(ax2, zm_matrix)

    plt.tight_layout()
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    print(f"✅ 最终排序热图已生成: {output_pdf}")

if __name__ == "__main__":
    plot_e1_sorted_heatmap()
