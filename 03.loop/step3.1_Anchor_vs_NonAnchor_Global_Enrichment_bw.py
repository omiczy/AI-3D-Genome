import pandas as pd
import numpy as np
import pyBigWig
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# --- 核心修改：强制 PDF 和 PS 导出可编辑的矢量字体 ---
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
# ----------------------------------------------------

def get_sig_star(p_val):
    if np.isnan(p_val): return ""
    if p_val < 0.001: return "***"
    if p_val < 0.01: return "**"
    if p_val < 0.05: return "*"
    return "ns"

def get_global_non_anchor_signals(bw_path, genome_fai, exclude_intervals, n_samples, bin_len):
    bw = pyBigWig.open(bw_path)
    chrom_sizes = {r[0]: int(r[1]) for r in genome_fai}
    main_chroms = list(chrom_sizes.keys())
    non_anchor_signals = []
    attempts = 0
    while len(non_anchor_signals) < n_samples and attempts < n_samples * 20:
        attempts += 1
        chrom = random.choice(main_chroms)
        c_size = chrom_sizes[chrom]
        if c_size <= bin_len: continue
        start = random.randint(0, c_size - bin_len)
        end = start + bin_len
        
        is_overlap = False
        if chrom in exclude_intervals:
            for (as_, ae) in exclude_intervals[chrom]:
                if max(start, as_) < min(end, ae):
                    is_overlap = True
                    break
        if not is_overlap:
            try:
                val = bw.stats(chrom, start, end, type="mean")[0]
                if val is not None:
                    non_anchor_signals.append(val)
            except:
                continue
    bw.close()
    return non_anchor_signals

def run_global_enrichment_analysis():
    fai_file = "ZM113_T2T_V2.genome.fa.fai"
    master_table = "step1.Comprehensive_Anchor_Master_Table.tsv"
    samples = ['TM1_CK', 'TM1_ET', 'ZM113_CK', 'ZM113_ET']
    sample_pre = {'TM1_CK': 'T_CK', 'TM1_ET': 'T_ET', 'ZM113_CK': 'Z_CK', 'ZM113_ET': 'Z_ET'}
    mods = ['ATAC', 'H3K4me3', 'H3K27me3', 'RNA', 'CG', 'CHG', 'CHH']

    df_fai = pd.read_csv(fai_file, sep='\t', header=None).values.tolist()
    df_master = pd.read_csv(master_table, sep='\t')
    
    all_anchor_regions = {}
    for _, row in df_master.iterrows():
        c = str(row['chr'])
        if c not in all_anchor_regions: all_anchor_regions[c] = []
        all_anchor_regions[c].append((int(row['s']), int(row['e'])))

    results = []

    for s_real in samples:
        s_p = sample_pre[s_real]
        if f"{s_p}_S" not in df_master.columns: continue
        df_active = df_master[df_master[f"{s_p}_S"] == 'YES']
        n_anchors = len(df_active)
        if n_anchors == 0: continue
        avg_len = int((df_active['e'] - df_active['s']).mean())
        print(f"--- 分析中: {s_real} (Anchor n={n_anchors}) ---")

        for mod in mods:
            bw_file = f"{s_real}.{mod}.merged.bw"
            if not os.path.exists(bw_file): continue
            
            sig_col = f"{s_real}_{mod}_site_mean" if mod in ['CG', 'CHG', 'CHH'] else f"{s_real}_{mod}_mean"
            if sig_col not in df_active.columns: continue
            anchor_vals = df_active[sig_col].dropna().values
            
            print(f"  正在随机抽取全基因组 Non-Anchor 信号 ({mod})...")
            non_anchor_vals = get_global_non_anchor_signals(bw_file, df_fai, all_anchor_regions, n_anchors, avg_len)
            
            if len(anchor_vals) == 0 or not non_anchor_vals: continue
            
            mean_a = np.mean(anchor_vals)
            mean_na = np.mean(non_anchor_vals)
            fe = mean_a / (mean_na + 1e-6)
            t_stat, p_val = ttest_ind(anchor_vals, non_anchor_vals, equal_var=False)
            
            results.append({
                'Sample': s_real, 'Modality': mod, 
                'Anchor_Mean': mean_a, 'NonAnchor_Global_Mean': mean_na, 
                'FE': fe, 'P_value': p_val
            })

    if not results:
        print("无结果生成。")
        return

    df_res = pd.DataFrame(results)
    df_res.to_csv("step3.1.Anchor_vs_NonAnchor_Global_Enrichment.tsv", sep='\t', index=False)
    
    # 动态获取绘图使用的排序
    plot_samples = df_res['Sample'].unique()
    plot_mods = df_res['Modality'].unique()
    
    plt.figure(figsize=(14, 8))
    sns.set_style("ticks")
    
    # 强制排序以确保后续文本标注对齐
    ax = sns.barplot(data=df_res, x='Sample', y='FE', hue='Modality', palette='Set2', order=plot_samples, hue_order=plot_mods)
    plt.axhline(1, color='red', ls='--', label='Random Level (FE=1)', alpha=0.5)
    
    # --- 核心修改：在柱子上标注显著性 P 值 ---
    # 根据 Seaborn 绘图逻辑，patches 的顺序是按 hue(组学) 分大类，再按 x(样本) 排列的
    for i, patch in enumerate(ax.patches):
        height = patch.get_height()
        if np.isnan(height) or height == 0: continue
        
        # 定位对应的数据
        mod_name = plot_mods[i // len(plot_samples)]
        s_name = plot_samples[i % len(plot_samples)]
        
        row = df_res[(df_res['Sample'] == s_name) & (df_res['Modality'] == mod_name)]
        if not row.empty:
            pval = row['P_value'].values[0]
            sig = get_sig_star(pval)
            # 标注位置计算
            x = patch.get_x() + patch.get_width() / 2
            y = height
            # 垂直标注，防止重叠
            ax.text(x, y + df_res['FE'].max()*0.02, sig, ha='center', va='bottom', 
                    fontsize=12, fontweight='bold', color='black', rotation=90)
    
    # 留出顶部空间显示文字
    ax.set_ylim(0, df_res['FE'].max() * 1.25)
    
    plt.title("Global Enrichment: Anchor vs Non-Anchor Regions", fontsize=16, fontweight='bold')
    plt.ylabel("Fold Enrichment (Signal Intensity Ratio)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("step3.1.Anchor_vs_NonAnchor_Global_Barplot.pdf")
    print(f"✨ 任务完成。显著性绘图已保存。")

if __name__ == "__main__":
    run_global_enrichment_analysis()
