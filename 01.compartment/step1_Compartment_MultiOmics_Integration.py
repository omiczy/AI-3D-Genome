import pandas as pd
import numpy as np
import os
import pyBigWig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu
import concurrent.futures
import warnings

warnings.filterwarnings('ignore')

def process_single_bw(bw_file, chroms, starts, ends, col_name):
    """独立进程：极速提取 BW 信号"""
    if not os.path.exists(bw_file):
        return col_name, [np.nan] * len(chroms)
    
    bw = pyBigWig.open(bw_file)
    vals = []
    for c, s, e in zip(chroms, starts, ends):
        try:
            val = bw.stats(c, int(s), int(e), type="mean")[0]
            vals.append(val if val is not None else np.nan)
        except:
            vals.append(np.nan)
    bw.close()
    return col_name, vals

def run_compartment_analysis():
    # --- 1. 配置参数 ---
    samples = ['TM1_CK', 'TM1_ET', 'ZM113_CK', 'ZM113_ET']
    mods = ['ATAC', 'H3K4me3', 'H3K27me3', 'CG', 'CHG', 'CHH', 'RNA'] # RNA 现在作为标准 BW 处理
    ab_colors = {'A': '#E64B35', 'B': '#3C5488'}
    
    all_corr_results = []
    all_matrices = [] 

    max_workers = min(len(mods), os.cpu_count() or 4)
    print(f"--- 启动加速引擎：分配 {max_workers} 个并行进程 ---")

    for s in samples:
        comp_file = f"{s}_compartment.ev.bed.chr"
        if not os.path.exists(comp_file):
            print(f"[{s}] 未找到 Compartment 文件: {comp_file}，跳过。")
            continue
            
        print(f"\n========== 正在处理样本: {s} ==========")
        
        # 1. 读取 100kb Compartment
        df_comp = pd.read_csv(comp_file, sep='\s+', header=0)
        df_comp.columns = ['chrom', 'start', 'end', 'E1', 'compartment']
        df_comp = df_comp.dropna(subset=['E1', 'compartment']).reset_index(drop=True)
        
        df_comp['start'] = df_comp['start'].astype(int)
        df_comp['end'] = df_comp['end'].astype(int)
        
        # 2. 并发提取所有 BigWig 信号 (包含 RNA)
        chroms, starts, ends = df_comp['chrom'].tolist(), df_comp['start'].tolist(), df_comp['end'].tolist()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for mod in mods:
                # 统一读取 .merged.10bp.bw 格式
                bw_file = f"{s}.{mod}.merged.10bp.bw"
                col_name = f"{mod}_mean"
                futures.append(executor.submit(process_single_bw, bw_file, chroms, starts, ends, col_name))
                
            for future in concurrent.futures.as_completed(futures):
                col_name, vals = future.result()
                df_comp[col_name] = vals
            
        # 3. 保存与记录
        out_matrix = f"step1.{s}_Compartment_Matrix.tsv"
        df_comp.to_csv(out_matrix, sep='\t', index=False)
        print(f"  --> 已生成多组学矩阵: {out_matrix}")
        
        df_plot = df_comp.copy()
        df_plot['Sample'] = s
        all_matrices.append(df_plot)

        # 4. 计算 E1 相关性
        sample_corr = {'Sample': s}
        for mod in mods:
            col_name = f"{mod}_mean"
            if col_name in df_comp.columns:
                valid_data = df_comp[['E1', col_name]].dropna()
                sample_corr[mod] = spearmanr(valid_data['E1'], valid_data[col_name])[0] if len(valid_data) > 10 else np.nan
        all_corr_results.append(sample_corr)

    # ================== 第二阶段：完美排版绘图 ==================
    if not all_matrices:
        print("[错误] 未生成任何矩阵数据，绘图终止。")
        return

    print("\n========== 绘制跨样本同坐标轴多面板箱线图 ==========")
    df_global = pd.concat(all_matrices, ignore_index=True)
    
    plt.rcParams['pdf.fonttype'] = 42
    fig, axes = plt.subplots(2, 4, figsize=(26, 14))
    axes = axes.flatten()

    for i, mod in enumerate(mods):
        ax = axes[i]
        col_name = f"{mod}_mean"
        
        plot_data = df_global[['Sample', 'compartment', col_name]].dropna()
        if plot_data.empty:
            ax.text(0.5, 0.5, 'Data Unavailable', ha='center', va='center', fontsize=15)
            ax.set_title(f"{mod} Profile")
            continue

        sns.boxplot(x='Sample', y=col_name, hue='compartment', data=plot_data, 
                    order=samples, hue_order=['A', 'B'], palette=ab_colors, 
                    showfliers=False, width=0.6, ax=ax, legend=(i==0))
        
        custom_x_labels = []
        for s in samples:
            sub_A = plot_data[(plot_data['Sample'] == s) & (plot_data['compartment'] == 'A')][col_name]
            sub_B = plot_data[(plot_data['Sample'] == s) & (plot_data['compartment'] == 'B')][col_name]
            nA, nB = len(sub_A), len(sub_B)
            
            if nA > 0 and nB > 0:
                _, p_val = mannwhitneyu(sub_A, sub_B, alternative='two-sided')
                p_str = f"P={p_val:.1e}" if p_val < 0.001 else f"P={p_val:.3f}"
            else:
                p_str = "P=N/A"
                
            custom_x_labels.append(f"{s}\n{p_str}\n(nA={nA}, nB={nB})")
            
        ax.set_xticks(range(len(samples)))
        ax.set_xticklabels(custom_x_labels, fontsize=10)
        ax.set_title(f"{mod} Intensity in Compartments", fontweight='bold', fontsize=14, pad=10)
        ax.set_ylabel("Mean Intensity", fontsize=12)
        ax.set_xlabel("")
        if i == 0:
            ax.legend(title="Compartment", loc='upper right', frameon=False)

    axes[7].axis('off')
    
    plt.tight_layout(pad=3.0)
    pdf_out = "step1.Global_Compartment_AB_MultiPanel.pdf"
    plt.savefig(pdf_out)
    plt.close()
    print(f"  --> [1] 完美排版多样本箱线图已保存: {pdf_out}")

    # --- 生成 E1 相关性热图 ---
    if all_corr_results:
        df_corr = pd.DataFrame(all_corr_results).set_index('Sample')
        df_corr.to_csv("step1.Global_E1_MultiOmics_Correlation.tsv", sep='\t')
        
        plt.figure(figsize=(9, 6))
        sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    vmin=-0.8, vmax=0.8, linewidths=1, linecolor='white')
        plt.title("Correlation: Compartment E1 Value vs Epigenetics & RNA", fontweight='bold', pad=15)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("step1.Global_E1_Correlation_Heatmap.pdf")
        plt.close()
        print("  --> [2] 热图已保存: step1.Global_E1_Correlation_Heatmap.pdf")

if __name__ == "__main__":
    run_compartment_analysis()
