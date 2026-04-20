import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def get_sig_star(p_val):
    if np.isnan(p_val): return ""
    if p_val < 0.001: return "***"
    if p_val < 0.01: return "**"
    if p_val < 0.05: return "*"
    return "ns"

def run_compartment_landscape():
    # --- 1. 配置参数 ---
    fai_file = "ZM113_T2T_V2.genome.fa.fai"
    samples = ['TM1_CK', 'TM1_ET', 'ZM113_CK', 'ZM113_ET']
    colors = {
        'TM1_CK': '#4DBBD5', 'TM1_ET': '#E64B35', 
        'ZM113_CK': '#00A087', 'ZM113_ET': '#3C5488'
    }
    mods = ['ATAC', 'H3K4me3', 'H3K27me3', 'CG', 'CHG', 'CHH', 'RNA']
    
    bin_size = 2000000  # 2Mb 分箱平滑，适合宏观展示
    sigma = 3  # 高斯平滑强度

    print(f"--- 正在读取 {fai_file} 构建串联伪基因组坐标 ---")
    if not os.path.exists(fai_file):
        print(f"错误：未找到索引文件 {fai_file}，请确保与该脚本在同级目录。")
        return
        
    df_fai = pd.read_csv(fai_file, sep='\t', header=None, names=['chrom', 'length', 'offset', 'linebases', 'linewidth'])
    main_chroms = [f"A{i:02d}" for i in range(1, 14)] + [f"D{i:02d}" for i in range(1, 14)]
    df_fai = df_fai[df_fai['chrom'].isin(main_chroms)].set_index('chrom').reindex(main_chroms)
    df_fai['cum_offset'] = df_fai['length'].shift(1).fillna(0).cumsum()
    total_len = df_fai['length'].sum()

    bins = np.arange(0, total_len + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2 / 1e6

    # --- 2. 核心数据提取与平滑计算 ---
    all_data = {s: {} for s in samples}
    rho_matrix_data = []
    annot_matrix_data = []

    for s in samples:
        matrix_file = f"step1.{s}_Compartment_Matrix.tsv"
        if not os.path.exists(matrix_file):
            print(f"[警告] 找不到矩阵文件: {matrix_file}，请先成功运行 step1 脚本。")
            continue
            
        print(f"  --> 正在平滑处理样本: {s} 的宏观分布数据...")
        df = pd.read_csv(matrix_file, sep='\t')
        df = df[df['chrom'].isin(main_chroms)].copy()
        df['g_pos'] = (df['start'] + df['end']) / 2 + df['chrom'].map(df_fai['cum_offset'])

        # 计算 E1 平滑线
        valid_e1 = df.dropna(subset=['g_pos', 'E1'])
        e1_sum, _ = np.histogram(valid_e1['g_pos'], bins=bins, weights=valid_e1['E1'])
        e1_count, _ = np.histogram(valid_e1['g_pos'], bins=bins)
        e1_avg = np.divide(e1_sum, e1_count, out=np.zeros_like(e1_sum), where=e1_count != 0)
        e1_smooth = gaussian_filter1d(e1_avg, sigma=sigma)
        all_data[s]['E1'] = e1_smooth

        # 计算多组学信号平滑线及相关性
        row_rho = {'Sample': s}
        row_annot = {'Sample': s}
        
        for mod in mods:
            col_name = f"{mod}_mean"
            if col_name in df.columns and not df[col_name].isna().all():
                valid_sig = df.dropna(subset=['g_pos', col_name])
                sig_sum, _ = np.histogram(valid_sig['g_pos'], bins=bins, weights=valid_sig[col_name])
                sig_count, _ = np.histogram(valid_sig['g_pos'], bins=bins)
                sig_avg = np.divide(sig_sum, sig_count, out=np.zeros_like(sig_sum), where=sig_count != 0)
                sig_smooth = gaussian_filter1d(sig_avg, sigma=sigma)
                
                all_data[s][mod] = sig_smooth
                
                rho, pval = spearmanr(e1_smooth, sig_smooth)
                row_rho[mod] = rho
                row_annot[mod] = f"{rho:.2f}\n{get_sig_star(pval)}"
                all_data[s][f"{mod}_rho"] = rho
                all_data[s][f"{mod}_pval"] = pval
            else:
                all_data[s][mod] = np.full_like(bin_centers, np.nan)
                row_rho[mod] = np.nan
                row_annot[mod] = "N/A"
                all_data[s][f"{mod}_rho"] = np.nan
                all_data[s][f"{mod}_pval"] = np.nan
                
        rho_matrix_data.append(row_rho)
        annot_matrix_data.append(row_annot)

    # --- 3. 绘制全基因组多面板趋势图 ---
    print("\n========== 正在绘制多样本同坐标系全景面板图 ==========")
    plt.rcParams['pdf.fonttype'] = 42
    
    total_panels = len(mods) + 1
    fig, axes = plt.subplots(total_panels, 1, figsize=(24, 4 * total_panels), sharex=True)
    
    def draw_chrom_lines(ax, add_text=False):
        for chrom, row in df_fai.iterrows():
            line_pos = row['cum_offset'] / 1e6
            ax.axvline(x=line_pos, color='black', linestyle=':', linewidth=1, alpha=0.3)
            if add_text:
                ax.text(line_pos + row['length']/2e6, ax.get_ylim()[1]*1.02, 
                        chrom, ha='center', fontweight='bold', fontsize=12)

    # Panel 0: E1
    ax_e1 = axes[0]
    for s in samples:
        if 'E1' in all_data[s]:
            ax_e1.plot(bin_centers, all_data[s]['E1'], color=colors[s], linewidth=2.5, label=s)
    ax_e1.axhline(0, color='grey', linestyle='--', linewidth=1.5, alpha=0.8)
    ax_e1.set_title("Compartment E1 Landscape Comparison", loc='left', fontweight='bold', fontsize=15, pad=15)
    ax_e1.set_ylabel("E1 Value\n(>0:A, <0:B)", fontweight='bold', fontsize=12)
    ax_e1.legend(loc='upper right', frameon=True, ncol=4)
    draw_chrom_lines(ax_e1, add_text=True)

    # Panel 1-7: 各组学
    for i, mod in enumerate(mods):
        ax = axes[i + 1]
        has_data = False
        
        for s in samples:
            if mod in all_data[s] and not np.isnan(all_data[s][mod]).all():
                has_data = True
                rho = all_data[s].get(f"{mod}_rho", np.nan)
                pval = all_data[s].get(f"{mod}_pval", np.nan)
                
                if not np.isnan(rho):
                    p_str = f"P={pval:.1e}" if pval < 0.001 else f"P={pval:.3f}"
                    label_str = f"{s} (ρ={rho:.2f}, {p_str})"
                else:
                    label_str = f"{s} (N/A)"
                    
                ax.plot(bin_centers, all_data[s][mod], color=colors[s], linewidth=2.5, alpha=0.85, label=label_str)

        if has_data:
            ax.set_title(f"{mod} Signal Landscape", loc='left', fontweight='bold', fontsize=15, pad=10)
            ax.set_ylabel(f"{mod} Intensity", fontweight='bold', fontsize=12)
            ax.legend(loc='upper left', frameon=False, fontsize=11, ncol=2)
            draw_chrom_lines(ax)
        else:
            ax.text(0.5, 0.5, f"{mod} Data Missing", ha='center', va='center', fontsize=15, transform=ax.transAxes)

    plt.xlabel("Genomic Position along Concatenated Chromosomes (Mb)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    pdf_landscape = "step2_Macro_Genomic_Landscape_MultiPanel.pdf"
    plt.savefig(pdf_landscape)
    plt.close()
    print(f"  --> [1] 全景趋势多面板图已保存: {pdf_landscape}")

    # --- 4. 绘制宏观尺度 Rho 热图 ---
    print("========== 正在生成宏观组学相关性热图 ==========")
    if rho_matrix_data:
        df_rho = pd.DataFrame(rho_matrix_data).set_index('Sample')
        df_annot = pd.DataFrame(annot_matrix_data).set_index('Sample')
        
        df_rho.to_csv("step2_Macro_Genomic_Rho_Matrix.csv")
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_rho, annot=df_annot, fmt="", cmap="vlag", center=0,
                    vmin=-0.8, vmax=0.8, linewidths=1.5, linecolor='white',
                    cbar_kws={'label': 'Macro Spearman Rho (E1 vs Signal)'},
                    annot_kws={"size": 11, "weight": "bold"})
        
        plt.title("Macro-Scale Correlation: Compartment E1 vs Epigenetics", fontsize=15, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()
        
        pdf_heatmap = "step2_Macro_Genomic_Rho_Heatmap.pdf"
        plt.savefig(pdf_heatmap, dpi=300)
        plt.close()
        print(f"  --> [2] 宏观相关性热图已保存: {pdf_heatmap}")

if __name__ == "__main__":
    run_compartment_landscape()
