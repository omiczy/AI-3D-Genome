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

# 忽略警告
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

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

def run_anchor_macro_landscape():
    # --- 1. 配置参数 ---
    fai_file = "ZM113_T2T_V2.genome.fa.fai"
    master_table = "step1.Comprehensive_Anchor_Master_Table.tsv"
    
    # 样本配置字典
    samples = {
        'TM1_CK': {'s_pre': 'T_CK', 'color': '#4DBBD5'},
        'TM1_ET': {'s_pre': 'T_ET', 'color': '#E64B35'},
        'ZM113_CK': {'s_pre': 'Z_CK', 'color': '#00A087'},
        'ZM113_ET': {'s_pre': 'Z_ET', 'color': '#3C5488'}
    }
    mods = ['ATAC', 'H3K4me3', 'H3K27me3', 'CG', 'CHG', 'CHH', 'RNA']
    
    bin_size = 2000000  # 2Mb 分箱平滑
    sigma = 3           # 高斯平滑强度

    print(f"--- 正在读取 {fai_file} 构建串联伪基因组坐标 ---")
    if not os.path.exists(fai_file) or not os.path.exists(master_table):
        print("错误：未找到索引文件或主表数据！")
        return
        
    df_fai = pd.read_csv(fai_file, sep='\t', header=None, names=['chrom', 'length', 'offset', 'linebases', 'linewidth'])
    main_chroms = [f"A{i:02d}" for i in range(1, 14)] + [f"D{i:02d}" for i in range(1, 14)]
    df_fai = df_fai[df_fai['chrom'].isin(main_chroms)].set_index('chrom').reindex(main_chroms)
    df_fai['cum_offset'] = df_fai['length'].shift(1).fillna(0).cumsum()
    total_len = df_fai['length'].sum()

    bins = np.arange(0, total_len + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2 / 1e6

    print(f"--- 正在读取 Anchor 主表并进行坐标系转换 ---")
    df_master = pd.read_csv(master_table, sep='\t')
    df_master = df_master[df_master['chr'].isin(main_chroms)].copy()
    df_master['g_pos_tmp'] = (df_master['s'] + df_master['e']) / 2 + df_master['chr'].map(df_fai['cum_offset'])

    # --- 2. 核心数据提取与平滑计算 ---
    # 结构：all_data[sample][mod/Anchor] = smoothed_array
    all_data = {s: {} for s in samples.keys()}
    rho_matrix_data = []
    annot_matrix_data = []

    for s_name, s_cfg in samples.items():
        print(f"  --> 正在处理样本: {s_name} 的宏观分布数据...")
        row_rho = {'Sample': s_name}
        row_annot = {'Sample': s_name}
        
        # A. 计算该样本的 Anchor 密度 (类似 Compartment 的 E1)
        state_col = f"{s_cfg['s_pre']}_S"
        if state_col in df_master.columns:
            s_anchors = df_master[df_master[state_col] == 'YES']
            a_counts, _ = np.histogram(s_anchors['g_pos_tmp'].dropna(), bins=bins)
            a_smooth = gaussian_filter1d(a_counts.astype(float), sigma=sigma)
            all_data[s_name]['Anchor'] = a_smooth
        else:
            all_data[s_name]['Anchor'] = np.full_like(bin_centers, np.nan)
            a_smooth = all_data[s_name]['Anchor']

        # B. 计算该样本各组学的信号强度
        for mod in mods:
            # 动态匹配 step1 的列名
            if mod == 'RNA': sig_col = f"{s_name}_RNA_mean"
            elif mod in ['CG', 'CHG', 'CHH']: sig_col = f"{s_name}_{mod}_site_mean"
            else: sig_col = f"{s_name}_{mod}_mean"

            if sig_col in df_master.columns and not df_master[sig_col].isna().all():
                valid_df = df_master.dropna(subset=['g_pos_tmp', sig_col])
                p_sum, _ = np.histogram(valid_df['g_pos_tmp'], bins=bins, weights=valid_df[sig_col])
                p_count, _ = np.histogram(valid_df['g_pos_tmp'], bins=bins)
                p_avg = np.divide(p_sum, p_count, out=np.zeros_like(p_sum), where=p_count != 0)
                p_smooth = gaussian_filter1d(p_avg, sigma=sigma)
                
                all_data[s_name][mod] = p_smooth
                
                # 计算宏观尺度相关性 (Anchor Density vs Signal)
                if not np.isnan(a_smooth).all():
                    rho, pval = spearmanr(a_smooth, p_smooth)
                    row_rho[mod] = rho
                    row_annot[mod] = f"{rho:.2f}\n{get_sig_star(pval)}"
                    all_data[s_name][f"{mod}_rho"] = rho
                    all_data[s_name][f"{mod}_pval"] = pval
                else:
                    row_rho[mod] = np.nan; row_annot[mod] = "N/A"
            else:
                all_data[s_name][mod] = np.full_like(bin_centers, np.nan)
                row_rho[mod] = np.nan
                row_annot[mod] = "N/A"
                all_data[s_name][f"{mod}_rho"] = np.nan
                all_data[s_name][f"{mod}_pval"] = np.nan
                
        rho_matrix_data.append(row_rho)
        annot_matrix_data.append(row_annot)

    # --- 3. 绘制全基因组多面板趋势图 (Shared X-axis) ---
    print("\n========== 正在绘制多样本同坐标系全景面板图 ==========")
    total_panels = len(mods) + 1
    fig, axes = plt.subplots(total_panels, 1, figsize=(24, 4 * total_panels), sharex=True)
    
    # 染色体参考线辅助函数
    def draw_chrom_lines(ax, add_text=False):
        for chrom, row in df_fai.iterrows():
            line_pos = row['cum_offset'] / 1e6
            ax.axvline(x=line_pos, color='black', linestyle=':', linewidth=1, alpha=0.3)
            if add_text:
                ax.text(line_pos + row['length']/2e6, ax.get_ylim()[1]*1.02, 
                        chrom, ha='center', fontweight='bold', fontsize=12)

    # Panel 0: 绘制 Anchor Density 对比 (基准特征)
    ax_anchor = axes[0]
    for s_name, s_cfg in samples.items():
        if 'Anchor' in all_data[s_name] and not np.isnan(all_data[s_name]['Anchor']).all():
            ax_anchor.plot(bin_centers, all_data[s_name]['Anchor'], color=s_cfg['color'], linewidth=2.5, label=s_name)
            # 添加填充增加视觉厚度
            ax_anchor.fill_between(bin_centers, all_data[s_name]['Anchor'], color=s_cfg['color'], alpha=0.1)
            
    ax_anchor.set_title("Loop Anchor Density Landscape Comparison", loc='left', fontweight='bold', fontsize=16, pad=15)
    ax_anchor.set_ylabel("Anchor Density", fontweight='bold', fontsize=12)
    ax_anchor.legend(loc='upper right', frameon=True, ncol=4)
    draw_chrom_lines(ax_anchor, add_text=True)

    # Panel 1-7: 绘制多组学信号对比
    for i, mod in enumerate(mods):
        ax = axes[i + 1]
        has_data = False
        
        for s_name, s_cfg in samples.items():
            if mod in all_data[s_name] and not np.isnan(all_data[s_name][mod]).all():
                has_data = True
                rho = all_data[s_name].get(f"{mod}_rho", np.nan)
                pval = all_data[s_name].get(f"{mod}_pval", np.nan)
                
                # 动态构建图例标注 (包含与 Anchor Density 的相关性)
                if not np.isnan(rho):
                    p_str = f"P={pval:.1e}" if pval < 0.001 else f"P={pval:.3f}"
                    label_str = f"{s_name} (ρ={rho:.2f}, {p_str})"
                else:
                    label_str = f"{s_name} (N/A)"
                    
                ax.plot(bin_centers, all_data[s_name][mod], color=s_cfg['color'], linewidth=2.5, alpha=0.85, label=label_str)

        if has_data:
            ax.set_title(f"{mod} Signal Landscape", loc='left', fontweight='bold', fontsize=15, pad=10)
            ax.set_ylabel(f"{mod} Intensity", fontweight='bold', fontsize=12)
            ax.legend(loc='upper left', frameon=False, fontsize=11, ncol=2)
            draw_chrom_lines(ax)
        else:
            ax.text(0.5, 0.5, f"{mod} Data Missing", ha='center', va='center', fontsize=15, transform=ax.transAxes)

    plt.xlabel("Genomic Position along Concatenated Chromosomes (Mb)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    pdf_landscape = "step2_Macro_Anchor_Landscape_MultiPanel.pdf"
    plt.savefig(pdf_landscape)
    plt.close()
    print(f"  --> [1] 全景趋势多面板图已保存: {pdf_landscape}")

    # --- 4. 绘制宏观尺度 Rho 相关性矩阵热图 ---
    print("========== 正在生成宏观组学相关性热图 ==========")
    if rho_matrix_data:
        df_rho = pd.DataFrame(rho_matrix_data).set_index('Sample')
        df_annot = pd.DataFrame(annot_matrix_data).set_index('Sample')
        df_rho.to_csv("step2_Macro_Anchor_Rho_Matrix.csv")
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_rho, annot=df_annot, fmt="", cmap="vlag", center=0,
                    vmin=-0.8, vmax=0.8, linewidths=1.5, linecolor='white',
                    cbar_kws={'label': 'Macro Spearman Rho (Anchor vs Signal)'},
                    annot_kws={"size": 11, "weight": "bold"})
        
        plt.title("Macro-Scale Correlation: Loop Anchor Density vs Epigenetics", fontsize=15, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()
        
        pdf_heatmap = "step2_Macro_Anchor_Rho_Heatmap.pdf"
        plt.savefig(pdf_heatmap, dpi=300)
        plt.close()
        print(f"  --> [2] 宏观相关性热图已保存: {pdf_heatmap}")

if __name__ == "__main__":
    run_anchor_macro_landscape()
