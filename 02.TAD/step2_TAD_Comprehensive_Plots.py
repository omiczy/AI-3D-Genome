import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu, spearmanr
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

def run_comprehensive_analysis():
    # ================= 1. 基础配置与数据加载 =================
    fai_file = "ZM113_T2T_V2.genome.fa.fai"
    input_data_file = "step1_combined_TAD_omics_data.tsv"
    
    samples = ['TM1_CK', 'TM1_ET', 'ZM113_CK', 'ZM113_ET']
    colors = {
        'TM1_CK': '#4DBBD5', 'TM1_ET': '#E64B35',
        'ZM113_CK': '#00A087', 'ZM113_ET': '#3C5488'
    }

    print(f"--- 正在读取基础合并数据: {input_data_file} ---")
    if not os.path.exists(input_data_file):
        print(f"错误：未找到数据文件 {input_data_file}。")
        return
        
    df_tad = pd.read_csv(input_data_file, sep='\t')
    
    # 核心修复：强制将关键列转换为浮点数，将隐藏的字符串转换为 NaN
    df_tad['insulation_score'] = pd.to_numeric(df_tad['insulation_score'], errors='coerce')
    omics_cols = [c for c in df_tad.columns if c.endswith('_signal')]
    for c in omics_cols:
        df_tad[c] = pd.to_numeric(df_tad[c], errors='coerce')
        
    mods = [c.replace('_signal', '') for c in omics_cols]

    # ================= 2. Module A: 绘制修复后的箱线图 =================
    print("\n========== [Module A] 正在生成 Boundary vs Interior 箱线图 ==========")
    n_mods = len(mods)
    n_cols = 3
    n_rows = (n_mods + n_cols - 1) // n_cols
    fig_box, axes_box = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    axes_box = axes_box.flatten()

    for i, mod in enumerate(mods):
        ax = axes_box[i]
        col_name = f"{mod}_signal"
        plot_data = df_tad.copy()
        
        if mod == 'RNA':
            plot_data[col_name] = np.log1p(plot_data[col_name])
            ax.set_ylabel(f"log1p({mod} Signal)")
        else:
            ax.set_ylabel(f"{mod} Signal")
            
        sns.boxplot(data=plot_data, x='sample', y=col_name, hue='category', 
                    ax=ax, showfliers=False, order=samples, palette="Set2")
        
        ax.set_title(f"{mod}: Boundary vs Interior")
        ax.tick_params(axis='x', rotation=45)
        
        y_min, y_max_auto = ax.get_ylim()
        y_range = y_max_auto - y_min
        ax.set_ylim(y_min, y_max_auto + y_range * 0.25)
        
        for j, sample in enumerate(samples):
            sub_data = plot_data[plot_data['sample'] == sample]
            b_vals = sub_data[sub_data['category'] == 'Boundary'][col_name].dropna()
            i_vals = sub_data[sub_data['category'] == 'Interior'][col_name].dropna()
            
            n_b, n_i = len(b_vals), len(i_vals)
            if n_b == 0 or n_i == 0: continue
                
            stat, p = mannwhitneyu(b_vals, i_vals, alternative='two-sided')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            
            text = f"p={p:.1e} {sig}\nn={n_b}/{n_i}"
            text_y = y_max_auto + y_range * 0.05 
            ax.text(j, text_y, text, ha='center', va='bottom', fontsize=10, color='black')

    for k in range(n_mods, len(axes_box)):
        fig_box.delaxes(axes_box[k])

    fig_box.tight_layout()
    box_pdf = 'step2_TAD_Boundary_vs_Interior_Boxplots.pdf'
    fig_box.savefig(box_pdf)
    plt.close(fig_box)
    print(f"  --> 箱线图已保存: {box_pdf}")

    # ================= 3. Module B: 全基因组串联平滑趋势与热图 =================
    print("\n========== [Module B] 正在生成全基因组宏观趋势与热图 ==========")
    bin_size = 2000000  # 2Mb 分箱平滑
    sigma = 3  

    if not os.path.exists(fai_file):
        print(f"错误：未找到索引文件 {fai_file}，跳过全基因组图绘制。")
        return

    df_fai = pd.read_csv(fai_file, sep='\t', header=None, names=['chrom', 'length', 'offset', 'linebases', 'linewidth'])
    main_chroms = [f"A{i:02d}" for i in range(1, 14)] + [f"D{i:02d}" for i in range(1, 14)]
    df_fai = df_fai[df_fai['chrom'].isin(main_chroms)].set_index('chrom').reindex(main_chroms)
    df_fai['cum_offset'] = df_fai['length'].shift(1).fillna(0).cumsum()
    total_len = df_fai['length'].sum()

    bins = np.arange(0, total_len + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2 / 1e6

    all_data = {s: {} for s in samples}
    rho_matrix_data = []
    annot_matrix_data = []

    for s in samples:
        df_s = df_tad[(df_tad['sample'] == s) & (df_tad['chrom'].isin(main_chroms))].copy()
        if df_s.empty: continue
            
        df_s['g_pos'] = (df_s['start'] + df_s['end']) / 2 + df_s['chrom'].map(df_fai['cum_offset'])

        valid_insulation = df_s.dropna(subset=['g_pos', 'insulation_score'])
        ins_sum, _ = np.histogram(valid_insulation['g_pos'], bins=bins, weights=valid_insulation['insulation_score'])
        ins_count, _ = np.histogram(valid_insulation['g_pos'], bins=bins)
        ins_avg = np.divide(ins_sum, ins_count, out=np.zeros_like(ins_sum), where=ins_count != 0)
        ins_smooth = gaussian_filter1d(ins_avg, sigma=sigma)
        all_data[s]['insulation_score'] = ins_smooth

        boundaries = df_s[df_s['category'] == 'Boundary']['g_pos'].dropna().values / 1e6
        all_data[s]['boundaries'] = boundaries

        row_rho = {'Sample': s}
        row_annot = {'Sample': s}

        for mod in mods:
            col_name = f"{mod}_signal"
            if col_name in df_s.columns and not df_s[col_name].isna().all():
                valid_sig = df_s.dropna(subset=['g_pos', col_name])
                weights_data = np.log1p(valid_sig[col_name]) if mod == 'RNA' else valid_sig[col_name]
                
                sig_sum, _ = np.histogram(valid_sig['g_pos'], bins=bins, weights=weights_data)
                sig_count, _ = np.histogram(valid_sig['g_pos'], bins=bins)
                sig_avg = np.divide(sig_sum, sig_count, out=np.zeros_like(sig_sum), where=sig_count != 0)
                sig_smooth = gaussian_filter1d(sig_avg, sigma=sigma)
                all_data[s][mod] = sig_smooth

                rho, pval = spearmanr(ins_smooth, sig_smooth)
                row_rho[mod] = rho
                row_annot[mod] = f"{rho:.2f}\n{get_sig_star(pval)}"
            else:
                all_data[s][mod] = np.full_like(bin_centers, np.nan)
                row_rho[mod], row_annot[mod] = np.nan, "N/A"

        rho_matrix_data.append(row_rho)
        annot_matrix_data.append(row_annot)

    plt.rcParams['pdf.fonttype'] = 42
    total_panels = len(mods) + 1
    fig_land, axes_land = plt.subplots(total_panels, 1, figsize=(24, 4 * total_panels), sharex=True)

    def draw_chrom_lines(ax, add_text=False):
        for chrom, row in df_fai.iterrows():
            line_pos = row['cum_offset'] / 1e6
            ax.axvline(x=line_pos, color='black', linestyle=':', linewidth=1, alpha=0.3)
            if add_text:
                y_max = ax.get_ylim()[1]
                ax.text(line_pos + row['length']/2e6, y_max * 1.05, chrom, ha='center', fontweight='bold', fontsize=12)

    ax_ins = axes_land[0]
    for s in samples:
        if 'insulation_score' in all_data[s]:
            ax_ins.plot(bin_centers, all_data[s]['insulation_score'], color=colors[s], linewidth=2.5, label=s)
            b_data = all_data[s].get('boundaries', [])
            if len(b_data) > 0:
                y_base = ax_ins.get_ylim()[0]
                rug_height = (ax_ins.get_ylim()[1] - ax_ins.get_ylim()[0]) * 0.05
                offset = samples.index(s) * rug_height * 1.1
                ax_ins.plot(b_data, np.zeros_like(b_data) + y_base + offset, '|', color=colors[s], alpha=0.5, markersize=10)

    ax_ins.axhline(0, color='grey', linestyle='--', linewidth=1.5, alpha=0.8)
    ax_ins.set_title("TAD Insulation Score Landscape Comparison (Lower means stronger boundary)", loc='left', fontweight='bold', fontsize=15, pad=20)
    ax_ins.set_ylabel("Insulation Score", fontweight='bold', fontsize=12)
    ax_ins.legend(loc='upper right', frameon=True, ncol=4)
    ax_ins.invert_yaxis() 
    draw_chrom_lines(ax_ins, add_text=True)

    for i, mod in enumerate(mods):
        ax = axes_land[i + 1]
        has_data = False
        for s in samples:
            if mod in all_data[s] and not np.isnan(all_data[s][mod]).all():
                has_data = True
                label_str = s
                ax.plot(bin_centers, all_data[s][mod], color=colors[s], linewidth=2.5, alpha=0.85, label=label_str)

        if has_data:
            ax.set_title(f"{mod} Signal Landscape", loc='left', fontweight='bold', fontsize=15, pad=10)
            y_label = f"log1p({mod}) Intensity" if mod == 'RNA' else f"{mod} Intensity"
            ax.set_ylabel(y_label, fontweight='bold', fontsize=12)
            ax.legend(loc='upper left', frameon=False, fontsize=11, ncol=2)
            draw_chrom_lines(ax)
        else:
            ax.text(0.5, 0.5, f"{mod} Data Missing", ha='center', va='center', fontsize=15, transform=ax.transAxes)

    plt.xlabel("Genomic Position along Concatenated Chromosomes (Mb)", fontsize=16, fontweight='bold')
    fig_land.tight_layout()
    land_pdf = "step2_TAD_Genomic_Landscape_MultiPanel.pdf"
    fig_land.savefig(land_pdf)
    plt.close(fig_land)
    print(f"  --> 全景趋势多面板图已保存: {land_pdf}")

    if rho_matrix_data:
        df_rho = pd.DataFrame(rho_matrix_data).set_index('Sample')
        df_annot = pd.DataFrame(annot_matrix_data).set_index('Sample')
        df_rho.to_csv("step2_TAD_Genomic_Rho_Matrix.csv")

        plt.figure(figsize=(10, 6))
        # 💡 核心修改：将 cmap="vlag_r" 替换为 cmap="vlag"
        sns.heatmap(df_rho, annot=df_annot, fmt="", cmap="vlag", center=0,
                    vmin=-0.8, vmax=0.8, linewidths=1.5, linecolor='white',
                    cbar_kws={'label': 'Macro Spearman Rho (Insulation vs Signal)'},
                    annot_kws={"size": 11, "weight": "bold"})

        plt.title("Macro-Scale Correlation: TAD Insulation Score vs Epigenetics", fontsize=15, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()

        heatmap_pdf = "step2_TAD_Genomic_Rho_Heatmap.pdf"
        plt.savefig(heatmap_pdf, dpi=300)
        plt.close()
        print(f"  --> 宏观相关性热图已保存: {heatmap_pdf}")

    print("\n🎉 所有分析图表整合生成完毕！")

if __name__ == "__main__":
    run_comprehensive_analysis()
