import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import pyBigWig
import os
import random
import warnings

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

def get_global_non_anchor_signals(bw_path, chrom_sizes, exclude_intervals, n_samples, bin_len):
    non_anchor_signals = []
    main_chroms = list(chrom_sizes.keys())
    try: bw = pyBigWig.open(bw_path)
    except: return []

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
                    is_overlap = True; break
        if not is_overlap:
            try:
                val = bw.stats(chrom, start, end, type="mean")[0]
                if val is not None: non_anchor_signals.append(val)
            except: continue
    bw.close()
    return non_anchor_signals

def run_anchor_vs_nonanchor_boxplots():
    fai_file = "ZM113_T2T_V2.genome.fa.fai"
    master_table = "step1.Comprehensive_Anchor_Master_Table.tsv"
    samples = ['TM1_CK', 'TM1_ET', 'ZM113_CK', 'ZM113_ET']
    mods = ['ATAC', 'H3K4me3', 'H3K27me3', 'CG', 'CHG', 'CHH']

    if not os.path.exists(fai_file) or not os.path.exists(master_table): return

    chrom_sizes = {}
    with open(fai_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts[0].startswith('A') or parts[0].startswith('D'):
                chrom_sizes[parts[0]] = int(parts[1])

    df_master = pd.read_csv(master_table, sep='\t')
    all_anchor_regions = {}
    for _, row in df_master.iterrows():
        c = str(row['chr'])
        if c not in chrom_sizes: continue
        if c not in all_anchor_regions: all_anchor_regions[c] = []
        all_anchor_regions[c].append((int(row['s']), int(row['e'])))

    plot_data_list = []
    for s_name in samples:
        s_pre = s_name.replace('TM1', 'T').replace('ZM113', 'Z').replace('_CK', '_CK_S').replace('_ET', '_ET_S')
        if s_pre in df_master.columns: df_active = df_master[df_master[s_pre] == 'YES'].copy()
        else: df_active = df_master.copy()
            
        n_anchors = len(df_active)
        if n_anchors == 0: continue
        avg_len = int((df_active['e'] - df_active['s']).mean())

        for mod in mods:
            bw_file = f"{s_name}.{mod}.merged.bw"
            if not os.path.exists(bw_file): continue
            
            sig_col = f"{s_name}_{mod}_site_mean" if mod in ['CG', 'CHG', 'CHH'] else f"{s_name}_{mod}_mean"
            if sig_col not in df_active.columns: continue
            anchor_vals = df_active[sig_col].dropna().values

            non_anchor_vals = get_global_non_anchor_signals(bw_file, chrom_sizes, all_anchor_regions, len(anchor_vals), avg_len)
            for val in anchor_vals: plot_data_list.append({'sample': s_name, 'modality': mod, 'category': 'Anchor', 'signal': val})
            for val in non_anchor_vals: plot_data_list.append({'sample': s_name, 'modality': mod, 'category': 'Non-Anchor', 'signal': val})

    if not plot_data_list: return
    df_plot = pd.DataFrame(plot_data_list)

    n_mods = len(mods); n_cols = 3; n_rows = (n_mods + n_cols - 1) // n_cols
    fig_box, axes_box = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    axes_box = axes_box.flatten()

    for i, mod in enumerate(mods):
        ax = axes_box[i]
        mod_data = df_plot[df_plot['modality'] == mod].copy()
        if mod_data.empty: continue

        sns.boxplot(data=mod_data, x='sample', y='signal', hue='category', 
                    ax=ax, showfliers=False, order=samples, palette=["#E64B35", "#B0B0B0"])
        
        ax.set_title(f"{mod}: Anchor vs Non-Anchor Background", fontweight='bold', pad=15)
        ax.set_ylabel(f"{mod} Intensity")
        ax.tick_params(axis='x', rotation=45)
        
        y_min, y_max_auto = ax.get_ylim()
        y_range = y_max_auto - y_min
        ax.set_ylim(y_min, y_max_auto + y_range * 0.25)
        
        for j, sample in enumerate(samples):
            sub_data = mod_data[mod_data['sample'] == sample]
            a_vals = sub_data[sub_data['category'] == 'Anchor']['signal'].values
            na_vals = sub_data[sub_data['category'] == 'Non-Anchor']['signal'].values
            if len(a_vals) == 0 or len(na_vals) == 0: continue
            stat, p = mannwhitneyu(a_vals, na_vals, alternative='two-sided')
            sig = get_sig_star(p)
            text = f"p={p:.1e}\n{sig}" if p < 0.05 else f"ns\n(p={p:.2f})"
            ax.text(j, y_max_auto + y_range * 0.05, text, ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')

    for k in range(n_mods, len(axes_box)): fig_box.delaxes(axes_box[k])
    fig_box.tight_layout()
    box_pdf = 'step5_Anchor_vs_NonAnchor_Boxplots.pdf'
    fig_box.savefig(box_pdf, dpi=300)
    plt.close(fig_box)
    df_plot.to_csv("step5_Anchor_vs_NonAnchor_PlotData.csv", index=False)
    print(f"✨ 完美！图表已保存至: {box_pdf}")

if __name__ == "__main__":
    run_anchor_vs_nonanchor_boxplots()
