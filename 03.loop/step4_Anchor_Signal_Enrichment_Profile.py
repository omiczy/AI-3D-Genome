import pandas as pd
import numpy as np
import pyBigWig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --- 核心修改：强制 PDF 和 PS 导出可编辑的矢量字体 ---
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
# ----------------------------------------------------

# --- 1. 全局配置 ---
INPUT_MASTER = "step1.Comprehensive_Anchor_Master_Table.tsv"
OMICS_TYPES = ['ATAC', 'H3K4me3', 'H3K27me3', 'CG', 'CHG', 'CHH']
SAMPLES = ['TM1_CK', 'TM1_ET', 'ZM113_CK', 'ZM113_ET']
COLORS = {'TM1_CK': '#4DBBD5', 'TM1_ET': '#E64B35', 'ZM113_CK': '#00A087', 'ZM113_ET': '#3C5488'}
WINDOW = 10000   
NBINS = 100      
NUM_CPUS = 16    

def extract_anchor_profile(args):
    chrom, center, bw_path = args
    start = int(center - WINDOW)
    end = int(center + WINDOW)
    if start < 0: return np.full(NBINS, np.nan)
    try:
        with pyBigWig.open(bw_path) as bw:
            if chrom not in bw.chroms(): return np.full(NBINS, np.nan)
            if end > bw.chroms(chrom): return np.full(NBINS, np.nan)
            vals = bw.stats(chrom, start, end, type="mean", nBins=NBINS)
            vals = [v if v is not None else 0.0 for v in vals]
            return np.array(vals)
    except Exception:
        return np.full(NBINS, np.nan)

def run_anchor_enrichment_analysis():
    print(f"--- 正在读取锚点主表数据: {INPUT_MASTER} ---")
    if not os.path.exists(INPUT_MASTER):
        print(f"❌ 错误: 找不到 {INPUT_MASTER}，请确保先运行了 step1。")
        return

    df_master = pd.read_csv(INPUT_MASTER, sep='\t')
    df_master['center'] = (df_master['s'] + df_master['e']) // 2
    profile_results = {o: {} for o in OMICS_TYPES}

    for sample in SAMPLES:
        s_pre = sample.replace('TM1', 'T').replace('ZM113', 'Z').replace('_CK', '_CK_S').replace('_ET', '_ET_S')
        if s_pre in df_master.columns:
            active_anchors = df_master[df_master[s_pre] == 'YES']
        else:
            active_anchors = df_master
            
        coords = active_anchors[['chr', 'center']].values
        print(f"\n[{sample}] 共有 {len(coords)} 个活跃锚点，开始提取多组学轮廓...")
        
        for omics in OMICS_TYPES:
            bw_path = f"{sample}.{omics}.merged.bw"
            if not os.path.exists(bw_path): continue
                
            tasks = [(chrom, center, bw_path) for chrom, center in coords]
            with Pool(NUM_CPUS) as pool:
                matrix = list(tqdm(pool.imap(extract_anchor_profile, tasks), total=len(tasks), desc=f"  - {omics}", leave=False))
            
            matrix = np.array(matrix)
            mean_profile = np.nanmean(matrix, axis=0)
            profile_results[omics][sample] = mean_profile

    print("\n🎨 正在绘制锚点表观修饰富集特征图...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    x_axis = np.linspace(-WINDOW/1000, WINDOW/1000, NBINS)
    
    for i, omics in enumerate(OMICS_TYPES):
        ax = axes[i]
        has_data = False
        for sample in SAMPLES:
            if sample in profile_results[omics]:
                y_data = profile_results[omics][sample]
                if not np.isnan(y_data).all():
                    has_data = True
                    ax.plot(x_axis, y_data, color=COLORS[sample], lw=2.5, label=sample, alpha=0.85)
        
        if has_data:
            ax.set_title(f"{omics} Enrichment at Loop Anchors", fontsize=14, fontweight='bold')
            ax.set_xlabel("Distance from Anchor Center (kb)", fontsize=11)
            ax.set_ylabel("Mean Signal Intensity", fontsize=11)
            ax.axvline(0, color='black', linestyle='--', lw=1.5, alpha=0.7)
            ax.set_xticks([-10, -5, 0, 5, 10])
            ax.set_xticklabels(['-10', '-5', 'Center', '5', '10'])
            ax.legend(loc='best', frameon=False)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            ax.text(0.5, 0.5, f"No Data for {omics}", ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    output_pdf = "step4_Anchor_Epigenetic_Enrichment_Profiles.pdf"
    plt.savefig(output_pdf, dpi=300)
    plt.close()
    
    out_df_list = []
    for omics in OMICS_TYPES:
        for sample in SAMPLES:
            if sample in profile_results[omics]:
                row = {'Omics': omics, 'Sample': sample}
                row.update({f"Bin_{j}": val for j, val in enumerate(profile_results[omics][sample])})
                out_df_list.append(row)
    pd.DataFrame(out_df_list).to_csv("step4_Anchor_Epigenetic_Enrichment_Matrix.csv", index=False)
    print(f"✨ 图表已保存至: {output_pdf}")

if __name__ == "__main__":
    run_anchor_enrichment_analysis()
