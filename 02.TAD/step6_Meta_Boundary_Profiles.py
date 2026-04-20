import pandas as pd
import numpy as np
import pyBigWig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --- 1. 全局配置 ---
INPUT_TSV = "step1_combined_TAD_omics_data.tsv"  # 读取包含了边界位置的文件

# 💡 新增了 RNA
OMICS_TYPES = ['ATAC', 'H3K4me3', 'H3K27me3', 'CG', 'CHG', 'CHH', 'RNA']
SAMPLES = ['TM1_CK', 'TM1_ET', 'ZM113_CK', 'ZM113_ET']
COLORS = {'TM1_CK': '#4DBBD5', 'TM1_ET': '#E64B35', 'ZM113_CK': '#00A087', 'ZM113_ET': '#3C5488'}

WINDOW = 200000  # 上下游各延伸 200kb
NBINS = 100      # 划分为 100 个区间 (每个 bin 4kb)
NUM_CPUS = 16    # 多线程数，根据服务器配置可调大

# --- 2. 提取单个边界序列的函数 ---
def extract_profile(args):
    chrom, center, bw_path = args
    start = int(center - WINDOW)
    end = int(center + WINDOW)
    
    # 越界保护
    if start < 0: return np.full(NBINS, np.nan)
    
    try:
        with pyBigWig.open(bw_path) as bw:
            if chrom not in bw.chroms():
                return np.full(NBINS, np.nan)
            if end > bw.chroms(chrom):
                return np.full(NBINS, np.nan)
            
            # 使用 pyBigWig 内置的统计方法直接获取 100 个 bin 的均值
            vals = bw.stats(chrom, start, end, type="mean", nBins=NBINS)
            # 处理无信号区域的 None 值为 0
            vals = [v if v is not None else 0.0 for v in vals]
            return np.array(vals)
    except Exception:
        return np.full(NBINS, np.nan)

# --- 3. 主程序 ---
def run_profile_analysis():
    print(f"--- 正在读取边界位置数据: {INPUT_TSV} ---")
    if not os.path.exists(INPUT_TSV):
        print(f"❌ 错误: 找不到 {INPUT_TSV}")
        return

    df = pd.read_csv(INPUT_TSV, sep='\t')
    
    # 获取所有的边界
    boundary_df = df[df['category'] == 'Boundary'].copy()
    boundary_df['center'] = (boundary_df['start'] + boundary_df['end']) // 2
    
    # 用于存储所有样本所有组学的平均轮廓图矩阵
    profile_results = {o: {} for o in OMICS_TYPES}

    # --- 4. 提取数据 ---
    for sample in SAMPLES:
        # 获取该样本自身的 TAD 边界
        s_boundaries = boundary_df[boundary_df['sample'] == sample]
        coords = s_boundaries[['chrom', 'center']].values
        
        print(f"\n[{sample}] 共有 {len(coords)} 个边界，开始提取多组学轮廓...")
        
        for omics in OMICS_TYPES:
            bw_path = f"{sample}.{omics}.merged.bw"
            if not os.path.exists(bw_path):
                print(f"  ⚠️ 找不到 {bw_path}，跳过该组学。")
                continue
                
            tasks = [(chrom, center, bw_path) for chrom, center in coords]
            
            with Pool(NUM_CPUS) as pool:
                matrix = list(tqdm(pool.imap(extract_profile, tasks), total=len(tasks), desc=f"  - {omics}", leave=False))
            
            # 将列表转为矩阵，并计算平均富集曲线 (忽略 NaN)
            matrix = np.array(matrix)
            mean_profile = np.nanmean(matrix, axis=0)
            
            # 💡 针对 RNA 巨大的信号极差，统一做 log1p 平滑处理
            if omics == 'RNA':
                mean_profile = np.log1p(mean_profile)
            
            profile_results[omics][sample] = mean_profile

    # --- 5. 可视化绘图 ---
    print("\n🎨 正在绘制多组学边界富集轮廓图 (Meta-Boundary Profile)...")
    
    # 💡 扩充为 2 行 4 列的布局，适配 7 种组学
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
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
                    # 绘制平滑曲线
                    ax.plot(x_axis, y_data, color=COLORS[sample], lw=2.5, label=sample, alpha=0.85)
        
        if has_data:
            title = f"{omics} Enrichment at Boundaries"
            if omics == 'RNA':
                title = "RNA (log1p) Enrichment at Boundaries"
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel("Distance from TAD Boundary (kb)", fontsize=11)
            ax.set_ylabel("Mean Signal Intensity", fontsize=11)
            
            # 画一条中心垂直虚线
            ax.axvline(0, color='black', linestyle='--', lw=1.5, alpha=0.7)
            
            # 格式化 X 轴
            ax.set_xticks([-200, -100, 0, 100, 200])
            ax.set_xticklabels(['-200', '-100', 'Center', '100', '200'])
            
            if i == 0:  # 只要第一个子图显示图例就可以了，保持画面干净
                ax.legend(loc='best', frameon=False)
            ax.grid(True, linestyle=':', alpha=0.6)
        else:
            ax.text(0.5, 0.5, f"No Data (Missing .bw) for {omics}", ha='center', va='center')

    # 删除多余的第 8 个空白子图
    fig.delaxes(axes[-1])

    plt.tight_layout()
    output_pdf = "step6_Meta_Boundary_Profiles.pdf"
    plt.savefig(output_pdf, dpi=300)
    plt.close()
    
    # 额外保存富集矩阵数据供之后使用
    out_df_list = []
    for omics in OMICS_TYPES:
        for sample in SAMPLES:
            if sample in profile_results[omics]:
                row = {'Omics': omics, 'Sample': sample}
                row.update({f"Bin_{j}": val for j, val in enumerate(profile_results[omics][sample])})
                out_df_list.append(row)
    pd.DataFrame(out_df_list).to_csv("step6_Meta_Boundary_Profile_Matrix.csv", index=False)
    
    print(f"✨ 完成！图表已保存至: {output_pdf}")

if __name__ == "__main__":
    run_profile_analysis()
