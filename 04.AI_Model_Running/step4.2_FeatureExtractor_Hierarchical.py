import pandas as pd
import numpy as np
import pyBigWig
import os
from multiprocessing import Pool
from tqdm import tqdm
import time

# --- 配置 ---
DATA_BASE = "/home/dell/project/HIC_tmp2/02.model/"
# (已移除 GENOME_GFF 和 TPM_FILE，不再需要它们)
NUM_CPUS = 64
OMICS_TYPES = ['ATAC', 'CG', 'CHG', 'CHH', 'H3K4me3', 'H3K27me3']
SAMPLES = ['TM1_CK', 'TM1_ET', 'ZM113_CK', 'ZM113_ET']
TIER_NAMES = ["1_Common_Core", "2_Conserved_High", "3_Responsive_Mid", "4_Individual_All"]

def extract_signals_only(args):
    """提取 12 维表观修饰信号"""
    row, bw_paths = args
    signals = []
    radius = 1500 
    for i in [1, 2]:
        chrom = row[f'chrom{i}']
        mid = int(row[f'mid{i}'])
        s, e = mid - radius, mid + radius
        for omics in OMICS_TYPES:
            bw_p = bw_paths[omics]
            try:
                with pyBigWig.open(bw_p) as bw:
                    val = bw.stats(chrom, max(0, s), min(e, bw.chroms(chrom)), type="mean")[0]
                    signals.append(val if val is not None else 0.0)
            except: 
                signals.append(0.0)
    return signals

def process_tier_sample(tier_name, sample_name):
    coords_file = f"step3_tier_{tier_name}_balanced.csv"
    if not os.path.exists(coords_file): return

    print(f"\n🚀 [Tier: {tier_name}] 处理样本: {sample_name}")
    base_df = pd.read_csv(coords_file)
    bw_paths = {o: os.path.join(DATA_BASE, f"{sample_name}.{o}.merged.bw") for o in OMICS_TYPES}

    # 1. 并行提取表观信号
    tasks = [(row, bw_paths) for _, row in base_df.iterrows()]
    with Pool(NUM_CPUS) as p:
        bw_results = list(tqdm(p.imap(extract_signals_only, tasks), total=len(tasks), desc=f"   - BW Signal"))

    feat_cols = [f"A1_{o}" for o in OMICS_TYPES] + [f"A2_{o}" for o in OMICS_TYPES]
    final_df = pd.concat([base_df.reset_index(drop=True), pd.DataFrame(bw_results, columns=feat_cols)], axis=1)

    # 2. 计算物理特征 (仅保留序列固有属性和物理距离)
    final_df['log_dist'] = np.log10(np.abs(final_df['mid1'] - final_df['mid2']) + 1)

    # 3. 保存 (带 Tier 前缀)
    out_file = f"step4.2_{tier_name}_{sample_name}_features.csv"
    final_df.to_csv(out_file, index=False)
    print(f"✅ 完成 -> {out_file} (已成功剔除 RNA 数据)")

if __name__ == "__main__":
    start_time = time.time()
    print("\n🧬 CottonLoop Hierarchical Feature Extractor (No RNA Mode)")
    
    # 双层循环：4 Tier x 4 Samples = 16 批次
    for tier in TIER_NAMES:
        for s in SAMPLES:
            # 移除了 gff_tree 和 tpm_lookup 的参数传递
            process_tier_sample(tier, s)

    print(f"\n✨ 全部完成！耗时: {time.time() - start_time:.2f}s")
