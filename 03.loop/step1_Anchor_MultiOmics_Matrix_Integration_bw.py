import pandas as pd
import numpy as np
import os
import pyBigWig

def run_anchor_integration_bw_main():
    # --- 1. 配置 ---
    loop_file = "Loop_PET.txt"
    tpm_file = "TPM.matrix.txt"
    # BigWig 修饰类型（此时 RNA 的平均值由 TPM 矩阵提供，故此处可不含 RNA 的 BW 逻辑）
    bw_mod_configs = ['ATAC', 'H3K4me3', 'H3K27me3', 'CG', 'CHG', 'CHH']
    dmr_files = {'T': 'MR_TM1_Diff.txt', 'Z': 'MR_ZM113_Diff.txt'}
    samples = ['TM1_CK', 'TM1_ET', 'ZM113_CK', 'ZM113_ET']

    # --- 2. 状态聚合与基因关联逻辑 ---
    print("Step 1: 正在从 Loop 提取 Anchor 并关联基因信息...")
    df_l = pd.read_csv(loop_file, sep='\t')
    
    # 列名映射配置 (包含 gene1 和 gene2)
    cols_map = {
        'chrom1':'chr','start1':'s','end1':'e',
        'TM1_CK_state':'T_CK_S','TM1_ET_state':'T_ET_S','ZM113_CK_state':'Z_CK_S','ZM113_ET_state':'Z_ET_S',
        'TM1_CK_PET':'T_CK_P','TM1_ET_PET':'T_ET_P','ZM113_CK_PET':'Z_CK_P','ZM113_ET_PET':'Z_ET_P',
        'gene1':'Gene'
    }
    
    # 提取 A1 和 A2 锚点
    a1 = df_l[list(cols_map.keys())].rename(columns=cols_map)
    a2_cols = ['chrom2','start2','end2'] + list(cols_map.keys())[3:-1] + ['gene2']
    a2 = df_l[a2_cols].rename(columns={
        'chrom2':'chr','start2':'s','end2':'e', 'gene2':'Gene', **cols_map
    })
    
    # 定义聚合逻辑
    agg_logic = {col: (lambda x: 'YES' if 'YES' in x.values else 'NO') for col in ['T_CK_S','T_ET_S','Z_CK_S','Z_ET_S']}
    agg_logic.update({col: 'sum' for col in ['T_CK_P','T_ET_P','Z_CK_P','Z_ET_P']})
    # 合并基因名（去重、去空、逗号分隔）
    agg_logic['Gene'] = lambda x: ",".join(set(str(i) for i in x if pd.notna(i) and str(i).lower() != 'nan'))
    
    df_master = pd.concat([a1, a2]).groupby(['chr','s','e']).agg(agg_logic).reset_index()
    df_master = df_master.sort_values(['chr', 's']).reset_index(drop=True)

    # --- 3. 映射 TPM 表达数据 (RNA) ---
    if os.path.exists(tpm_file):
        print(f"Step 2: 正在从 {tpm_file} 映射基因表达数据 (RNA_mean)...")
        # 读取 TPM 矩阵并设置 Gene_ID 为索引
        df_tpm = pd.read_csv(tpm_file, sep='\s+')
        tpm_dict = df_tpm.set_index('Gene_ID').to_dict('index')
        
        for s in samples:
            col_name = f"{s}_RNA_mean"
            tpm_results = []
            
            for _, row in df_master.iterrows():
                # 分离锚点关联的多个基因
                genes = [g.strip() for g in str(row['Gene']).split(',') if g.strip()]
                # 获取存在于 TPM 矩阵中的基因表达值
                vals = [tpm_dict[g][s] for g in genes if g in tpm_dict]
                # 计算均值，若无有效基因则填 nan
                tpm_results.append(np.mean(vals) if vals else np.nan)
            
            df_master[col_name] = tpm_results
    else:
        print(f"  [跳过] 未找到 TPM 文件: {tpm_file}")

    # --- 4. 映射 BigWig 信号 ---
    for mod in bw_mod_configs:
        print(f"Step 3: 正在从 BigWig 映射 {mod} 信号...")
        for s in samples:
            bw_path = f"{s}.{mod}.merged.bw"
            mean_col = f"{s}_{mod}_mean" if mod in ['ATAC', 'H3K4me3', 'H3K27me3'] else f"{s}_{mod}_site_mean"
            sum_col = f"{s}_{mod}_sum"
            
            if not os.path.exists(bw_path):
                continue
            
            bw = pyBigWig.open(bw_path)
            means = []
            sums = []
            
            for _, row in df_master.iterrows():
                chrom, start, end = row['chr'], int(row['s']), int(row['e'])
                try:
                    val = bw.stats(chrom, start, end, type="mean")[0]
                    if val is None:
                        means.append(np.nan); sums.append(0.0)
                    else:
                        means.append(val); sums.append(val * (end - start))
                except:
                    means.append(np.nan); sums.append(0.0)
            
            df_master[mean_col] = means
            if mod in ['ATAC', 'H3K4me3', 'H3K27me3']:
                df_master[sum_col] = sums
            bw.close()

    # --- 5. 映射 DMR 数据 ---
    for sp_key, path in dmr_files.items():
        prefix = 'TM1' if sp_key == 'T' else 'ZM113'
        if not os.path.exists(path): continue
        print(f"Step 4: 正在映射 {prefix} DMR 数据...")
        df_d = pd.read_csv(path, sep='\t')
        for t in ['CK', 'ET']:
            col = f"{prefix}_{t}_meanMethy"
            d_s, d_c = np.zeros(len(df_master)), np.zeros(len(df_master))
            for chrom, group in df_d.groupby('chrom'):
                idx = df_master[df_master['chr'] == chrom].index
                if len(idx) == 0: continue
                a_s, a_e = df_master.loc[idx, 's'].values, df_master.loc[idx, 'e'].values
                for _, row in group.iterrows():
                    mask = (a_s < row['end']) & (a_e > row['start'])
                    d_s[idx[mask]] += row[col]
                    d_c[idx[mask]] += 1
            df_master[f"{prefix}_{t}_DMR_mean"] = np.divide(d_s, d_c, out=np.full_like(d_s, np.nan), where=d_c > 0)
            df_master[f"{prefix}_{t}_DMR_sum"] = d_s

    # --- 6. 保存最终主表 ---
    final_output = "step1.Comprehensive_Anchor_Master_Table.tsv"
    df_master.to_csv(final_output, sep='\t', index=False)
    print(f"\n[任务完成] 大矩阵已生成（包含基因 TPM 表达数据）。")

if __name__ == "__main__":
    run_anchor_integration_bw_main()
