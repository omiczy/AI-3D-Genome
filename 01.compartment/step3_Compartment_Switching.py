import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu, kruskal
import os
import itertools
import warnings

warnings.filterwarnings('ignore')

def get_sig_star(p_val):
    if np.isnan(p_val): return "ns"
    if p_val < 0.001: return "***"
    if p_val < 0.01: return "**"
    if p_val < 0.05: return "*"
    return "ns"

def get_cld_letters(data_dict, alpha=0.05):
    groups = [g for g in data_dict.keys() if len(data_dict[g]) > 5]
    if not groups: return {}
    if len(groups) == 1: return {groups[0]: 'a'}
    
    groups.sort(key=lambda g: np.median(data_dict[g]), reverse=True)
    n = len(groups)
    
    p_vals = np.ones((n, n))
    comparisons = n * (n - 1) / 2
    for i in range(n):
        for j in range(i+1, n):
            try:
                _, p = mannwhitneyu(data_dict[groups[i]], data_dict[groups[j]], alternative='two-sided')
            except:
                p = 1.0
            p = min(1.0, p * comparisons)
            p_vals[i, j] = p
            p_vals[j, i] = p

    cliques = []
    subsets = []
    for r in range(n, 0, -1):
        subsets.extend(itertools.combinations(range(n), r))
    
    for sub in subsets:
        is_clique = True
        for i in range(len(sub)):
            for j in range(i+1, len(sub)):
                if p_vals[sub[i], sub[j]] < alpha:
                    is_clique = False
                    break
            if not is_clique: break
        
        if is_clique:
            is_maximal = True
            for c in cliques:
                if set(sub).issubset(set(c)):
                    is_maximal = False
                    break
            if is_maximal:
                cliques.append(sub)
                
    clique_letters = [chr(ord('a') + i) for i in range(len(cliques))]
    res = {groups[i]: "" for i in range(n)}
    for i in range(n):
        for c_idx, c in enumerate(cliques):
            if i in c:
                res[groups[i]] += clique_letters[c_idx]
                
    for g in res:
        res[g] = "".join(sorted(list(res[g])))
        
    return res

def run_switching_analysis():
    comparisons = {
        'TM1': ('TM1_CK', 'TM1_ET'),
        'ZM113': ('ZM113_CK', 'ZM113_ET')
    }
    mods = ['ATAC', 'H3K4me3', 'H3K27me3', 'CG', 'CHG', 'CHH', 'RNA']
    
    trans_colors = {
        'A-to-A': '#E64B35', 'B-to-B': '#3C5488', 
        'B-to-A': '#4DBBD5', 'A-to-B': '#00A087'  
    }
    group_order = ['A-to-A', 'B-to-B', 'A-to-B', 'B-to-A']

    # 两套结果存储
    all_corr_delta, all_pval_delta = [], []
    all_corr_log2fc, all_pval_log2fc = [], []

    # 伪计数：防止计算 log2(0)
    PSEUDOCOUNT = 1e-4

    for prefix, (ck_name, et_name) in comparisons.items():
        file_ck = f"step1.{ck_name}_Compartment_Matrix.tsv"
        file_et = f"step1.{et_name}_Compartment_Matrix.tsv"
        
        if not (os.path.exists(file_ck) and os.path.exists(file_et)):
            print(f"[警告] 找不到 {prefix} 的 CK 或 ET 矩阵，跳过。")
            continue
            
        print(f"\n========== 正在分析转换动态 (双轨制): {ck_name} vs {et_name} ==========")
        
        df_ck = pd.read_csv(file_ck, sep='\t')
        df_et = pd.read_csv(file_et, sep='\t')
        
        cols_to_merge = ['chrom', 'start', 'end', 'E1', 'compartment'] + [f"{m}_mean" for m in mods]
        df_ck = df_ck[cols_to_merge].rename(columns={'E1': 'E1_CK', 'compartment': 'comp_CK'})
        df_et = df_et[cols_to_merge].rename(columns={'E1': 'E1_ET', 'compartment': 'comp_ET'})
        
        for m in mods:
            df_ck.rename(columns={f"{m}_mean": f"{m}_CK"}, inplace=True)
            df_et.rename(columns={f"{m}_mean": f"{m}_ET"}, inplace=True)
            
        df = pd.merge(df_ck, df_et, on=['chrom', 'start', 'end'], how='inner')
        df = df.dropna(subset=['comp_CK', 'comp_ET', 'E1_CK', 'E1_ET'])
        
        # 定义转换状态
        df['Transition'] = df['comp_CK'] + '-to-' + df['comp_ET']
        
        # 计算 E1 的绝对变化 (E1 不计算 log2FC)
        df['Delta_E1'] = df['E1_ET'] - df['E1_CK']
        
        # 计算多组学的 Delta 和 log2FC
        for m in mods:
            if f"{m}_CK" in df.columns and f"{m}_ET" in df.columns:
                df[f'Delta_{m}'] = df[f"{m}_ET"] - df[f"{m}_CK"]
                # 计算 log2 Fold Change
                val_ck = df[f"{m}_CK"].fillna(0) + PSEUDOCOUNT
                val_et = df[f"{m}_ET"].fillna(0) + PSEUDOCOUNT
                df[f'log2FC_{m}'] = np.log2(val_et / val_ck)
            else:
                df[f'Delta_{m}'] = np.nan
                df[f'log2FC_{m}'] = np.nan

        out_bed = f"step3.{prefix}_Compartment_Transition_Details.tsv"
        df.to_csv(out_bed, sep='\t', index=False)
        print(f"  --> 已输出完整转换矩阵 (含 Delta & log2FC): {out_bed}")

        # ---------------- 绘制两套箱线图 ----------------
        for metric_type in ['Delta', 'log2FC']:
            print(f"  --> 正在绘制 {metric_type} 统计箱线图...")
            fig, axes = plt.subplots(2, 4, figsize=(24, 12))
            axes = axes.flatten()
            
            # E1 只在 Delta 图中展示
            plot_mods = ['E1'] + mods if metric_type == 'Delta' else mods
            
            for i, mod in enumerate(plot_mods):
                ax_sub = axes[i]
                col_name = f'{metric_type}_{mod}'
                if col_name not in df.columns or df[col_name].isna().all():
                    ax_sub.axis('off')
                    continue
                    
                plot_data = df[['Transition', col_name]].dropna()
                data_dict = {g: plot_data[plot_data['Transition'] == g][col_name].values for g in group_order}
                
                sns.boxplot(x='Transition', y=col_name, data=plot_data,
                            order=group_order, palette=trans_colors, showfliers=False, ax=ax_sub)
                ax_sub.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
                
                valid_arrays = [data_dict[g] for g in group_order if len(data_dict[g]) > 5]
                if len(valid_arrays) == 4:
                    _, p_kw = kruskal(*valid_arrays)
                    p_kw_str = f"P={p_kw:.1e}" if p_kw < 0.001 else f"P={p_kw:.3f}"
                    cld_letters = get_cld_letters(data_dict) if p_kw < 0.05 else {g: 'a' for g in group_order}
                else:
                    p_kw_str = "N/A"
                    cld_letters = {}

                ymin, ymax = ax_sub.get_ylim()
                y_range = ymax - ymin
                ax_sub.set_ylim(ymin, ymax + y_range * 0.12)
                
                for idx, g in enumerate(group_order):
                    if g in cld_letters and len(data_dict[g]) > 0:
                        q1, q3 = np.percentile(data_dict[g], [25, 75])
                        iqr = q3 - q1
                        upper_whisker = q3 + 1.5 * iqr
                        text_y = min(np.max(data_dict[g]), upper_whisker) + y_range * 0.03
                        ax_sub.text(idx, text_y, cld_letters[g], ha='center', va='bottom', 
                                    fontweight='bold', color='black', fontsize=12)

                title_prefix = "Δ" if metric_type == 'Delta' else "log2FC"
                ax_sub.set_title(f"{title_prefix} {mod}\n(Kruskal-Wallis {p_kw_str})", fontweight='bold', fontsize=14)
                ax_sub.set_ylabel(f"{metric_type} (ET vs CK)", fontsize=12)
                ax_sub.set_xlabel("")
            
            # 关闭多余的空白子图
            for j in range(len(plot_mods), 8):
                axes[j].axis('off')

            plt.tight_layout()
            plt.savefig(f"step3.{prefix}_{metric_type}_Signal_Boxplots.pdf")
            plt.close()

        # ---------------- 计算双轨相关性 ----------------
        print(f"  --> 正在计算 {prefix} 的动态驱动相关性 (Delta E1 vs Delta/log2FC Mods)...")
        row_corr_d, row_pval_d = {'Sample': prefix}, {'Sample': prefix}
        row_corr_l, row_pval_l = {'Sample': prefix}, {'Sample': prefix}
        
        for mod in mods:
            # 1. Delta vs Delta
            col_delta = f'Delta_{mod}'
            if col_delta in df.columns:
                valid_d = df[['Delta_E1', col_delta]].dropna()
                if len(valid_d) > 50:
                    rho, p = spearmanr(valid_d['Delta_E1'], valid_d[col_delta])
                    row_corr_d[mod], row_pval_d[mod] = rho, p
                else:
                    row_corr_d[mod], row_pval_d[mod] = np.nan, np.nan
                    
            # 2. Delta vs log2FC
            col_l2fc = f'log2FC_{mod}'
            if col_l2fc in df.columns:
                valid_l = df[['Delta_E1', col_l2fc]].dropna()
                if len(valid_l) > 50:
                    rho, p = spearmanr(valid_l['Delta_E1'], valid_l[col_l2fc])
                    row_corr_l[mod], row_pval_l[mod] = rho, p
                else:
                    row_corr_l[mod], row_pval_l[mod] = np.nan, np.nan
                    
        all_corr_delta.append(row_corr_d)
        all_pval_delta.append(row_pval_d)
        all_corr_log2fc.append(row_corr_l)
        all_pval_log2fc.append(row_pval_l)

    # ---------------- 绘制双轨全局汇总条形图 ----------------
    def plot_global_bar(corr_list, pval_list, metric_name):
        if not corr_list: return
        df_corr = pd.DataFrame(corr_list).set_index('Sample').T
        df_pval = pd.DataFrame(pval_list).set_index('Sample').T
        
        ax = df_corr.plot(kind='barh', figsize=(11, 7), color=['#4DBBD5', '#E64B35'], width=0.7)
        ax.margins(x=0.25) 
        plt.axvline(0, color='black', linewidth=1)
        
        for i, col in enumerate(df_corr.columns):
            bars = ax.containers[i]
            for j, bar in enumerate(bars):
                width = bar.get_width()
                if np.isnan(width): continue
                
                mod_name = df_corr.index[j]
                p_val = df_pval.loc[mod_name, col]
                
                sig_star = get_sig_star(p_val)
                p_text = f"P={p_val:.1e}" if p_val < 0.001 else f"P={p_val:.3f}"
                
                if width > 0:
                    full_text, ha = f"  {p_text} {sig_star}", 'left'
                else:
                    full_text, ha = f"{sig_star} {p_text}  ", 'right'
                    
                y_pos = bar.get_y() + bar.get_height() / 2
                ax.text(width, y_pos, full_text, va='center', ha=ha, fontsize=10, fontweight='bold', color='#333333')
        
        title = f"Drivers of Switching: Correlation of ΔE1 vs {metric_name} Signal"
        plt.title(title, fontweight='bold', fontsize=15, pad=15)
        plt.xlabel("Spearman Rho (Dynamic Correlation)", fontsize=13)
        plt.ylabel("Epigenetic Modalities", fontsize=13)
        plt.legend(title="Comparison", frameon=False, loc='lower right')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        out_name = f"step3.Global_{metric_name}_Correlation_Barplot.pdf"
        plt.savefig(out_name)
        plt.close()
        print(f"  --> 动态驱动因子图 ({metric_name}) 已保存: {out_name}")

    print("\n========== 生成全局动态驱动修饰条形图 ==========")
    plot_global_bar(all_corr_delta, all_pval_delta, "Delta")
    plot_global_bar(all_corr_log2fc, all_pval_log2fc, "log2FC")

if __name__ == "__main__":
    run_switching_analysis()
