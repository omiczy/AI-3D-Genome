import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kruskal, mannwhitneyu
import os
import warnings

warnings.filterwarnings('ignore')

# --- 1. 配置参数 ---
INPUT_OMICS = "step1_combined_TAD_omics_data.tsv"
DIFF_FILES = {
    'TM1': 'T_CK_vs_T_ET.diff_boundary.xls.chr',
    'ZM113': 'Z_CK_vs_Z_ET.diff_boundary.xls.chr'
}

OMICS_TYPES = ['ATAC', 'H3K4me3', 'H3K27me3', 'CG', 'CHG', 'CHH', 'RNA']
COLORS_BAR = {'TM1': '#4DBBD5', 'ZM113': '#E64B35'}
GROUP_ORDER = ['stable', 'strengthened', 'weakened', 'appeared', 'disappeared']
COLORS_BOX = ['#95a5a6', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

def get_sig_stars(p_val):
    if p_val < 0.001: return "***"
    elif p_val < 0.01: return "**"
    elif p_val < 0.05: return "*"
    else: return "ns"

def get_letters(data_dict):
    valid_groups = [g for g in GROUP_ORDER if g in data_dict and len(data_dict[g]) > 5]
    if not valid_groups: return {g: "" for g in GROUP_ORDER}
    
    medians = {g: np.median(data_dict[g]) for g in valid_groups}
    sorted_groups = sorted(valid_groups, key=lambda x: medians[x], reverse=True)
    
    letters = {g: "" for g in GROUP_ORDER}
    current_letter = 'a'
    letters[sorted_groups[0]] = current_letter
    
    for i in range(1, len(sorted_groups)):
        _, p_val = mannwhitneyu(data_dict[sorted_groups[i]], data_dict[sorted_groups[i-1]], alternative='two-sided')
        if p_val < 0.05:
            current_letter = chr(ord(current_letter) + 1)
        letters[sorted_groups[i]] = current_letter
    return letters

def run_dynamic_analysis():
    print(f"--- 正在读取多组学合并数据: {INPUT_OMICS} ---")
    if not os.path.exists(INPUT_OMICS): 
        print(f"❌ 找不到 {INPUT_OMICS}，请检查路径。")
        return

    df_omics = pd.read_csv(INPUT_OMICS, sep='\t')
    
    df_omics['insulation_score'] = pd.to_numeric(df_omics['insulation_score'], errors='coerce')
    
    for o in OMICS_TYPES:
        col = f"{o}_signal"
        if col in df_omics.columns:
            df_omics[col] = pd.to_numeric(df_omics[col], errors='coerce')
            if o == 'RNA': df_omics[col] = np.expm1(df_omics[col].fillna(0)) 

    print("🔄 正在基于原始 boundary_strength 重新计算边界动态变化...")
    
    delta_data = {}
    for cultivar, diff_file in DIFF_FILES.items():
        if not os.path.exists(diff_file): continue
            
        df_diff = pd.read_csv(diff_file, sep='\t')
        df_diff.rename(columns={'type': 'Dynamic_Type'}, inplace=True)
        
        col_ck = f'boundary_strength_{cultivar}_CK'
        col_et = f'boundary_strength_{cultivar}_ET'
        
        if col_ck in df_diff.columns and col_et in df_diff.columns:
            df_diff[col_ck] = pd.to_numeric(df_diff[col_ck], errors='coerce')
            df_diff[col_et] = pd.to_numeric(df_diff[col_et], errors='coerce')
            
            min_ck = df_diff[df_diff[col_ck] > 0][col_ck].min()
            min_et = df_diff[df_diff[col_et] > 0][col_et].min()
            pseudo_val = min(min_ck, min_et) / 2.0 if pd.notna(min_ck) and pd.notna(min_et) else 0.05
            
            ck_vals = df_diff[col_ck].fillna(pseudo_val)
            et_vals = df_diff[col_et].fillna(pseudo_val)
            
            df_diff['Boundary_logFC'] = np.log2(et_vals / ck_vals)
            df_diff['Boundary_AbsDiff'] = et_vals - ck_vals
            df_diff.replace([np.inf, -np.inf], np.nan, inplace=True)
        else:
            continue
        
        ck = df_omics[df_omics['sample'] == f"{cultivar}_CK"].set_index(['chrom', 'start', 'end'])
        et = df_omics[df_omics['sample'] == f"{cultivar}_ET"].set_index(['chrom', 'start', 'end'])
        
        deltas = pd.DataFrame(index=ck.index)
        
        for o in OMICS_TYPES:
            col = f"{o}_signal"
            if col in ck.columns and col in et.columns:
                deltas[f'Delta_abs_{o}'] = et[col] - ck[col]
                deltas[f'Delta_log2fc_{o}'] = np.log2(et[col] + 1) - np.log2(ck[col] + 1)
                
        merged = df_diff.join(deltas, on=['chrom', 'start', 'end'], how='inner')
        delta_data[cultivar] = merged

    if not delta_data: return

    # =========================================================
    # 🎨 严格对齐的双引擎绘图循环 (AbsDiff & log2FC)
    # =========================================================
    modes = [
        {
            'id': 'abs', 
            'suffix': 'AbsDiff', 
            'omics_y_label': 'Change (ET - CK)', 
            'boundary_col': 'Boundary_AbsDiff',
            'boundary_name': '$\Delta$ Boundary Strength',
            'bar_x_label': 'Spearman Rho ($\Delta$ Boundary Strength vs $\Delta$ ABS Signal)'
        },
        {
            'id': 'log2fc', 
            'suffix': 'log2FC', 
            'omics_y_label': 'log2FC Signal', 
            'boundary_col': 'Boundary_logFC',
            'boundary_name': 'log2FC Boundary Strength',
            'bar_x_label': 'Spearman Rho (log2FC Boundary Strength vs log2FC Signal)'
        }
    ]

    for mode in modes:
        print(f"\n🚀 正在生成 [{mode['suffix']}] 严格对齐图表...")
        
        # 1. 绘制全局动态相关性条形图
        bar_results = []
        for cultivar in delta_data.keys():
            df_c = delta_data[cultivar].dropna(subset=[mode['boundary_col']])
            for o in OMICS_TYPES:
                col = f"Delta_{mode['id']}_{o}"
                if col in df_c.columns:
                    valid_df = df_c.dropna(subset=[col])
                    if len(valid_df) > 10:
                        rho, pval = spearmanr(valid_df[mode['boundary_col']], valid_df[col])
                        bar_results.append({'Cultivar': cultivar, 'Omics': o, 'Rho': rho, 'Pval': pval})
        
        df_bar = pd.DataFrame(bar_results)
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(OMICS_TYPES))
        height = 0.35
        
        tm1_data = df_bar[df_bar['Cultivar'] == 'TM1'].set_index('Omics').reindex(OMICS_TYPES) if 'TM1' in delta_data else pd.DataFrame()
        zm_data = df_bar[df_bar['Cultivar'] == 'ZM113'].set_index('Omics').reindex(OMICS_TYPES) if 'ZM113' in delta_data else pd.DataFrame()
        
        if not tm1_data.empty: ax.barh(y_pos + height/2, tm1_data['Rho'], height, label='TM1', color=COLORS_BAR['TM1'])
        if not zm_data.empty: ax.barh(y_pos - height/2, zm_data['Rho'], height, label='ZM113', color=COLORS_BAR['ZM113'])
        
        for i, o in enumerate(OMICS_TYPES):
            if not tm1_data.empty and not pd.isna(tm1_data.loc[o, 'Rho']):
                rho, pval = tm1_data.loc[o, 'Rho'], tm1_data.loc[o, 'Pval']
                sig = get_sig_stars(pval)
                text = f"P={pval:.1e} {sig}" if pval < 0.05 else f"P={pval:.3f} ns"
                x_offset, ha = (0.02, 'left') if rho > 0 else (-0.02, 'right')
                ax.text(rho + x_offset, i + height/2, text, va='center', ha=ha, fontsize=9, fontweight='bold')
                
            if not zm_data.empty and not pd.isna(zm_data.loc[o, 'Rho']):
                rho_z, pval_z = zm_data.loc[o, 'Rho'], zm_data.loc[o, 'Pval']
                sig_z = get_sig_stars(pval_z)
                text_z = f"P={pval_z:.1e} {sig_z}" if pval_z < 0.05 else f"P={pval_z:.3f} ns"
                x_offset_z, ha_z = (0.02, 'left') if rho_z > 0 else (-0.02, 'right')
                ax.text(rho_z + x_offset_z, i - height/2, text_z, va='center', ha=ha_z, fontsize=9, fontweight='bold')

        ax.axvline(0, color='black', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(OMICS_TYPES)
        ax.set_xlabel(mode['bar_x_label'], fontsize=12)
        ax.set_title(f"Drivers of TAD Boundary Switching [{mode['suffix']}]", fontsize=14, fontweight='bold')
        ax.legend(title='Comparison', loc='lower right')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        if not df_bar.empty:
            x_max = df_bar['Rho'].abs().max() + 0.15
            ax.set_xlim(-x_max, x_max)
            
        plt.tight_layout()
        plt.savefig(f"step7_Global_Delta_Correlation_Barplot_{mode['suffix']}.pdf")
        plt.close()

        # 2. 绘制所有品种的边界动态变化箱线图
        for cultivar, df_cultivar in delta_data.items():
            print(f"  🎨 正在绘制 {cultivar} 边界动态变化箱线图 ({mode['suffix']})...")
            
            plot_features = [mode['boundary_col']] + [f"Delta_{mode['id']}_{o}" for o in OMICS_TYPES]
            if mode['id'] == 'abs':
                feature_names = [mode['boundary_name']] + [f"$\Delta$ {o}" for o in OMICS_TYPES]
            else:
                feature_names = [mode['boundary_name']] + [f"log2FC {o}" for o in OMICS_TYPES]
            
            fig, axes = plt.subplots(2, 4, figsize=(22, 12))
            axes = axes.flatten()
            
            for i, feat in enumerate(plot_features):
                ax = axes[i]
                group_data = [df_cultivar[df_cultivar['Dynamic_Type'] == g][feat].dropna().values for g in GROUP_ORDER]
                group_data = [d for d in group_data if len(d) > 0]
                
                p_kw = np.nan
                if len(group_data) > 1: stat, p_kw = kruskal(*group_data)
                    
                sns.boxplot(data=df_cultivar, x='Dynamic_Type', y=feat, order=GROUP_ORDER, 
                            palette=COLORS_BOX, ax=ax, showfliers=False)
                
                ax.axhline(0, color='grey', linestyle='--', lw=1.5, alpha=0.8)
                title = f"{feature_names[i]}\n(Kruskal-Wallis P={p_kw:.1e})" if not np.isnan(p_kw) else feature_names[i]
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xlabel("")
                
                current_ylabel = mode['boundary_name'] if feat == mode['boundary_col'] else mode['omics_y_label']
                ax.set_ylabel(current_ylabel)
                ax.tick_params(axis='x', rotation=15)
                
                # 💡 核心修复：将所有的显著性字母强制统一拉到同一水平高度 (顶部)！
                if not np.isnan(p_kw) and p_kw < 0.05:
                    data_dict = {g: df_cultivar[df_cultivar['Dynamic_Type'] == g][feat].dropna().values for g in GROUP_ORDER}
                    letters = get_letters(data_dict)
                    
                    y_min, y_max = ax.get_ylim()
                    y_range = y_max - y_min
                    
                    # 设定一个全局统一的字母高度：图表当前最高点上方 2% 的位置
                    unified_letter_y = y_max + y_range * 0.02
                    
                    # 拔高图表顶部空间，给字母留出空隙，防止与 title 重叠
                    ax.set_ylim(y_min, y_max + y_range * 0.15)
                    
                    for j, g in enumerate(GROUP_ORDER):
                        if letters[g]:
                            # 无论箱子在哪，字母都在同一高度 unified_letter_y
                            ax.text(j, unified_letter_y, letters[g], 
                                    ha='center', va='bottom', fontweight='bold', color='black')

            for k in range(len(plot_features), len(axes)): fig.delaxes(axes[k])
            plt.tight_layout()
            plt.savefig(f"step7_{cultivar}_Delta_Signal_Boxplots_{mode['suffix']}.pdf")
            plt.close()

    print("\n✨ 所有图表已生成完毕，方差分析字母已完美拉平对齐！")

if __name__ == "__main__":
    run_dynamic_analysis()
