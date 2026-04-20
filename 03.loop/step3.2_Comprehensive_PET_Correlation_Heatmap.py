import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os
import matplotlib

# 设置后端以防止远程服务器报错
matplotlib.use('Agg')

def get_sig_star(p_val):
    """根据 P 值返回显著性星号"""
    if p_val < 0.001: return "***"
    if p_val < 0.01: return "**"
    if p_val < 0.05: return "*"
    return "ns"

def run_stylish_correlation_analysis():
    input_file = "step1.Comprehensive_Anchor_Master_Table.tsv"
    if not os.path.exists(input_file):
        print(f"Error: 找不到输入文件 '{input_file}'")
        return

    print("正在加载数据并计算全组学相关性矩阵...")
    df = pd.read_csv(input_file, sep='\t')

    # 样本配置
    samples = {
        'TM1_CK': {'s_pre': 'T_CK', 'full': 'TM1_CK'},
        'TM1_ET': {'s_pre': 'T_ET', 'full': 'TM1_ET'},
        'ZM113_CK': {'s_pre': 'Z_CK', 'full': 'ZM113_CK'},
        'ZM113_ET': {'s_pre': 'Z_ET', 'full': 'ZM113_ET'}
    }

    # 维度配置
    mods = {
        'ATAC': 'ATAC_mean',
        'H3K4me3': 'H3K4me3_mean',
        'H3K27me3': 'H3K27me3_mean',
        'RNA': 'RNA_mean',
        'CG': 'CG_site_mean',
        'CHG': 'CHG_site_mean',
        'CHH': 'CHH_site_mean'
    }

    rho_matrix = []
    star_matrix = []

    for s_name, cfg in samples.items():
        row_rho = {'Sample': s_name}
        row_star = {'Sample': s_name}
        
        pet_col = f"{cfg['s_pre']}_P"
        state_col = f"{cfg['s_pre']}_S"
        
        # 仅分析在该样本中活跃的锚点 (YES) 且 PET > 0
        df_sub = df[(df[state_col] == 'YES') & (df[pet_col] > 0)].copy()
        
        for mod_label, suffix in mods.items():
            val_col = f"{cfg['full']}_{suffix}"
            
            if val_col in df_sub.columns:
                # 剔除空值
                data = df_sub[[pet_col, val_col]].dropna()
                
                if len(data) > 50:
                    # 计算 Spearman 相关性
                    rho, p_val = spearmanr(data[pet_col], data[val_col])
                    row_rho[mod_label] = rho
                    row_star[mod_label] = f"{rho:.2f}\n{get_sig_star(p_val)}"
                else:
                    row_rho[mod_label] = np.nan
                    row_star[mod_label] = "-"
            else:
                row_rho[mod_label] = np.nan
                row_star[mod_label] = "N/A"
        
        rho_matrix.append(row_rho)
        star_matrix.append(row_star)

    # 转换为 DataFrame
    df_rho = pd.DataFrame(rho_matrix).set_index('Sample')
    df_annot = pd.DataFrame(star_matrix).set_index('Sample')
    
    # 保存原始数值矩阵
    df_rho.to_csv("step3.2.PET_Correlation_Rho_Matrix.csv")

    # --- 绘图阶段 ---
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制热图
    # cmap 'vlag' 是学术界常用的发散色带
    sns.heatmap(df_rho, 
                annot=df_annot, 
                fmt="", 
                cmap="vlag", 
                center=0, 
                vmin=-0.6, vmax=0.6,
                linewidths=1.5, 
                linecolor='white',
                cbar_kws={'label': "Spearman's Rho (ρ)", 'shrink': 0.8},
                annot_kws={"size": 10, "weight": "bold"})
    
    # 美化标签
    plt.title("Correlation: Loop Intensity vs Epigenetic Signals", fontsize=14, fontweight='bold', pad=25)
    plt.xlabel("Genomic Modalities", fontsize=12, labelpad=15)
    plt.ylabel("Samples", fontsize=12, labelpad=15)
    
    # 旋转轴标签
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    output_pdf = "step3.2.Comprehensive_PET_Correlation_Heatmap.pdf"
    plt.savefig(output_pdf, dpi=300)
    plt.close()
    
    print(f"\n[任务完成]")
    print(f"相关性数据矩阵: step3.2.PET_Correlation_Rho_Matrix.csv")
    print(f"好看的热图已生成: {output_pdf}")

if __name__ == "__main__":
    run_stylish_correlation_analysis()
