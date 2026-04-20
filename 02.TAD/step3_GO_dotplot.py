import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

# --- 1. 环境设置 (Adobe AI 兼容) ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'

# --- 2. 关键词映射 ---
indicator_keywords = {
    'D/E (Conductivity/MDA)': ['lipid', 'membrane', 'cell wall', 'fatty acid', 'senescence'],
    'F (Proline)': ['proline', 'amide', 'arginine'],
    'G (Soluble Sugar)': ['sugar', 'glycogen', 'energy reserve', 'glucose', 'starch', 'oligosaccharide', 'carbohydrate'],
    'H (T-AOC)': ['ros', 'reactive oxygen', 'antioxidant', 'flavonoid', 'anthocyanin', 'peroxide', 'light intensity', 'light stimulus', 'photo']
}

def load_and_filter(filename, label):
    try:
        df = pd.read_csv(filename, sep='\t')
        res = []
        for ind, keys in indicator_keywords.items():
            pattern = '|'.join(keys)
            sub = df[df['Description'].str.contains(pattern, case=False, na=False)].copy()
            if not sub.empty:
                sub['Indicator'] = ind
                sub['Module_Type'] = label
                res.append(sub)
        return pd.concat(res) if res else pd.DataFrame()
    except: return pd.DataFrame()

# --- 3. 数据处理与逻辑分类 ---
df_neg = load_and_filter('负相关-all.txt', 'Negative')
df_pos = load_and_filter('正相关-all.txt', 'Positive')

# 提取关键信息用于判定逻辑分类
all_descriptions = set(df_neg['Description']).union(set(df_pos['Description']))
logic_category = {}

for desc in all_descriptions:
    # 检查显著性 (此处以 0.05 为逻辑分类门槛，以 0.01 为最终入选门槛)
    is_neg_sig = not df_neg[(df_neg['Description'] == desc) & (df_neg['p.adjust'] < 0.05)].empty
    is_pos_sig = not df_pos[(df_pos['Description'] == desc) & (df_pos['p.adjust'] < 0.05)].empty
    
    if is_neg_sig and is_pos_sig:
        logic_category[desc] = (1, "Both Significant")
    elif is_pos_sig:
        logic_category[desc] = (0, "Positive Only")
    else:
        logic_category[desc] = (2, "Negative Only")

# 合并并应用分类
all_data = pd.concat([df_neg, df_pos], ignore_index=True)
all_data['Logic_Rank'] = all_data['Description'].map(lambda x: logic_category.get(x, (3, "Other"))[0])
all_data['Logic_Label'] = all_data['Description'].map(lambda x: logic_category.get(x, (3, "Other"))[1])

# 过滤： padjust 至少有一个 < 0.01
valid_descs = all_data[all_data['p.adjust'] < 0.01]['Description'].unique()
plot_df = all_data[all_data['Description'].isin(valid_descs)].copy()

if not plot_df.empty:
    plot_df['-log10(p.adj)'] = -np.log10(plot_df['p.adjust'].astype(float))
    
    # 综合排序：先按 Logic_Rank (正->双->负)，再按 Indicator (D->H)，最后按显著性
    indicator_order = ['D/E (Conductivity/MDA)', 'F (Proline)', 'G (Soluble Sugar)', 'H (T-AOC)']
    plot_df['Indicator_Cat'] = pd.Categorical(plot_df['Indicator'], categories=indicator_order, ordered=True)
    
    # 执行排序
    plot_df = plot_df.sort_values(['Logic_Rank', 'Indicator_Cat', '-log10(p.adj)'], ascending=[True, True, True])
    
    # 生成 Y 轴标签
    plot_df['Y_Label'] = plot_df.apply(lambda x: f"[{x['Logic_Label']}] {x['Description']}", axis=1)
    y_labels_order = plot_df['Y_Label'].unique()
    plot_df['Y_Label'] = pd.Categorical(plot_df['Y_Label'], categories=y_labels_order, ordered=True)

    # --- 4. 绘图 ---
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(16, 15))
    
    norm = mcolors.Normalize(vmin=plot_df['-log10(p.adj)'].min(), vmax=plot_df['-log10(p.adj)'].max())
    x_map = {ind: i for i, ind in enumerate(indicator_order)}
    offset = 0.18

    # 绘制
    for m_type, color, shift in [('Negative', 'Blues', -offset), ('Positive', 'Reds', offset)]:
        sub = plot_df[plot_df['Module_Type'] == m_type]
        if not sub.empty:
            ax.scatter([x_map[ind] + shift for ind in sub['Indicator']], sub['Y_Label'],
                       s=(sub['Count'] / plot_df['Count'].max()) * 900 + 150,
                       c=sub['-log10(p.adj)'], cmap=color, norm=norm,
                       edgecolors='black', linewidths=0.6, alpha=0.8)

    # 细节处理
    ax.set_xticks(range(len(indicator_order)))
    ax.set_xticklabels(indicator_order, fontsize=12, fontweight='bold')
    
    # 添加 Logic 分类辅助线
    unique_y = plot_df[['Y_Label', 'Logic_Rank']].drop_duplicates()
    for i in range(len(unique_y) - 1):
        if unique_y.iloc[i]['Logic_Rank'] != unique_y.iloc[i+1]['Logic_Rank']:
            ax.axhline(y=i+0.5, color='red', linestyle='-', alpha=0.5, linewidth=1.5) # 逻辑分界线用红色
        elif plot_df['Y_Label'].unique()[i][:5] != plot_df['Y_Label'].unique()[i+1][:5]:
            ax.axhline(y=i+0.5, color='gray', linestyle='--', alpha=0.3) # 指标分界线用灰色

    # Colorbars
    cax_pos = fig.add_axes([0.91, 0.58, 0.012, 0.2])
    fig.colorbar(ScalarMappable(norm=norm, cmap='Reds'), cax=cax_pos).set_label('Positive ($-log_{10}p$)')
    cax_neg = fig.add_axes([0.91, 0.25, 0.012, 0.2])
    fig.colorbar(ScalarMappable(norm=norm, cmap='Blues'), cax=cax_neg).set_label('Negative ($-log_{10}p$)')

    plt.subplots_adjust(right=0.85)
    plt.title('GO Enrichment Rearranged by Regulation Pattern (Positive -> Both -> Negative)', fontsize=18, pad=35)
    plt.savefig('Physiology_GO_Logical_Order.pdf', bbox_inches='tight', transparent=True)
    print("PDF generated: Physiology_GO_Logical_Order.pdf")
