import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time

SAMPLES = ['TM1_CK', 'TM1_ET', 'ZM113_CK', 'ZM113_ET']
TIER_NAMES = ["1_Common_Core", "2_Conserved_High", "3_Responsive_Mid", "4_Individual_All"]

# ⚠️ 修改点：已移除 'A1_TPM', 'A2_TPM'，特征总数为 14
FEATURE_COLS = [
    'A1_ATAC', 'A1_CG', 'A1_CHG', 'A1_CHH', 'A1_H3K4me3', 'A1_H3K27me3',
    'A2_ATAC', 'A2_CG', 'A2_CHG', 'A2_CHH', 'A2_H3K4me3', 'A2_H3K27me3',
    'gc', 'log_dist'
]

TRAIN_CHRS = [f'A{i:02d}' for i in range(1, 11)] + [f'D{i:02d}' for i in range(1, 11)]
VAL_CHRS = ['A11', 'D11']
TEST_CHRS = ['A12', 'A13', 'D12', 'D13']

def process_tier_manager(tier_name):
    print(f"\n📦 [Tier: {tier_name}] 数据整合与拆分 (No RNA Features)...")
    
    files = [f"step4.2_{tier_name}_{s}_features.csv" for s in SAMPLES]
    dfs = [pd.read_csv(f) for f in files if os.path.exists(f)]
    if not dfs: return
    
    all_data = pd.concat(dfs, ignore_index=True)
    print(f"   - 合并后总数据: {len(all_data)}")

    train_df = all_data[all_data['chrom1'].isin(TRAIN_CHRS)].copy()
    val_df = all_data[all_data['chrom1'].isin(VAL_CHRS)].copy()
    test_df = all_data[all_data['chrom1'].isin(TEST_CHRS)].copy()

    scaler = StandardScaler()
    train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
    val_df[FEATURE_COLS] = scaler.transform(val_df[FEATURE_COLS])
    test_df[FEATURE_COLS] = scaler.transform(test_df[FEATURE_COLS])

    if not os.path.exists('model_assets'): os.makedirs('model_assets')
    joblib.dump(scaler, f'model_assets/step4.3_{tier_name}_scaler.pkl')
    
    train_df.to_csv(f'step4.3_{tier_name}_train.csv', index=False)
    val_df.to_csv(f'step4.3_{tier_name}_val.csv', index=False)
    test_df.to_csv(f'step4.3_{tier_name}_test.csv', index=False)
    
    print(f"✅ {tier_name} 就绪: Train({len(train_df)}), Test({len(test_df)})")

if __name__ == "__main__":
    start = time.time()
    print("🚀 Step 4.3: Hierarchical Data Manager (Epigenome Only)")
    for t in TIER_NAMES: process_tier_manager(t)
    print(f"\n✨ 全部完成！耗时: {time.time()-start:.2f}s")
