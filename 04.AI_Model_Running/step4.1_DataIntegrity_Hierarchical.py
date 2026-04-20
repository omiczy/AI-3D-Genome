import os
import sys
import time

# --- 1. 核心配置 ---
DATA_BASE = "/home/dell/project/HIC_tmp2/02.model/"
GENOME_FA = os.path.join(DATA_BASE, "ZM113_T2T_V2.genome.fa")
GENOME_GFF = os.path.join(DATA_BASE, "ZM113_T2T_V2.genome.gff")

# 预期样本列表
SAMPLES = ['TM1_CK', 'TM1_ET', 'ZM113_CK', 'ZM113_ET']

# 金字塔文件列表
TIER_NAMES = ["1_Common_Core", "2_Conserved_High", "3_Responsive_Mid", "4_Individual_All"]
BALANCED_FILES = [f"step3_tier_{t}_balanced.csv" for t in TIER_NAMES]

BW_SUFFIX = ".merged.bw"
REPORT_FILE = "step4.1_DataIntegrity_Report.txt"

def log_status(msg, success=True, file_handle=None):
    symbol = "✅" if success else "❌"
    output = f"{symbol} {msg}"
    print(output)
    if file_handle: file_handle.write(output + "\n")

def main():
    with open(REPORT_FILE, 'w', encoding='utf-8') as report:
        header = f"🚀 Step 4.1: Hierarchical Data Integrity Check (No RNA) | {time.strftime('%H:%M:%S')}"
        print("\n" + "="*60 + "\n" + header + "\n" + "="*60)
        report.write(header + "\n" + "="*60 + "\n")

        # 1. 基础参考基因组文件检查
        print("[检查基础参考基因组文件]")
        for f in [GENOME_FA, GENOME_GFF]:
            log_status(f"文件存在: {os.path.basename(f)}", os.path.exists(f), report)

        # 2. BigWig 表观修饰文件检查
        print("\n[检查 BigWig 表观修饰文件]")
        if os.path.exists(DATA_BASE):
            bw_files = [f for f in os.listdir(DATA_BASE) if f.endswith(BW_SUFFIX)]
            log_status(f"BigWig 文件总数: {len(bw_files)} (预期应为 4个样本 x 6种修饰 = 24)", len(bw_files) >= 24, report)
        else:
            bw_files = []
            log_status(f"数据目录 {DATA_BASE} 不存在！", False, report)

        # 3. 检查 4 个金字塔层级的平衡数据集文件
        print("\n[检查金字塔层级平衡数据集]")
        for bf in BALANCED_FILES:
            exists = os.path.exists(bf)
            msg = f"层级文件: {bf}"
            if exists: 
                msg += f" (Size: {os.path.getsize(bf)//1024} KB)"
            log_status(msg, exists, report)

        # 4. GFF 格式与样本名称校验
        try:
            print("\n[格式与样本名解析校验]")
            
            # 简单校验 GFF 是否能正常读取 gene ID
            valid_gff = False
            if os.path.exists(GENOME_GFF):
                with open(GENOME_GFF, 'r') as f:
                    for line in f:
                        if '\tgene\t' in line and 'ID=' in line:
                            valid_gff = True
                            break
            log_status(f"GFF 注释文件结构解析正常", valid_gff, report)

            # 校验预期的 4 个样本是否有对应的 BW 文件
            bw_samples = set([f.split('.')[0] for f in bw_files])
            missing_samples = set(SAMPLES) - bw_samples
            
            if not missing_samples:
                log_status(f"BigWig 样本完整性校验: 检测到所有预期样本 ({', '.join(SAMPLES)})", True, report)
            else:
                log_status(f"BigWig 样本缺失: 找不到以下样本的 .bw 文件: {', '.join(missing_samples)}", False, report)

        except Exception as e:
            log_status(f"校验过程中发生异常: {str(e)}", False, report)

        print("\n" + "="*60 + f"\n📢 完整检查报告已保存至: {REPORT_FILE}\n" + "="*60)

if __name__ == "__main__":
    main()
