# scripts/verify_data.py
"""验证所有文件是否存在"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm

manifests = [
    r"C:\T1+MIND_V2\data\ADNI\manifest_train.csv",
    r"C:\T1+MIND_V2\data\ADNI\manifest_val.csv",
    r"C:\T1+MIND_V2\data\PUSH\manifest_test.csv",
    r"C:\T1+MIND_V2\data\SMHC\manifest_test.csv",
]

for manifest_path in manifests:
    print(f"\n检查 {manifest_path}")
    df = pd.read_csv(manifest_path)
    
    missing_t1w = []
    missing_mind = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if not Path(row['t1w_path']).exists():
            missing_t1w.append(row['subject_id'])
        if not Path(row['mind_path']).exists():
            missing_mind.append(row['subject_id'])
    
    if missing_t1w:
        print(f"❌ 缺失T1w: {len(missing_t1w)} 个")
        print(f"   {missing_t1w[:5]}")
    else:
        print(f"✓ 所有T1w文件存在")
    
    if missing_mind:
        print(f"❌ 缺失MIND: {len(missing_mind)} 个")
        print(f"   {missing_mind[:5]}")
    else:
        print(f"✓ 所有MIND文件存在")
