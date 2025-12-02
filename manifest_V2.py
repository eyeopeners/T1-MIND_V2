# scripts/split_manifest_by_subject.py
"""
从统一的manifest.csv拆分出各数据集的文件
✓ ADNI按被试（subject）划分，避免数据泄露
✓ PUSH和SMHC全部作为测试集
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def extract_subject_id(full_id):
    """
    提取被试ID（去掉扫描日期）
    
    Examples:
        "002_S_0413_2017-06-21" -> "002_S_0413"
        "002_S_0413_2025-02-19" -> "002_S_0413"
        "BD_001" -> "BD_001" (PUSH数据保持不变)
        "31-001" -> "31-001" (SMHC数据保持不变)
    """
    parts = full_id.split('_')
    
    # ADNI数据格式: XXX_S_XXXX_YYYY-MM-DD
    if len(parts) >= 4 and parts[1] in ['S', 'M']:
        return '_'.join(parts[:3])  # 取前三部分 "002_S_0413"
    
    # 其他数据集保持原样
    return full_id

def split_by_subject(df_adni, val_ratio=0.2, random_state=42):
    """
    按被试ID进行分层拆分，确保同一被试的所有扫描在同一集合
    
    Args:
        df_adni: ADNI数据的DataFrame
        val_ratio: 验证集比例
        random_state: 随机种子
    
    Returns:
        train_df, val_df
    """
    np.random.seed(random_state)
    
    # 1. 按被试分组
    subject_groups = defaultdict(list)
    for idx, row in df_adni.iterrows():
        subject_id = extract_subject_id(row['subject_id'])
        subject_groups[subject_id].append(idx)
    
    print(f"\n总样本数: {len(df_adni)}")
    print(f"唯一被试数: {len(subject_groups)}")
    
    # 检查每个被试有多少次扫描
    scans_per_subject = [len(scans) for scans in subject_groups.values()]
    print(f"平均每个被试的扫描次数: {np.mean(scans_per_subject):.2f}")
    print(f"最多扫描次数: {max(scans_per_subject)}")
    print(f"单次扫描的被试数: {sum(1 for x in scans_per_subject if x == 1)}")
    print(f"多次扫描的被试数: {sum(1 for x in scans_per_subject if x > 1)}")
    
    # 2. 确定每个被试的主要标签（使用最常见的标签）
    subject_labels = {}
    for subject_id, indices in subject_groups.items():
        labels = df_adni.loc[indices, 'label'].values
        # 使用众数作为该被试的主要标签
        unique, counts = np.unique(labels, return_counts=True)
        main_label = unique[np.argmax(counts)]
        subject_labels[subject_id] = main_label
    
    # 3. 按标签分层
    subjects_by_label = defaultdict(list)
    for subject_id, label in subject_labels.items():
        subjects_by_label[label].append(subject_id)
    
    print(f"\n被试标签分布:")
    for label in sorted(subjects_by_label.keys()):
        print(f"  Label {label}: {len(subjects_by_label[label])} subjects")
    
    # 4. 对每个标签单独拆分
    train_subjects = []
    val_subjects = []
    
    for label, subjects in subjects_by_label.items():
        subjects = list(subjects)
        np.random.shuffle(subjects)
        
        n_val = max(1, int(len(subjects) * val_ratio))
        
        val_subjects.extend(subjects[:n_val])
        train_subjects.extend(subjects[n_val:])
    
    print(f"\n拆分后的被试数:")
    print(f"  训练集: {len(train_subjects)} subjects")
    print(f"  验证集: {len(val_subjects)} subjects")
    
    # 5. 收集样本
    train_indices = []
    val_indices = []
    
    for subject_id, indices in subject_groups.items():
        if subject_id in train_subjects:
            train_indices.extend(indices)
        elif subject_id in val_subjects:
            val_indices.extend(indices)
    
    train_df = df_adni.loc[train_indices].copy()
    val_df = df_adni.loc[val_indices].copy()
    
    # 6. 验证没有泄露
    train_subj_set = set([extract_subject_id(sid) for sid in train_df['subject_id']])
    val_subj_set = set([extract_subject_id(sid) for sid in val_df['subject_id']])
    
    overlap = train_subj_set & val_subj_set
    assert len(overlap) == 0, f"发现{len(overlap)}个重叠被试！"
    
    print(f"\n✓ 验证通过：训练集和验证集无被试重叠")
    
    return train_df, val_df

def split_unified_manifest(
    unified_manifest_path,
    output_dir="./data",
    val_ratio=0.2,
    random_state=42
):
    """
    拆分统一的manifest文件
    
    Args:
        unified_manifest_path: 统一的manifest.csv路径
        output_dir: 输出目录
        val_ratio: 验证集比例（针对ADNI被试数）
        random_state: 随机种子
    """
    # 加载数据
    df = pd.read_csv(unified_manifest_path)
    
    print("="*60)
    print("统一manifest信息：")
    print(f"总样本数: {len(df)}")
    print("="*60)
    
    # 1. 按数据集拆分
    # ADNI: 不以 'BD' 或 '31-' 开头
    df_adni = df[~df['subject_id'].str.startswith('BD') & 
                 ~df['subject_id'].str.startswith('31-')].copy()
    
    # PUSH: 以 'BD' 开头
    df_push = df[df['subject_id'].str.startswith('BD')].copy()
    
    # SMHC: 以 '31-' 开头
    df_smhc = df[df['subject_id'].str.startswith('31-')].copy()
    
    print(f"\n数据集划分:")
    print(f"ADNI: {len(df_adni)} samples")
    print(f"  NC: {(df_adni['label']==0).sum()}")
    print(f"  MCI: {(df_adni['label']==1).sum()}")
    print(f"  AD: {(df_adni['label']==2).sum()}")
    
    print(f"\nPUSH: {len(df_push)} samples (全部用于测试)")
    print(f"  NC: {(df_push['label']==0).sum()}")
    print(f"  MCI: {(df_push['label']==1).sum()}")
    print(f"  AD: {(df_push['label']==2).sum()}")
    
    print(f"\nSMHC: {len(df_smhc)} samples (全部用于测试)")
    print(f"  NC: {(df_smhc['label']==0).sum()}")
    print(f"  MCI: {(df_smhc['label']==1).sum()}")
    print(f"  AD: {(df_smhc['label']==2).sum()}")
    
    # 2. 创建输出目录
    output_dir = Path(output_dir)
    (output_dir / "ADNI").mkdir(parents=True, exist_ok=True)
    (output_dir / "PUSH").mkdir(parents=True, exist_ok=True)
    (output_dir / "SMHC").mkdir(parents=True, exist_ok=True)
    
    # 3. 处理ADNI数据 - 按被试拆分
    print("\n" + "="*60)
    print("ADNI数据集 - 按被试拆分（避免数据泄露）")
    print("="*60)
    
    train_df, val_df = split_by_subject(
        df_adni,
        val_ratio=val_ratio,
        random_state=random_state
    )
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    
    # 保存
    train_df.to_csv(output_dir / "ADNI" / "manifest_train.csv", index=False)
    val_df.to_csv(output_dir / "ADNI" / "manifest_val.csv", index=False)
    
    print(f"\n最终样本数:")
    print(f"训练集: {len(train_df)} samples")
    print(f"  NC: {(train_df['label']==0).sum()}")
    print(f"  MCI: {(train_df['label']==1).sum()}")
    print(f"  AD: {(train_df['label']==2).sum()}")
    
    print(f"\n验证集: {len(val_df)} samples")
    print(f"  NC: {(val_df['label']==0).sum()}")
    print(f"  MCI: {(val_df['label']==1).sum()}")
    print(f"  AD: {(val_df['label']==2).sum()}")
    
    # 4. 处理外部验证数据集（全部作为测试集）
    df_push['split'] = 'test'
    df_push.to_csv(output_dir / "PUSH" / "manifest_test.csv", index=False)
    
    df_smhc['split'] = 'test'
    df_smhc.to_csv(output_dir / "SMHC" / "manifest_test.csv", index=False)
    
    # 5. 生成统计报告
    print("\n" + "="*60)
    print("拆分完成！")
    print("="*60)
    print(f"\n文件保存在 {output_dir}")
    print("├── ADNI/")
    print("│   ├── manifest_train.csv")
    print("│   └── manifest_val.csv")
    print("├── PUSH/")
    print("│   └── manifest_test.csv")
    print("└── SMHC/")
    print("    └── manifest_test.csv")
    
    # 6. 额外验证：检查数据泄露
    print("\n" + "="*60)
    print("数据泄露检查")
    print("="*60)
    
    train_subjects = set([extract_subject_id(sid) for sid in train_df['subject_id']])
    val_subjects = set([extract_subject_id(sid) for sid in val_df['subject_id']])
    
    overlap = train_subjects & val_subjects
    
    if overlap:
        print(f"⚠️ 警告：发现 {len(overlap)} 个重叠被试！")
        print(f"示例: {list(overlap)[:5]}")
    else:
        print(f"✓ 确认：训练集和验证集无被试重叠")
        print(f"  训练集被试数: {len(train_subjects)}")
        print(f"  验证集被试数: {len(val_subjects)}")
    
    return train_df, val_df, df_push, df_smhc


if __name__ == "__main__":
    # 使用新的按被试拆分方法
    train_df, val_df, push_df, smhc_df = split_unified_manifest(
        unified_manifest_path=r"C:\T1+MIND_V2\manifest.csv",
        output_dir=r"C:\T1+MIND_V2\data",
        val_ratio=0.2,  # 20%的被试用于验证
        random_state=42
    )
    
    print("\n" + "="*60)
    print("全部完成！可以开始训练了")
    print("="*60)
