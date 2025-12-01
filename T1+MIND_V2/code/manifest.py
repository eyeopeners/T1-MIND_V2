# scripts/split_manifest.py
"""
从统一的manifest.csv拆分出各数据集的文件
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split

def split_unified_manifest(
    unified_manifest_path,
    output_dir="./data",
    val_ratio=0.2,
    n_folds=5,
    random_state=42
):
    """
    拆分统一的manifest文件
    
    Args:
        unified_manifest_path: 统一的manifest.csv路径
        output_dir: 输出目录
        val_ratio: 验证集比例（仅在不做交叉验证时使用）
        n_folds: 交叉验证折数（设为None则不做交叉验证）
        random_state: 随机种子
    """
    # 加载数据
    df = pd.read_csv(unified_manifest_path)
    
    print("="*60)
    print("统一manifest信息：")
    print(f"总样本数: {len(df)}")
    print("="*60)
    
    # 1. 按subject_id前缀拆分数据集
    df_adni = df[~df['subject_id'].str.startswith('BD') & 
                 ~df['subject_id'].str.startswith('31-')].copy()
    df_push = df[df['subject_id'].str.startswith('BD')].copy()
    df_smhc = df[df['subject_id'].str.startswith('31-')].copy()
    
    print(f"\nADNI: {len(df_adni)} samples")
    print(f"  NC: {(df_adni['label']==0).sum()}")
    print(f"  MCI: {(df_adni['label']==1).sum()}")
    print(f"  AD: {(df_adni['label']==2).sum()}")
    
    print(f"\nPUSH: {len(df_push)} samples")
    print(f"  NC: {(df_push['label']==0).sum()}")
    print(f"  MCI: {(df_push['label']==1).sum()}")
    print(f"  AD: {(df_push['label']==2).sum()}")
    
    print(f"\nSMHC: {len(df_smhc)} samples")
    print(f"  NC: {(df_smhc['label']==0).sum()}")
    print(f"  MCI: {(df_smhc['label']==1).sum()}")
    print(f"  AD: {(df_smhc['label']==2).sum()}")
    
    # 2. 创建输出目录
    output_dir = Path(output_dir)
    (output_dir / "ADNI").mkdir(parents=True, exist_ok=True)
    (output_dir / "PUSH").mkdir(parents=True, exist_ok=True)
    (output_dir / "SMHC").mkdir(parents=True, exist_ok=True)
    
    # 3. 处理ADNI数据（训练+验证）
    if n_folds is None:
        # 不做交叉验证，简单拆分
        print("\n" + "="*60)
        print("使用简单训练/验证拆分")
        print("="*60)
        
        train_df, val_df = train_test_split(
            df_adni,
            test_size=val_ratio,
            stratify=df_adni['label'],
            random_state=random_state
        )
        
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        
        # 保存
        train_df.to_csv(output_dir / "ADNI" / "manifest_train.csv", index=False)
        val_df.to_csv(output_dir / "ADNI" / "manifest_val.csv", index=False)
        
        print(f"\nTrain: {len(train_df)} samples")
        print(f"  NC: {(train_df['label']==0).sum()}")
        print(f"  MCI: {(train_df['label']==1).sum()}")
        print(f"  AD: {(train_df['label']==2).sum()}")
        
        print(f"\nVal: {len(val_df)} samples")
        print(f"  NC: {(val_df['label']==0).sum()}")
        print(f"  MCI: {(val_df['label']==1).sum()}")
        print(f"  AD: {(val_df['label']==2).sum()}")
        
    else:
        # 做K折交叉验证
        print("\n" + "="*60)
        print(f"使用{n_folds}折交叉验证")
        print("="*60)
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_adni, df_adni['label'])):
            train_df = df_adni.iloc[train_idx].copy()
            val_df = df_adni.iloc[val_idx].copy()
            
            train_df['split'] = 'train'
            train_df['fold'] = fold_idx
            val_df['split'] = 'val'
            val_df['fold'] = fold_idx
            
            # 保存每一折
            train_df.to_csv(
                output_dir / "ADNI" / f"manifest_train_fold{fold_idx}.csv", 
                index=False
            )
            val_df.to_csv(
                output_dir / "ADNI" / f"manifest_val_fold{fold_idx}.csv", 
                index=False
            )
            
            print(f"\nFold {fold_idx}:")
            print(f"  Train: {len(train_df)} samples")
            print(f"    NC: {(train_df['label']==0).sum()}, "
                  f"MCI: {(train_df['label']==1).sum()}, "
                  f"AD: {(train_df['label']==2).sum()}")
            print(f"  Val: {len(val_df)} samples")
            print(f"    NC: {(val_df['label']==0).sum()}, "
                  f"MCI: {(val_df['label']==1).sum()}, "
                  f"AD: {(val_df['label']==2).sum()}")
        
        # 同时保存默认的fold0作为标准训练/验证集
        train_idx, val_idx = list(skf.split(df_adni, df_adni['label']))[0]
        train_df = df_adni.iloc[train_idx].copy()
        val_df = df_adni.iloc[val_idx].copy()
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        train_df.to_csv(output_dir / "ADNI" / "manifest_train.csv", index=False)
        val_df.to_csv(output_dir / "ADNI" / "manifest_val.csv", index=False)
    
    # 4. 处理外部验证数据集
    # PUSH - 全部作为测试集
    df_push['split'] = 'test'
    df_push.to_csv(output_dir / "PUSH" / "manifest_test.csv", index=False)
    
    # SMHC - 全部作为测试集
    df_smhc['split'] = 'test'
    df_smhc.to_csv(output_dir / "SMHC" / "manifest_test.csv", index=False)
    
    print("\n" + "="*60)
    print("拆分完成！")
    print("="*60)
    print(f"\n文件保存在 {output_dir}")
    print("├── ADNI/")
    if n_folds is None:
        print("│   ├── manifest_train.csv")
        print("│   └── manifest_val.csv")
    else:
        print(f"│   ├── manifest_train.csv (默认，fold 0)")
        print(f"│   ├── manifest_val.csv (默认，fold 0)")
        for i in range(n_folds):
            print(f"│   ├── manifest_train_fold{i}.csv")
            print(f"│   └── manifest_val_fold{i}.csv")
    print("├── PUSH/")
    print("│   └── manifest_test.csv")
    print("└── SMHC/")
    print("    └── manifest_test.csv")
    
    return df_adni, df_push, df_smhc


if __name__ == "__main__":
    # 使用示例
    
    # 选项1：不做交叉验证（推荐，因为有外部验证集）
    split_unified_manifest(
        unified_manifest_path=r"C:\T1+MIND_V2\manifest.csv",
        output_dir=r"C:\T1+MIND_V2\data",
        val_ratio=0.2,
        n_folds=None,  # 不做交叉验证
        random_state=42
    )
    
    # 选项2：做5折交叉验证（如果想更稳健）
    # split_unified_manifest(
    #     unified_manifest_path="./all_data_manifest.csv",
    #     output_dir="./data",
    #     n_folds=5,
    #     random_state=42
    # )
