# scripts/full_training_pipeline.py
"""
完整训练流程：从数据拆分到模型评估
"""

import subprocess
import sys
from pathlib import Path
import yaml

def run_command(cmd, description):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Error running: {cmd}")
        sys.exit(1)
    print(f"✓ {description} completed!")

def main():
    """完整训练流程"""
    
    # ===== Step 1: 拆分数据 =====
    run_command(
        "python scripts/split_manifest.py",
        "Step 1: 拆分manifest文件"
    )
    
    # ===== Step 2: 预计算组间先验 =====
    run_command(
        "python preprocess_group_prior.py",
        "Step 2: 计算组间先验"
    )
    
    # ===== Step 3: 主实验训练 =====
    run_command(
        "python train.py --config config_4090.yaml",
        "Step 3: 主实验训练"
    )
    
    # ===== Step 4: 评估主实验 =====
    run_command(
        "python evaluate.py --checkpoint ./checkpoints/TPFN_RTX4090_v1/best_model.pth",
        "Step 4: 评估主实验"
    )
    
    # ===== Step 5: 消融实验 =====
    ablation_configs = [
        ("config_ablation_no_text.yaml", "无文本先验"),
        ("config_ablation_no_group.yaml", "无组间先验"),
        ("config_ablation_no_individual.yaml", "无个体先验"),
        ("config_ablation_no_multitask.yaml", "无多任务学习"),
        ("config_ablation_image_only.yaml", "仅影像模态"),
        ("config_ablation_graph_only.yaml", "仅图网络模态"),
    ]
    
    for config, name in ablation_configs:
        run_command(
            f"python train.py --config {config}",
            f"Step 5.{ablation_configs.index((config, name))+1}: 消融实验 - {name}"
        )
    
    # ===== Step 6: 生成对比表格 =====
    run_command(
        "python scripts/generate_comparison_table.py",
        "Step 6: 生成结果对比表格"
    )
    
    print("\n" + "="*60)
    print("🎉 完整训练流程完成！")
    print("="*60)
    print("\n结果文件：")
    print("  - 主实验：./results/main_experiment/")
    print("  - 消融实验：./results/ablation_studies/")
    print("  - 对比表格：./results/comparison_table.csv")

if __name__ == "__main__":
    main()
