# scripts/generate_cover_matrix.py
"""
从HCP-MMP图谱生成覆盖矩阵
覆盖矩阵定义：cover_ctx[s, r] = 1 表示第s个切片包含第r个脑区
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

def generate_cover_matrix_from_atlas(
    atlas_path,
    output_path="./data/cover_matrix_ctx.npy",
    num_regions=360,
    axis=2  # 2表示轴向切片（z轴），可选0(矢状面)、1(冠状面)、2(轴向)
):
    """
    从图谱文件生成覆盖矩阵
    
    Args:
        atlas_path: HCP-MMP图谱路径（.nii 或 .nii.gz）
        output_path: 输出路径
        num_regions: 脑区数量（HCP-MMP为360）
        axis: 切片方向（0=矢状面, 1=冠状面, 2=轴向）
    
    Returns:
        cover_matrix: [S, num_regions] 二值矩阵
    """
    
    print("="*60)
    print("Generating Cover Matrix from Atlas")
    print("="*60)
    print(f"Atlas path: {atlas_path}")
    print(f"Output path: {output_path}")
    print(f"Number of regions: {num_regions}")
    print(f"Slice axis: {axis} ({'Sagittal' if axis==0 else 'Coronal' if axis==1 else 'Axial'})")
    
    # 1. 加载图谱
    print("\nLoading atlas...")
    atlas_nii = nib.load(atlas_path)
    atlas_data = atlas_nii.get_fdata().astype(np.int32)
    
    print(f"  Atlas shape: {atlas_data.shape}")
    print(f"  Unique labels: {len(np.unique(atlas_data))}")
    print(f"  Label range: [{atlas_data.min()}, {atlas_data.max()}]")
    
    # 2. 获取切片维度
    num_slices = atlas_data.shape[axis]
    print(f"\nNumber of slices: {num_slices}")
    
    # 3. 初始化覆盖矩阵
    cover_matrix = np.zeros((num_slices, num_regions), dtype=np.float32)
    
    # 4. 对每个切片，检查包含哪些脑区
    print("\nGenerating cover matrix...")
    for s in tqdm(range(num_slices), desc="Processing slices"):
        # 提取当前切片
        if axis == 0:
            slice_data = atlas_data[s, :, :]
        elif axis == 1:
            slice_data = atlas_data[:, s, :]
        else:  # axis == 2
            slice_data = atlas_data[:, :, s]
        
        # 获取该切片上的所有标签
        labels_in_slice = np.unique(slice_data)
        labels_in_slice = labels_in_slice[labels_in_slice > 0]  # 排除背景(0)
        
        # 对于HCP-MMP图谱，标签可能是1-360或其他编号方式
        # 需要映射到0-359的索引
        for label in labels_in_slice:
            if 1 <= label <= num_regions:
                cover_matrix[s, label - 1] = 1.0  # 标签从1开始，索引从0开始
    
    # 5. 统计信息
    print("\n" + "="*60)
    print("Cover Matrix Statistics")
    print("="*60)
    print(f"Shape: {cover_matrix.shape}")
    print(f"Total non-zero elements: {np.sum(cover_matrix):.0f}")
    print(f"Average regions per slice: {cover_matrix.sum(axis=1).mean():.2f}")
    print(f"Min regions per slice: {cover_matrix.sum(axis=1).min():.0f}")
    print(f"Max regions per slice: {cover_matrix.sum(axis=1).max():.0f}")
    
    # 检查是否有脑区没有被任何切片覆盖
    regions_covered = cover_matrix.sum(axis=0)
    uncovered_regions = np.where(regions_covered == 0)[0]
    if len(uncovered_regions) > 0:
        print(f"\n⚠️ Warning: {len(uncovered_regions)} regions not covered by any slice:")
        print(f"  Regions: {uncovered_regions[:10]}")  # 只显示前10个
    else:
        print(f"\n✓ All {num_regions} regions are covered")
    
    # 检查是否有切片没有覆盖任何脑区
    slices_with_regions = cover_matrix.sum(axis=1)
    empty_slices = np.where(slices_with_regions == 0)[0]
    if len(empty_slices) > 0:
        print(f"\n⚠️ Warning: {len(empty_slices)} slices contain no regions:")
        print(f"  Slices: {empty_slices}")
    else:
        print(f"\n✓ All {num_slices} slices contain at least one region")
    
    # 6. 可视化（可选）
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 覆盖矩阵热图
        sns.heatmap(cover_matrix.T, cmap='YlOrRd', ax=axes[0, 0], 
                   cbar_kws={'label': 'Coverage'})
        axes[0, 0].set_xlabel('Slice Index')
        axes[0, 0].set_ylabel('Region Index')
        axes[0, 0].set_title('Cover Matrix Heatmap')
        
        # 每个切片的脑区数量
        axes[0, 1].plot(cover_matrix.sum(axis=1))
        axes[0, 1].set_xlabel('Slice Index')
        axes[0, 1].set_ylabel('Number of Regions')
        axes[0, 1].set_title('Regions per Slice')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 每个脑区出现的切片数量
        axes[1, 0].hist(cover_matrix.sum(axis=0), bins=30, edgecolor='black')
        axes[1, 0].set_xlabel('Number of Slices')
        axes[1, 0].set_ylabel('Number of Regions')
        axes[1, 0].set_title('Slices per Region Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 稀疏性可视化
        sparsity = 1 - (cover_matrix.sum() / cover_matrix.size)
        axes[1, 1].text(0.5, 0.6, f'Sparsity: {sparsity:.2%}', 
                       ha='center', va='center', fontsize=16)
        axes[1, 1].text(0.5, 0.4, f'Non-zero: {cover_matrix.sum():.0f} / {cover_matrix.size}',
                       ha='center', va='center', fontsize=12)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Matrix Statistics')
        
        plt.tight_layout()
        
        # 保存图像
        output_dir = Path(output_path).parent
        viz_path = output_dir / 'cover_matrix_visualization.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Visualization saved to: {viz_path}")
        
    except ImportError:
        print("\n⚠️ matplotlib/seaborn not available, skipping visualization")
    
    # 7. 保存
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path, cover_matrix)
    print(f"\n✓ Cover matrix saved to: {output_path}")
    
    return cover_matrix


def generate_cover_matrix_hcpex(
    atlas_path,
    output_ctx_path="./data/cover_matrix_ctx.npy",
    output_sub_path="./data/cover_matrix_sub.npy",
    num_ctx_regions=360,
    num_sub_regions=66,
    axis=2
):
    """
    从HCPex图谱生成皮层和皮下覆盖矩阵
    
    HCPex标签约定：
    - 1-360: 皮层脑区（HCP-MMP）
    - 361-426: 皮下结构（66个）
    
    Args:
        atlas_path: HCPex图谱路径
        output_ctx_path: 皮层覆盖矩阵输出路径
        output_sub_path: 皮下覆盖矩阵输出路径
        num_ctx_regions: 皮层脑区数量
        num_sub_regions: 皮下结构数量
        axis: 切片方向
    
    Returns:
        cover_ctx, cover_sub: 两个覆盖矩阵
    """
    
    print("="*60)
    print("Generating Cover Matrices from HCPex Atlas")
    print("="*60)
    print(f"Atlas path: {atlas_path}")
    print(f"Cortical regions: {num_ctx_regions}")
    print(f"Subcortical regions: {num_sub_regions}")
    
    # 1. 加载图谱
    print("\nLoading HCPex atlas...")
    atlas_nii = nib.load(atlas_path)
    atlas_data = atlas_nii.get_fdata().astype(np.int32)
    
    print(f"  Atlas shape: {atlas_data.shape}")
    print(f"  Unique labels: {len(np.unique(atlas_data))}")
    print(f"  Label range: [{atlas_data.min()}, {atlas_data.max()}]")
    
    # 2. 获取切片数
    num_slices = atlas_data.shape[axis]
    print(f"\nNumber of slices: {num_slices}")
    
    # 3. 初始化覆盖矩阵
    cover_ctx = np.zeros((num_slices, num_ctx_regions), dtype=np.float32)
    cover_sub = np.zeros((num_slices, num_sub_regions), dtype=np.float32)
    
    # 4. 处理每个切片
    print("\nGenerating cover matrices...")
    for s in tqdm(range(num_slices), desc="Processing slices"):
        # 提取切片
        if axis == 0:
            slice_data = atlas_data[s, :, :]
        elif axis == 1:
            slice_data = atlas_data[:, s, :]
        else:
            slice_data = atlas_data[:, :, s]
        
        # 获取标签
        labels_in_slice = np.unique(slice_data)
        labels_in_slice = labels_in_slice[labels_in_slice > 0]
        
        for label in labels_in_slice:
            if 1 <= label <= num_ctx_regions:
                # 皮层脑区
                cover_ctx[s, label - 1] = 1.0
            elif num_ctx_regions < label <= num_ctx_regions + num_sub_regions:
                # 皮下结构
                cover_sub[s, label - num_ctx_regions - 1] = 1.0
    
    # 5. 统计
    print("\n" + "="*60)
    print("Cortical Cover Matrix Statistics")
    print("="*60)
    print(f"Shape: {cover_ctx.shape}")
    print(f"Average regions per slice: {cover_ctx.sum(axis=1).mean():.2f}")
    print(f"Coverage: {(cover_ctx.sum(axis=0) > 0).sum()} / {num_ctx_regions} regions")
    
    print("\n" + "="*60)
    print("Subcortical Cover Matrix Statistics")
    print("="*60)
    print(f"Shape: {cover_sub.shape}")
    print(f"Average regions per slice: {cover_sub.sum(axis=1).mean():.2f}")
    print(f"Coverage: {(cover_sub.sum(axis=0) > 0).sum()} / {num_sub_regions} regions")
    
    # 6. 保存
    output_dir = Path(output_ctx_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_ctx_path, cover_ctx)
    np.save(output_sub_path, cover_sub)
    
    print(f"\n✓ Cortical cover matrix saved to: {output_ctx_path}")
    print(f"✓ Subcortical cover matrix saved to: {output_sub_path}")
    
    return cover_ctx, cover_sub


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate cover matrix from atlas')
    parser.add_argument('--atlas', type=str, required=True, 
                       help='Path to atlas file (.nii or .nii.gz)')
    parser.add_argument('--atlas_type', type=str, default='hcp-mmp',
                       choices=['hcp-mmp', 'hcpex'],
                       help='Atlas type: hcp-mmp (360 cortical) or hcpex (360+66)')
    parser.add_argument('--output_ctx', type=str, default='./data/cover_matrix_ctx.npy',
                       help='Output path for cortical cover matrix')
    parser.add_argument('--output_sub', type=str, default='./data/cover_matrix_sub.npy',
                       help='Output path for subcortical cover matrix (HCPex only)')
    parser.add_argument('--axis', type=int, default=2,
                       choices=[0, 1, 2],
                       help='Slice axis: 0=Sagittal, 1=Coronal, 2=Axial')
    
    args = parser.parse_args()
    
    if args.atlas_type == 'hcp-mmp':
        # 只生成皮层覆盖矩阵
        cover_ctx = generate_cover_matrix_from_atlas(
            atlas_path=args.atlas,
            output_path=args.output_ctx,
            num_regions=360,
            axis=args.axis
        )
    else:  # hcpex
        # 生成皮层和皮下覆盖矩阵
        cover_ctx, cover_sub = generate_cover_matrix_hcpex(
            atlas_path=args.atlas,
            output_ctx_path=args.output_ctx,
            output_sub_path=args.output_sub,
            num_ctx_regions=360,
            num_sub_regions=66,
            axis=args.axis
        )
    
    print("\n" + "="*60)
    print("✓ Cover matrix generation completed!")
    print("="*60)
