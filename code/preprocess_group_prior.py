# preprocess_group_prior.py - CSV格式版本

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import f_oneway
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
ADNI_MANIFEST = r"C:\T1+MIND_V2\data\ADNI\manifest_train.csv"
SAVE_PATH = r"C:\T1+MIND_V2\code\group_priors"

def load_single_mind_matrix(mind_path):
    """
    加载单个MIND矩阵（CSV格式）
    
    CSV格式说明：
    - 第一行：空值, lh_L-181, lh_L-182, ..., rh_R-180
    - 第一列：lh_L-181, lh_L-182, ..., rh_R-180
    - 数值部分：[360, 360]的距离矩阵，对角线为0
    
    Args:
        mind_path: MIND矩阵CSV文件路径
        
    Returns:
        mind_matrix: [360, 360] 的numpy数组
    """
    try:
        # 使用pandas读取CSV
        # index_col=0 表示第一列作为索引
        df = pd.read_csv(mind_path, index_col=0)
        
        # 转换为numpy数组
        mind_numeric = df.values.astype(np.float32)
        
        # 验证形状
        expected_shape = (360, 360)
        if mind_numeric.shape != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape}, got {mind_numeric.shape} "
                f"in file {mind_path}"
            )
        
        # 验证对角线为0
        diag_vals = np.diag(mind_numeric)
        if not np.allclose(diag_vals, 0, atol=1e-6):
            # 警告但不终止
            # print(f"Warning: Diagonal not exactly zero in {Path(mind_path).name}, forcing to zero")
            np.fill_diagonal(mind_numeric, 0)
        
        # 验证对称性（MIND矩阵应该是对称的）
        if not np.allclose(mind_numeric, mind_numeric.T, atol=1e-4):
            # 如果不对称，对称化
            # print(f"Warning: Matrix not symmetric in {Path(mind_path).name}, symmetrizing")
            mind_numeric = (mind_numeric + mind_numeric.T) / 2
        
        # 检查是否有NaN或Inf
        if np.isnan(mind_numeric).any():
            nan_count = np.isnan(mind_numeric).sum()
            print(f"Warning: {nan_count} NaN values found in {Path(mind_path).name}, replacing with 0")
            mind_numeric = np.nan_to_num(mind_numeric, nan=0.0)
        
        if np.isinf(mind_numeric).any():
            inf_count = np.isinf(mind_numeric).sum()
            print(f"Warning: {inf_count} Inf values found in {Path(mind_path).name}, replacing with 0")
            mind_numeric = np.nan_to_num(mind_numeric, posinf=0.0, neginf=0.0)
        
        # 验证数值范围（MIND距离应该是非负的）
        if (mind_numeric < 0).any():
            neg_count = (mind_numeric < 0).sum()
            print(f"Warning: {neg_count} negative values found in {Path(mind_path).name}, taking absolute")
            mind_numeric = np.abs(mind_numeric)
        
        return mind_numeric
        
    except FileNotFoundError:
        raise FileNotFoundError(f"MIND file not found: {mind_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Empty CSV file: {mind_path}")
    except Exception as e:
        raise Exception(f"Error loading {mind_path}: {str(e)}")


def load_mind_matrices(manifest_path, split='train'):
    """
    从manifest加载所有MIND矩阵
    
    Returns:
        mind_dict: {
            'NC': [N_nc, 360, 360],
            'MCI': [N_mci, 360, 360],
            'AD': [N_ad, 360, 360]
        }
    """
    df = pd.read_csv(manifest_path)
    
    # 如果有split列，过滤
    if 'split' in df.columns:
        df = df[df['split'] == split]
    
    mind_dict = {'NC': [], 'MCI': [], 'AD': []}
    label_map = {0: 'NC', 1: 'MCI', 2: 'AD'}
    
    print(f"\nLoading MIND matrices from {split} set...")
    print(f"Total samples: {len(df)}")
    print(f"  NC: {(df['label']==0).sum()}")
    print(f"  MCI: {(df['label']==1).sum()}")
    print(f"  AD: {(df['label']==2).sum()}")
    
    failed_loads = []
    success_count = {'NC': 0, 'MCI': 0, 'AD': 0}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading MIND"):
        mind_path = row['mind_path']
        label = label_map[row['label']]
        
        try:
            mind = load_single_mind_matrix(mind_path)
            mind_dict[label].append(mind)
            success_count[label] += 1
        except Exception as e:
            error_msg = str(e)
            # 只保存简短的错误信息
            failed_loads.append({
                'path': mind_path,
                'label': label,
                'error': error_msg[:100]  # 截断长错误信息
            })
            continue
    
    print(f"\n{'='*60}")
    print("Loading Summary:")
    print(f"{'='*60}")
    for key in ['NC', 'MCI', 'AD']:
        total = (df['label'] == {'NC': 0, 'MCI': 1, 'AD': 2}[key]).sum()
        success = success_count[key]
        print(f"{key}: {success}/{total} ({success/total*100:.1f}%) loaded successfully")
    
    # 转换为numpy数组
    for key in mind_dict:
        if len(mind_dict[key]) > 0:
            mind_dict[key] = np.stack(mind_dict[key], axis=0)
            print(f"\n{key} array shape: {mind_dict[key].shape}")
            print(f"  Mean: {mind_dict[key].mean():.4f}")
            print(f"  Std: {mind_dict[key].std():.4f}")
            print(f"  Range: [{mind_dict[key].min():.4f}, {mind_dict[key].max():.4f}]")
        else:
            raise ValueError(f"No {key} samples loaded! Check your data paths.")
    
    if failed_loads:
        print(f"\n{'='*60}")
        print(f"⚠️ Failed to load {len(failed_loads)} files")
        print(f"{'='*60}")
        
        # 按类别统计失败数
        fail_by_label = {'NC': 0, 'MCI': 0, 'AD': 0}
        for fail in failed_loads:
            fail_by_label[fail['label']] += 1
        
        print("Failed by label:")
        for key in ['NC', 'MCI', 'AD']:
            if fail_by_label[key] > 0:
                print(f"  {key}: {fail_by_label[key]} files")
        
        # 显示前5个失败案例
        print("\nFirst 5 failed cases:")
        for i, fail in enumerate(failed_loads[:5]):
            print(f"  {i+1}. {Path(fail['path']).name}")
            print(f"     Label: {fail['label']}")
            print(f"     Error: {fail['error']}")
        
        # 保存完整失败列表
        fail_df = pd.DataFrame(failed_loads)
        fail_path = Path(SAVE_PATH) / 'failed_loads.csv'
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        fail_df.to_csv(fail_path, index=False)
        print(f"\n  Full list saved to: {fail_path}")
    
    return mind_dict


def compute_statistical_prior(mind_dict):
    """
    计算统计学先验（ANOVA + FDR校正）
    
    Returns:
        prior_stat: [360, 360] - 统计先验权重
    """
    print("\n" + "="*60)
    print("Computing statistical prior (ANOVA)...")
    print("="*60)
    
    nc_mind = mind_dict['NC']    # [N_nc, 360, 360]
    mci_mind = mind_dict['MCI']  # [N_mci, 360, 360]
    ad_mind = mind_dict['AD']    # [N_ad, 360, 360]
    
    N_nc, N_mci, N_ad = len(nc_mind), len(mci_mind), len(ad_mind)
    print(f"Sample sizes: NC={N_nc}, MCI={N_mci}, AD={N_ad}")
    
    # 对每条边进行ANOVA
    prior_stat = np.zeros((360, 360), dtype=np.float32)
    p_values_all = []
    positions_all = []
    
    print("\nPerforming ANOVA for each edge...")
    for i in tqdm(range(360), desc="ANOVA"):
        for j in range(i+1, 360):  # 只计算上三角（对称矩阵）
            # 三组在边(i,j)上的值
            nc_vals = nc_mind[:, i, j]
            mci_vals = mci_mind[:, i, j]
            ad_vals = ad_mind[:, i, j]
            
            # ANOVA检验
            try:
                f_stat, p_val = f_oneway(nc_vals, mci_vals, ad_vals)
                
                # 保存p值和位置，用于FDR校正
                p_values_all.append(p_val)
                positions_all.append((i, j))
                
            except Exception as e:
                # 如果ANOVA失败（如方差为0），p值设为1
                p_values_all.append(1.0)
                positions_all.append((i, j))
    
    # FDR校正（Benjamini-Hochberg）
    print("\nApplying FDR correction (α=0.05)...")
    p_values_all = np.array(p_values_all)
    sorted_indices = np.argsort(p_values_all)
    n_tests = len(p_values_all)
    alpha = 0.05
    
    significant_count = 0
    for rank, idx in enumerate(sorted_indices):
        threshold = (rank + 1) / n_tests * alpha
        p_val = p_values_all[idx]
        
        if p_val <= threshold:
            # 显著，计算权重
            weight = -np.log10(p_val + 1e-10)
            i, j = positions_all[idx]
            prior_stat[i, j] = weight
            prior_stat[j, i] = weight
            significant_count += 1
        else:
            # 不再显著，后续的都不显著
            break
    
    # 归一化到[0, 1]
    max_weight = prior_stat.max()
    if max_weight > 0:
        prior_stat = prior_stat / max_weight
    
    print(f"\nStatistical prior computed:")
    print(f"  Significant edges: {significant_count} / {n_tests} ({significant_count/n_tests*100:.2f}%)")
    print(f"  Mean weight (non-zero): {prior_stat[prior_stat > 0].mean():.4f}")
    
    return prior_stat


def compute_effect_size_prior(mind_dict):
    """
    计算效应量先验（Cohen's d）
    
    Returns:
        prior_effect: [360, 360] - 效应量先验
    """
    print("\n" + "="*60)
    print("Computing effect size prior (Cohen's d)...")
    print("="*60)
    
    nc_mind = mind_dict['NC']
    mci_mind = mind_dict['MCI']
    ad_mind = mind_dict['AD']
    
    prior_effect = np.zeros((360, 360), dtype=np.float32)
    
    print("\nComputing effect sizes...")
    for i in tqdm(range(360), desc="Effect Size"):
        for j in range(i+1, 360):
            nc_vals = nc_mind[:, i, j]
            mci_vals = mci_mind[:, i, j]
            ad_vals = ad_mind[:, i, j]
            
            # 三个配对比较的Cohen's d
            # NC vs AD
            mean_diff_na = np.abs(nc_vals.mean() - ad_vals.mean())
            pooled_std_na = np.sqrt((nc_vals.var() + ad_vals.var()) / 2)
            d_na = mean_diff_na / (pooled_std_na + 1e-8)
            
            # NC vs MCI
            mean_diff_nm = np.abs(nc_vals.mean() - mci_vals.mean())
            pooled_std_nm = np.sqrt((nc_vals.var() + mci_vals.var()) / 2)
            d_nm = mean_diff_nm / (pooled_std_nm + 1e-8)
            
            # MCI vs AD
            mean_diff_ma = np.abs(mci_vals.mean() - ad_vals.mean())
            pooled_std_ma = np.sqrt((mci_vals.var() + ad_vals.var()) / 2)
            d_ma = mean_diff_ma / (pooled_std_ma + 1e-8)
            
            # 取最大效应量
            max_d = max(d_na, d_nm, d_ma)
            
            prior_effect[i, j] = max_d
            prior_effect[j, i] = max_d
    
    # 归一化
    max_effect = prior_effect.max()
    if max_effect > 0:
        prior_effect = prior_effect / max_effect
    
    print(f"\nEffect size prior computed:")
    print(f"  Max effect size: {prior_effect.max():.4f}")
    print(f"  Mean effect size (non-zero): {prior_effect[prior_effect > 0].mean():.4f}")
    print(f"  Large effects (d>0.8): {(prior_effect > 0.8).sum() / 2:.0f} edges")
    
    return prior_effect


def compute_network_topology_prior(mind_dict):
    """
    计算网络拓扑先验（变异系数）
    
    Returns:
        prior_topo: [360, 360] - 拓扑先验
    """
    print("\n" + "="*60)
    print("Computing network topology prior...")
    print("="*60)
    
    nc_mind = mind_dict['NC']
    mci_mind = mind_dict['MCI']
    ad_mind = mind_dict['AD']
    
    prior_topo = np.zeros((360, 360), dtype=np.float32)
    
    print("\nComputing coefficient of variation across groups...")
    for i in tqdm(range(360), desc="Topology"):
        for j in range(i+1, 360):
            # 三组的均值
            nc_mean = nc_mind[:, i, j].mean()
            mci_mean = mci_mind[:, i, j].mean()
            ad_mean = ad_mind[:, i, j].mean()
            
            means = np.array([nc_mean, mci_mean, ad_mean])
            
            # 变异系数
            cv = means.std() / (means.mean() + 1e-8)
            
            prior_topo[i, j] = cv
            prior_topo[j, i] = cv
    
    # 归一化
    max_cv = prior_topo.max()
    if max_cv > 0:
        prior_topo = prior_topo / max_cv
    
    print(f"\nTopology prior computed:")
    print(f"  Max CV: {prior_topo.max():.4f}")
    print(f"  Mean CV (non-zero): {prior_topo[prior_topo > 0].mean():.4f}")
    
    return prior_topo


def compute_and_save_group_prior(adni_manifest, save_dir):
    """
    完整流程：加载数据 → 计算三种先验 → 融合 → 保存
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Group Prior Computation Pipeline")
    print("="*60)
    print(f"Input manifest: {adni_manifest}")
    print(f"Output directory: {save_dir}")
    
    # 1. 加载MIND矩阵
    mind_dict = load_mind_matrices(adni_manifest, split='train')
    
    # 验证是否所有类别都有数据
    for key in ['NC', 'MCI', 'AD']:
        if key not in mind_dict or len(mind_dict[key]) == 0:
            raise ValueError(f"No {key} samples found in training data!")
    
    # 2. 计算三种先验
    prior_stat = compute_statistical_prior(mind_dict)
    prior_effect = compute_effect_size_prior(mind_dict)
    prior_topo = compute_network_topology_prior(mind_dict)
    
    # 3. 融合（加权平均）
    print("\n" + "="*60)
    print("Combining priors...")
    print("="*60)
    
    w_stat = 0.4
    w_effect = 0.3
    w_topo = 0.3
    
    combined_prior = (
        w_stat * prior_stat +
        w_effect * prior_effect +
        w_topo * prior_topo
    )
    
    # 再次归一化到[0, 1]
    combined_prior = combined_prior / (combined_prior.max() + 1e-8)
    
    print(f"\nCombined prior statistics:")
    print(f"  Shape: {combined_prior.shape}")
    print(f"  Range: [{combined_prior.min():.4f}, {combined_prior.max():.4f}]")
    print(f"  Mean: {combined_prior.mean():.4f}")
    print(f"  Std: {combined_prior.std():.4f}")
    print(f"  Non-zero edges: {(combined_prior > 0).sum() / 2:.0f} / {360*359/2:.0f}")
    
    # 4. 保存
    np.save(save_dir / 'statistical_prior.npy', prior_stat)
    np.save(save_dir / 'effect_size_prior.npy', prior_effect)
    np.save(save_dir / 'topology_prior.npy', prior_topo)
    np.save(save_dir / 'combined_prior.npy', combined_prior)
    
    print(f"\n✓ All priors saved to {save_dir}:")
    print(f"  - statistical_prior.npy")
    print(f"  - effect_size_prior.npy")
    print(f"  - topology_prior.npy")
    print(f"  - combined_prior.npy")
    
    # 5. 可视化（可选）
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 统计先验
        im1 = axes[0, 0].imshow(prior_stat, cmap='YlOrRd', aspect='auto')
        axes[0, 0].set_title('Statistical Prior (ANOVA + FDR)', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 效应量先验
        im2 = axes[0, 1].imshow(prior_effect, cmap='YlOrRd', aspect='auto')
        axes[0, 1].set_title("Effect Size Prior (Cohen's d)", fontsize=14, fontweight='bold')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 拓扑先验
        im3 = axes[1, 0].imshow(prior_topo, cmap='YlOrRd', aspect='auto')
        axes[1, 0].set_title('Topology Prior (CV)', fontsize=14, fontweight='bold')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 组合先验
        im4 = axes[1, 1].imshow(combined_prior, cmap='YlOrRd', aspect='auto')
        axes[1, 1].set_title(f'Combined Prior (weights: {w_stat}, {w_effect}, {w_topo})', 
                            fontsize=14, fontweight='bold')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        viz_path = save_dir / 'prior_visualization.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - prior_visualization.png")
        
    except ImportError:
        print("\n⚠️ matplotlib/seaborn not available, skipping visualization")
    except Exception as e:
        print(f"\n⚠️ Visualization failed: {str(e)}")
    
    return {
        'statistical': prior_stat,
        'effect_size': prior_effect,
        'topology': prior_topo,
        'combined': combined_prior
    }


if __name__ == "__main__":
    import sys
    
    try:
        # 运行
        prior_dict = compute_and_save_group_prior(ADNI_MANIFEST, SAVE_PATH)
        
        print("\n" + "="*60)
        print("🎉 Group prior computation completed successfully!")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ Error occurred:")
        print("="*60)
        print(f"{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
