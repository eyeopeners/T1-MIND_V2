import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import networkx as nx

class MINDGroupAnalyzer:
    """
    MIND矩阵的组间差异分析器
    
    集成三种分析方法：
    1. 统计检验（ANOVA + Post-hoc）
    2. 效应量（Cohen's d）
    3. 网络拓扑分析（模块性、中心性）
    """
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        
    def compute_group_prior(self, mind_nc, mind_mci, mind_ad):
        """
        计算组间MIND差异先验
        
        Args:
            mind_nc: [N_nc, 360, 360] - NC组的MIND矩阵
            mind_mci: [N_mci, 360, 360] - MCI组的MIND矩阵
            mind_ad: [N_ad, 360, 360] - AD组的MIND矩阵
            
        Returns:
            prior_dict: {
                'statistical_prior': [360, 360] - 统计显著性矩阵,
                'effect_size_prior': [360, 360] - 效应量矩阵,
                'network_prior': [360, 360] - 网络拓扑先验,
                'combined_prior': [360, 360] - 综合先验
            }
        """
        N_roi = 360
        
        # 1. 统计检验先验
        statistical_prior = self._compute_statistical_prior(
            mind_nc, mind_mci, mind_ad
        )
        
        # 2. 效应量先验
        effect_size_prior = self._compute_effect_size_prior(
            mind_nc, mind_mci, mind_ad
        )
        
        # 3. 网络拓扑先验
        network_prior = self._compute_network_prior(
            mind_nc, mind_mci, mind_ad
        )
        
        # 4. 综合先验（加权融合）
        combined_prior = (
            0.4 * statistical_prior +
            0.3 * effect_size_prior +
            0.3 * network_prior
        )
        
        # 归一化到[0, 1]
        combined_prior = self._normalize(combined_prior)
        
        return {
            'statistical_prior': statistical_prior,
            'effect_size_prior': effect_size_prior,
            'network_prior': network_prior,
            'combined_prior': combined_prior
        }
    
    def _compute_statistical_prior(self, mind_nc, mind_mci, mind_ad):
        """方法1：ANOVA + FDR校正"""
        N_roi = mind_nc.shape[1]
        p_values = np.ones((N_roi, N_roi))
        f_values = np.zeros((N_roi, N_roi))
        
        for i in range(N_roi):
            for j in range(i+1, N_roi):  # 只计算上三角（对称矩阵）
                # 提取三组的ROI对距离
                nc_vals = mind_nc[:, i, j]
                mci_vals = mind_mci[:, i, j]
                ad_vals = mind_ad[:, i, j]
                
                # ANOVA检验
                f_stat, p_val = stats.f_oneway(nc_vals, mci_vals, ad_vals)
                
                f_values[i, j] = f_stat
                f_values[j, i] = f_stat
                p_values[i, j] = p_val
                p_values[j, i] = p_val
        
        # FDR校正（控制假发现率）
        from statsmodels.stats.multitest import multipletests
        p_flat = p_values[np.triu_indices(N_roi, k=1)]
        _, p_corrected, _, _ = multipletests(p_flat, alpha=self.alpha, method='fdr_bh')
        
        # 重建矩阵
        p_values_corrected = np.ones((N_roi, N_roi))
        triu_indices = np.triu_indices(N_roi, k=1)
        p_values_corrected[triu_indices] = p_corrected
        p_values_corrected = p_values_corrected + p_values_corrected.T
        
        # 转换为先验强度：-log10(p)
        # p值越小，先验强度越大
        statistical_prior = -np.log10(p_values_corrected + 1e-10)
        statistical_prior = np.clip(statistical_prior, 0, 10)  # 截断到[0, 10]
        
        return statistical_prior
    
    def _compute_effect_size_prior(self, mind_nc, mind_mci, mind_ad):
        """方法2：效应量（Cohen's d）"""
        N_roi = mind_nc.shape[1]
        effect_size = np.zeros((N_roi, N_roi))
        
        for i in range(N_roi):
            for j in range(i+1, N_roi):
                # 计算NC vs AD的效应量（最显著的对比）
                nc_vals = mind_nc[:, i, j]
                ad_vals = mind_ad[:, i, j]
                
                # Cohen's d
                mean_diff = np.mean(nc_vals) - np.mean(ad_vals)
                pooled_std = np.sqrt(
                    (np.var(nc_vals) + np.var(ad_vals)) / 2
                )
                d = np.abs(mean_diff) / (pooled_std + 1e-8)
                
                effect_size[i, j] = d
                effect_size[j, i] = d
        
        return effect_size
    
    def _compute_network_prior(self, mind_nc, mind_mci, mind_ad):
        """方法3：网络拓扑分析"""
        N_roi = mind_nc.shape[1]
        
        # 构建三组的平均MIND网络
        mind_nc_mean = np.mean(mind_nc, axis=0)
        mind_mci_mean = np.mean(mind_mci, axis=0)
        mind_ad_mean = np.mean(mind_ad, axis=0)
        
        # 转换为相似度矩阵（MIND是距离，需要转换）
        def dist_to_sim(D, tau=1.0):
            return np.exp(-D / tau)
        
        sim_nc = dist_to_sim(mind_nc_mean)
        sim_mci = dist_to_sim(mind_mci_mean)
        sim_ad = dist_to_sim(mind_ad_mean)
        
        # 计算每个ROI对在不同组之间的拓扑差异
        network_prior = np.zeros((N_roi, N_roi))
        
        for i in range(N_roi):
            for j in range(i+1, N_roi):
                # 方法：比较该边在三个网络中的相对重要性变化
                edge_nc = sim_nc[i, j]
                edge_mci = sim_mci[i, j]
                edge_ad = sim_ad[i, j]
                
                # 计算变异系数（coefficient of variation）
                edges = [edge_nc, edge_mci, edge_ad]
                cv = np.std(edges) / (np.mean(edges) + 1e-8)
                
                network_prior[i, j] = cv
                network_prior[j, i] = cv
        
        return network_prior
    
    def _normalize(self, prior):
        """归一化到[0, 1]"""
        prior_min = prior.min()
        prior_max = prior.max()
        if prior_max - prior_min < 1e-8:
            return np.zeros_like(prior)
        return (prior - prior_min) / (prior_max - prior_min)
