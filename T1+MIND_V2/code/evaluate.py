# evaluate.py
"""
评估脚本：深度分析模型性能
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from tqdm import tqdm
import yaml

from model import TriLevelPriorFusionNetwork
from dataset import ADDiagnosisDataset


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, checkpoint_path, cfg_path, device='cuda'):
        self.device = torch.device(device)
        
        # 加载配置
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # 加载模型
        self.model = TriLevelPriorFusionNetwork(self.cfg['model'])
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 加载组间先验
        group_prior = np.load(self.cfg['data']['group_prior_path'])
        self.model.group_prior.copy_(torch.from_numpy(group_prior).float())
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Best epoch: {checkpoint['epoch']}")
        
    @torch.no_grad()
    def evaluate_dataset(self, dataset_name='PUSH', save_dir='./results'):
        """
        全面评估数据集
        
        Args:
            dataset_name: 'ADNI', 'PUSH', or 'SMHC'
            save_dir: 结果保存目录
        """
        save_dir = Path(save_dir) / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建数据集
        if dataset_name == 'PUSH':
            manifest_path = self.cfg['data']['push_test_manifest']
        elif dataset_name == 'SMHC':
            manifest_path = self.cfg['data']['smhc_test_manifest']
        else:
            manifest_path = self.cfg['data']['adni_val_manifest']
        
        dataset = ADDiagnosisDataset(
            manifest_path=manifest_path,
            split='test',
            num_slices=self.cfg['model']['num_slices'],
            transform=None,
            dataset_name=dataset_name
        )
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 收集预测结果和中间表示
        all_labels = []
        all_preds_3way = []
        all_probs_3way = []
        all_subject_ids = []
        
        # 中间表示
        all_text_emb = []
        all_z_img = []
        all_z_graph = []
        all_z_individual = []
        all_modal_weights = []
        all_slice_attention = []
        all_selected_indices = []
        all_node_attention = []
        
        print(f"\nEvaluating {dataset_name} dataset...")
        for batch in tqdm(loader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            out = self.model(batch)
            
            # 预测
            probs_3way = torch.softmax(out['logits_3way'], dim=-1)
            preds_3way = probs_3way.argmax(dim=-1)
            
            all_labels.append(batch['label'].cpu().numpy())
            all_preds_3way.append(preds_3way.cpu().numpy())
            all_probs_3way.append(probs_3way.cpu().numpy())
            all_subject_ids.extend(batch['subject_id'])
            
            # 中间表示
            all_text_emb.append(out['text_embeddings'].cpu().numpy())
            all_z_img.append(out['z_img'].cpu().numpy())
            all_z_graph.append(out['z_graph'].cpu().numpy())
            all_z_individual.append(out['z_individual'].cpu().numpy())
            all_modal_weights.append(out['modal_weights'].cpu().numpy())
            all_slice_attention.append(out['slice_attention'].cpu().numpy())
            all_selected_indices.append(out['selected_indices'].cpu().numpy())
            all_node_attention.append(out['node_attention'].cpu().numpy())
        
        # 合并结果
        all_labels = np.concatenate(all_labels)
        all_preds_3way = np.concatenate(all_preds_3way)
        all_probs_3way = np.concatenate(all_probs_3way)
        
        all_text_emb = np.concatenate(all_text_emb)
        all_z_img = np.concatenate(all_z_img)
        all_z_graph = np.concatenate(all_z_graph)
        all_z_individual = np.concatenate(all_z_individual)
        all_modal_weights = np.concatenate(all_modal_weights)
        all_slice_attention = np.concatenate(all_slice_attention)
        all_selected_indices = np.concatenate(all_selected_indices)
        all_node_attention = np.concatenate(all_node_attention)
        
        # 保存原始结果
        results = {
            'subject_ids': all_subject_ids,
            'labels': all_labels,
            'predictions': all_preds_3way,
            'probabilities': all_probs_3way,
            'text_embeddings': all_text_emb,
            'z_img': all_z_img,
            'z_graph': all_z_graph,
            'z_individual': all_z_individual,
            'modal_weights': all_modal_weights,
            'slice_attention': all_slice_attention,
            'selected_indices': all_selected_indices,
            'node_attention': all_node_attention
        }
        
        np.savez(save_dir / 'predictions.npz', **results)
        
        # 1. 分类报告
        self._generate_classification_report(
            all_labels, all_preds_3way, all_probs_3way, save_dir
        )
        
        # 2. 混淆矩阵
        self._plot_confusion_matrix(all_labels, all_preds_3way, save_dir)
        
        # 3. ROC曲线
        self._plot_roc_curves(all_labels, all_probs_3way, save_dir)
        
        # 4. PR曲线
        self._plot_pr_curves(all_labels, all_probs_3way, save_dir)
        
        # 5. 模态权重分析
        self._analyze_modal_weights(all_modal_weights, all_labels, save_dir)
        
        # 6. 切片注意力分析
        self._analyze_slice_attention(all_slice_attention, all_selected_indices, 
                                      all_labels, save_dir)
        
        # 7. ROI注意力分析
        self._analyze_roi_attention(all_node_attention, all_labels, save_dir)
        
        # 8. 特征空间可视化
        self._visualize_feature_space(
            all_z_img, all_z_graph, all_z_individual, all_labels, save_dir
        )
        
        print(f"\n✓ Results saved to {save_dir}")
        
    def _generate_classification_report(self, labels, preds, probs, save_dir):
        """生成分类报告"""
        # 分类报告
        report = classification_report(
            labels, preds, 
            target_names=['NC', 'MCI', 'AD'],
            digits=4
        )
        
        print("\n" + "="*60)
        print("Classification Report")
        print("="*60)
        print(report)
        
        # 保存
        with open(save_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        # 详细的分类指标（包括置信度分析）
        df_results = pd.DataFrame({
            'subject_id': range(len(labels)),
            'true_label': labels,
            'pred_label': preds,
            'prob_NC': probs[:, 0],
            'prob_MCI': probs[:, 1],
            'prob_AD': probs[:, 2],
            'max_prob': probs.max(axis=1),
            'correct': (labels == preds).astype(int)
        })
        
        df_results.to_csv(save_dir / 'detailed_results.csv', index=False)
        
        # 按置信度分层的准确率
        bins = [0, 0.5, 0.7, 0.9, 1.0]
        df_results['confidence_bin'] = pd.cut(df_results['max_prob'], bins=bins)
        
        print("\nAccuracy by Confidence Level:")
        print(df_results.groupby('confidence_bin')['correct'].agg(['count', 'mean']))
        
    def _plot_confusion_matrix(self, labels, preds, save_dir):
        """绘制混淆矩阵"""
        cm = confusion_matrix(labels, preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 原始混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['NC', 'MCI', 'AD'],
                   yticklabels=['NC', 'MCI', 'AD'],
                   ax=axes[0])
        axes[0].set_title('Confusion Matrix (Counts)')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # 归一化混淆矩阵
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=['NC', 'MCI', 'AD'],
                   yticklabels=['NC', 'MCI', 'AD'],
                   ax=axes[1])
        axes[1].set_title('Confusion Matrix (Normalized)')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_roc_curves(self, labels, probs, save_dir):
        """绘制ROC曲线"""
        from sklearn.preprocessing import label_binarize
        
        # 二值化标签
        labels_bin = label_binarize(labels, classes=[0, 1, 2])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 为每个类别绘制ROC曲线
        class_names = ['NC', 'MCI', 'AD']
        colors = ['blue', 'orange', 'red']
        
        for i, (name, color) in enumerate(zip(class_names, colors)):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'{name} (AUC = {roc_auc:.3f})')
        
        # 对角线
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_pr_curves(self, labels, probs, save_dir):
        """绘制Precision-Recall曲线"""
        from sklearn.preprocessing import label_binarize
        
        labels_bin = label_binarize(labels, classes=[0, 1, 2])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        class_names = ['NC', 'MCI', 'AD']
        colors = ['blue', 'orange', 'red']
        
        for i, (name, color) in enumerate(zip(class_names, colors)):
            precision, recall, _ = precision_recall_curve(labels_bin[:, i], probs[:, i])
            pr_auc = auc(recall, precision)
            
            ax.plot(recall, precision, color=color, lw=2,
                   label=f'{name} (AUC = {pr_auc:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves', fontsize=14)
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _analyze_modal_weights(self, modal_weights, labels, save_dir):
        """分析模态权重（可解释性）"""
        # modal_weights: [N, 4] - (img, graph, subcortex, individual)
        
        modal_names = ['Image', 'Graph (MIND)', 'Subcortex', 'Individual']
        
        # 按类别统计
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for cls_idx, cls_name in enumerate(['NC', 'MCI', 'AD']):
            mask = (labels == cls_idx)
            weights_cls = modal_weights[mask]
            
            # 箱线图
            axes[cls_idx].boxplot(weights_cls, labels=modal_names)
            axes[cls_idx].set_title(f'{cls_name} (N={mask.sum()})', fontsize=14)
            axes[cls_idx].set_ylabel('Weight', fontsize=12)
            axes[cls_idx].tick_params(axis='x', rotation=45)
            axes[cls_idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'modal_weights_by_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 统计表
        df_weights = pd.DataFrame(modal_weights, columns=modal_names)
        df_weights['label'] = labels
        
        summary = df_weights.groupby('label').agg(['mean', 'std'])
        print("\nModal Weights Summary:")
        print(summary)
        
        summary.to_csv(save_dir / 'modal_weights_summary.csv')
        
    def _analyze_slice_attention(self, slice_attention, selected_indices, labels, save_dir):
        """分析切片注意力分布"""
        # slice_attention: [N, S]
        # selected_indices: [N, K]
        
        # 1. 平均注意力分布（按类别）
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for cls_idx, cls_name in enumerate(['NC', 'MCI', 'AD']):
            mask = (labels == cls_idx)
            attn_cls = slice_attention[mask].mean(axis=0)  # [S]
            
            axes[cls_idx].plot(attn_cls, linewidth=2)
            axes[cls_idx].set_title(f'{cls_name} - Avg Slice Attention', fontsize=14)
            axes[cls_idx].set_xlabel('Slice Index', fontsize=12)
            axes[cls_idx].set_ylabel('Attention Score', fontsize=12)
            axes[cls_idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'slice_attention_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 选中切片的分布（热图）
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 统计每个切片被选中的次数
        S = slice_attention.shape[1]
        K = selected_indices.shape[1]
        
        selection_counts = np.zeros((3, S))  # [3类, S切片]
        
        for cls_idx in range(3):
            mask = (labels == cls_idx)
            indices_cls = selected_indices[mask]  # [N_cls, K]
            
            for s in range(S):
                selection_counts[cls_idx, s] = (indices_cls == s).sum()
        
        # 归一化
        selection_counts = selection_counts / selection_counts.sum(axis=1, keepdims=True)
        
        sns.heatmap(selection_counts, cmap='YlOrRd', 
                   yticklabels=['NC', 'MCI', 'AD'],
                   xticklabels=[f'{i}' if i % 10 == 0 else '' for i in range(S)],
                   ax=ax)
        ax.set_title('Slice Selection Frequency', fontsize=14)
        ax.set_xlabel('Slice Index', fontsize=12)
        ax.set_ylabel('Class', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'slice_selection_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _analyze_roi_attention(self, node_attention, labels, save_dir):
        """分析ROI注意力（识别判别性脑区）"""
        # node_attention: [N, 360]
        
        # 加载HCP-MMP图谱的ROI名称（如果有）
        # 这里简化处理，使用ROI索引
        
        # 1. Top判别ROI（按类别）
        top_k = 20
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for cls_idx, cls_name in enumerate(['NC', 'MCI', 'AD']):
            mask = (labels == cls_idx)
            attn_cls = node_attention[mask].mean(axis=0)  # [360]
            
            # 排序
            top_indices = np.argsort(attn_cls)[-top_k:][::-1]
            top_values = attn_cls[top_indices]
            
            # 条形图
            axes[cls_idx].barh(range(top_k), top_values)
            axes[cls_idx].set_yticks(range(top_k))
            axes[cls_idx].set_yticklabels([f'ROI {i}' for i in top_indices])
            axes[cls_idx].set_xlabel('Attention Score', fontsize=12)
            axes[cls_idx].set_title(f'{cls_name} - Top {top_k} ROIs', fontsize=14)
            axes[cls_idx].invert_yaxis()
            axes[cls_idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'top_roi_attention.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROI注意力的类间差异
        # 计算每个ROI在三类之间的注意力方差
        attn_by_class = []
        for cls_idx in range(3):
            mask = (labels == cls_idx)
            attn_cls = node_attention[mask].mean(axis=0)
            attn_by_class.append(attn_cls)
        
        attn_by_class = np.stack(attn_by_class, axis=0)  # [3, 360]
        roi_variance = attn_by_class.var(axis=0)  # [360]
        
        # 找出方差最大的ROI（最判别的脑区）
        top_discriminative_rois = np.argsort(roi_variance)[-top_k:][::-1]
        
        print(f"\nTop {top_k} Discriminative ROIs (by variance):")
        for i, roi_idx in enumerate(top_discriminative_rois):
            print(f"  {i+1}. ROI {roi_idx}: variance={roi_variance[roi_idx]:.4f}, "
                  f"NC={attn_by_class[0, roi_idx]:.4f}, "
                  f"MCI={attn_by_class[1, roi_idx]:.4f}, "
                  f"AD={attn_by_class[2, roi_idx]:.4f}")
        
        # 保存
        np.save(save_dir / 'top_discriminative_rois.npy', top_discriminative_rois)
        
    def _visualize_feature_space(self, z_img, z_graph, z_individual, labels, save_dir):
        """可视化特征空间（t-SNE）"""
        from sklearn.manifold import TSNE
        
        # 拼接所有特征
        z_all = np.concatenate([z_img, z_graph, z_individual], axis=1)
        
        # t-SNE降维
        print("\nComputing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        z_2d = tsne.fit_transform(z_all)
        
        # 绘制
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['blue', 'orange', 'red']
        markers = ['o', 's', '^']
        class_names = ['NC', 'MCI', 'AD']
        
        for cls_idx, (color, marker, name) in enumerate(zip(colors, markers, class_names)):
            mask = (labels == cls_idx)
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                      c=color, marker=marker, s=50, alpha=0.6,
                      label=f'{name} (N={mask.sum()})',
                      edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('Feature Space Visualization (t-SNE)', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'tsne_feature_space.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ t-SNE visualization saved")


def main():
    # 配置
    checkpoint_path = "./checkpoints/TPFN_v1/best_model.pth"
    cfg_path = "./config.yaml"
    
    evaluator = ModelEvaluator(checkpoint_path, cfg_path)
    
    # 评估所有数据集
    for dataset_name in ['ADNI', 'PUSH', 'SMHC']:
        print(f"\n{'='*60}")
        print(f"Evaluating {dataset_name}")
        print(f"{'='*60}")
        
        evaluator.evaluate_dataset(
            dataset_name=dataset_name,
            save_dir=f'./results/{dataset_name}'
        )


if __name__ == "__main__":
    main()

