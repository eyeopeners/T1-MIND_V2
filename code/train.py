# train.py
"""
训练脚本
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

from model import TriLevelPriorFusionNetwork
from dataset import ADDiagnosisDataset, CustomTransform


class Trainer:
    """训练器"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建保存目录
        self.save_dir = Path(cfg['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化wandb（可选）
        if cfg.get('use_wandb', False):
            wandb.init(
                project=cfg['project_name'],
                config=cfg,
                name=cfg['exp_name']
            )
        
        # 构建数据集
        self._build_datasets()
        
        # 构建模型
        self._build_model()
        
        # 优化器和调度器
        self._build_optimizer()
        
        # 指标记录
        self.best_val_f1 = 0.0
        self.best_epoch = 0


    def _build_datasets(self):
        """构建数据集"""
        cfg = self.cfg
        
        # ADNI训练集
        self.train_dataset = ADDiagnosisDataset(
            manifest_path=cfg['data']['adni_train_manifest'],
            split='train',
            num_slices=cfg['model']['num_slices'],
            transform=CustomTransform(split='train'),
            dataset_name='ADNI'
        )
        
        # ADNI验证集
        self.val_dataset = ADDiagnosisDataset(
            manifest_path=cfg['data']['adni_val_manifest'],
            split='val',
            num_slices=cfg['model']['num_slices'],
            transform=None,
            dataset_name='ADNI'
        )
        
        # PUSH测试集（外部验证）
        self.test_push_dataset = ADDiagnosisDataset(
            manifest_path=cfg['data']['push_test_manifest'],
            split='test',
            num_slices=cfg['model']['num_slices'],
            transform=None,
            dataset_name='PUSH'
        )
        
        # SMHC测试集（只包含NC和MCI）
        self.test_smhc_dataset = ADDiagnosisDataset(
            manifest_path=cfg['data']['smhc_test_manifest'],
            split='test',
            num_slices=cfg['model']['num_slices'],
            transform=None,
            dataset_name='SMHC'
        )
        
        # DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg['training']['batch_size'],
            shuffle=True,
            num_workers=cfg['training']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg['training']['batch_size'],
            shuffle=False,
            num_workers=cfg['training']['num_workers'],
            pin_memory=True
        )
        
        self.test_push_loader = DataLoader(
            self.test_push_dataset,
            batch_size=cfg['training']['batch_size'],
            shuffle=False,
            num_workers=cfg['training']['num_workers'],
            pin_memory=True
        )
        
        self.test_smhc_loader = DataLoader(
            self.test_smhc_dataset,
            batch_size=cfg['training']['batch_size'],
            shuffle=False,
            num_workers=cfg['training']['num_workers'],
            pin_memory=True
        )
        
        print(f"Dataset sizes:")
        print(f"  Train (ADNI): {len(self.train_dataset)}")
        print(f"  Val (ADNI): {len(self.val_dataset)}")
        print(f"  Test (PUSH): {len(self.test_push_dataset)}")
        print(f"  Test (SMHC): {len(self.test_smhc_dataset)}")
        
    def _build_model(self):
        """构建模型"""
        cfg = self.cfg
        
        # 创建模型
        self.model = TriLevelPriorFusionNetwork(cfg['model'])
        
        # 加载组间先验
        group_prior_path = cfg['data']['group_prior_path']
        group_prior = np.load(group_prior_path)
        
        # ✓ 确保加载到模型的 buffer 中
        self.model.group_prior.copy_(torch.from_numpy(group_prior).float())
        
        print(f"Loaded group prior from {group_prior_path}")
        print(f"  Prior range: [{group_prior.min():.4f}, {group_prior.max():.4f}]")
        
        # 移动到设备
        self.model = self.model.to(self.device)
        
        # 打印模型参数量
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,} (trainable: {num_trainable:,})")
        
    def _build_optimizer(self):
        """构建优化器"""
        cfg = self.cfg
        
        # 确保学习率是浮点数
        base_lr = float(cfg.get('lr', 1e-4))  # 添加float()转换
        weight_decay = float(cfg.get('weight_decay', 1e-4))  # 添加float()转换
        
        # 分组参数（不同模块使用不同学习率）
        param_groups = [
            {
                'params': self.model.text_encoder.parameters(),
                'lr': base_lr * 0.1  # 文本编码器用更小的学习率
            },
            {
                'params': self.model.slice_selector.parameters(),
                'lr': base_lr
            },
            {
                'params': self.model.mind_encoder.parameters(),
                'lr': base_lr
            },
            {
                'params': self.model.individual_prior_encoder.parameters(),
                'lr': base_lr
            },
            {
                'params': self.model.subcortex_encoder.parameters(),
                'lr': base_lr
            },
            {
                'params': self.model.cross_modal_fusion.parameters(),
                'lr': base_lr
            },
            {
                'params': self.model.prediction_head.parameters(),
                'lr': base_lr * 2.0  # 预测头用更大的学习率
            }
        ]
        
        # 优化器
        optimizer_type = cfg.get('optimizer', 'adamw').lower()
        
        if optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                param_groups,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                param_groups,
                momentum=0.9,
                weight_decay=weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # 学习率调度器
        scheduler_type = cfg.get('scheduler', 'cosine').lower()
        
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.get('epochs', 100),
                eta_min=base_lr * 0.01
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.get('lr_step', 30),
                gamma=cfg.get('lr_gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            self.scheduler = None
        
        print(f"Optimizer: {optimizer_type}")
        print(f"Base learning rate: {base_lr}")
        print(f"Weight decay: {weight_decay}")
        print(f"Scheduler: {scheduler_type}")
        
    # train.py - 修改 train_epoch 方法

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        epoch_losses = {
            'loss_total': [],
            'loss_fused': [],
            'loss_ad': [],
            'loss_nm': [],
            'loss_text': [],
            'loss_sparse': []
        }
        
        for batch in pbar:
            # 移动到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            out = self.model(batch)
            
            # 计算损失
            loss_dict = self.model.compute_loss(out, batch['label'], self.cfg)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss_dict['loss_total'].backward()
            
            # ✓ 修复：使用 .get() 提供默认值
            max_grad_norm = self.cfg.get('training', {}).get('max_grad_norm', 1.0)
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_grad_norm  # ← 使用默认值
            )
            
            self.optimizer.step()
            
            # 记录
            for key in epoch_losses:
                epoch_losses[key].append(loss_dict[key].item())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['loss_total'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # 计算epoch平均损失
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        return avg_losses

    
    @torch.no_grad()
    def validate(self, loader, dataset_name='ADNI'):
        """验证 - 计算所有任务的指标"""
        self.model.eval()
        
        # 收集预测
        all_labels = []
        all_probs_3way = []
        all_probs_nc_ad = []
        all_probs_nc_mci = []
        all_probs_mci_ad = []
        
        for batch in tqdm(loader, desc=f"Validating on {dataset_name}"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            out = self.model(batch)
            
            # 三分类
            probs_3way = torch.softmax(out['logits_3way'], dim=-1)
            all_probs_3way.append(probs_3way.cpu().numpy())
            
            # NC vs AD
            probs_nc_ad = torch.softmax(out['logits_nc_ad'], dim=-1)
            all_probs_nc_ad.append(probs_nc_ad.cpu().numpy())
            
            # NC vs MCI
            probs_nc_mci = torch.softmax(out['logits_nc_mci'], dim=-1)
            all_probs_nc_mci.append(probs_nc_mci.cpu().numpy())
            
            # MCI vs AD
            probs_mci_ad = torch.softmax(out['logits_mci_ad'], dim=-1)
            all_probs_mci_ad.append(probs_mci_ad.cpu().numpy())
            
            all_labels.append(batch['label'].cpu().numpy())
        
        # 合并
        all_labels = np.concatenate(all_labels)
        all_probs_3way = np.concatenate(all_probs_3way)
        all_probs_nc_ad = np.concatenate(all_probs_nc_ad)
        all_probs_nc_mci = np.concatenate(all_probs_nc_mci)
        all_probs_mci_ad = np.concatenate(all_probs_mci_ad)
        
        all_preds_3way = all_probs_3way.argmax(axis=1)
        
        metrics = {}
        
        # ===== 1. 三分类指标 =====
        metrics['3way_acc'] = accuracy_score(all_labels, all_preds_3way)
        metrics['3way_f1_macro'] = f1_score(all_labels, all_preds_3way, average='macro')
        metrics['3way_f1_weighted'] = f1_score(all_labels, all_preds_3way, average='weighted')
        
        f1_per_class = f1_score(all_labels, all_preds_3way, average=None)
        metrics['3way_f1_nc'] = f1_per_class[0]
        metrics['3way_f1_mci'] = f1_per_class[1]
        metrics['3way_f1_ad'] = f1_per_class[2]
        
        # AUC (one-vs-rest)
        try:
            from sklearn.preprocessing import label_binarize
            labels_bin = label_binarize(all_labels, classes=[0, 1, 2])
            metrics['3way_auc_macro'] = roc_auc_score(
                labels_bin, all_probs_3way, average='macro', multi_class='ovr'
            )
            metrics['3way_auc_nc'] = roc_auc_score(labels_bin[:, 0], all_probs_3way[:, 0])
            metrics['3way_auc_mci'] = roc_auc_score(labels_bin[:, 1], all_probs_3way[:, 1])
            metrics['3way_auc_ad'] = roc_auc_score(labels_bin[:, 2], all_probs_3way[:, 2])
        except:
            pass
        
        # ===== 2. NC vs AD =====
        mask_nc_ad = (all_labels == 0) | (all_labels == 2)
        if mask_nc_ad.sum() > 0:
            labels_nc_ad = (all_labels[mask_nc_ad] == 2).astype(int)
            probs_nc_ad = all_probs_nc_ad[mask_nc_ad]
            preds_nc_ad = probs_nc_ad.argmax(axis=1)
            
            metrics['nc_ad_acc'] = accuracy_score(labels_nc_ad, preds_nc_ad)
            metrics['nc_ad_f1'] = f1_score(labels_nc_ad, preds_nc_ad)
            metrics['nc_ad_sensitivity'] = recall_score(labels_nc_ad, preds_nc_ad)
            metrics['nc_ad_specificity'] = recall_score(labels_nc_ad, preds_nc_ad, pos_label=0)
            try:
                metrics['nc_ad_auc'] = roc_auc_score(labels_nc_ad, probs_nc_ad[:, 1])
            except:
                metrics['nc_ad_auc'] = 0.0
        
        # ===== 3. NC vs MCI =====
        mask_nc_mci = (all_labels == 0) | (all_labels == 1)
        if mask_nc_mci.sum() > 0:
            labels_nc_mci = (all_labels[mask_nc_mci] == 1).astype(int)
            probs_nc_mci = all_probs_nc_mci[mask_nc_mci]
            preds_nc_mci = probs_nc_mci.argmax(axis=1)
            
            metrics['nc_mci_acc'] = accuracy_score(labels_nc_mci, preds_nc_mci)
            metrics['nc_mci_f1'] = f1_score(labels_nc_mci, preds_nc_mci)
            metrics['nc_mci_sensitivity'] = recall_score(labels_nc_mci, preds_nc_mci)
            metrics['nc_mci_specificity'] = recall_score(labels_nc_mci, preds_nc_mci, pos_label=0)
            try:
                metrics['nc_mci_auc'] = roc_auc_score(labels_nc_mci, probs_nc_mci[:, 1])
            except:
                metrics['nc_mci_auc'] = 0.0
        
        # ===== 4. MCI vs AD =====
        mask_mci_ad = (all_labels == 1) | (all_labels == 2)
        if mask_mci_ad.sum() > 0:
            labels_mci_ad = (all_labels[mask_mci_ad] == 2).astype(int)
            probs_mci_ad = all_probs_mci_ad[mask_mci_ad]
            preds_mci_ad = probs_mci_ad.argmax(axis=1)
            
            metrics['mci_ad_acc'] = accuracy_score(labels_mci_ad, preds_mci_ad)
            metrics['mci_ad_f1'] = f1_score(labels_mci_ad, preds_mci_ad)
            metrics['mci_ad_sensitivity'] = recall_score(labels_mci_ad, preds_mci_ad)
            metrics['mci_ad_specificity'] = recall_score(labels_mci_ad, preds_mci_ad, pos_label=0)
            try:
                metrics['mci_ad_auc'] = roc_auc_score(labels_mci_ad, probs_mci_ad[:, 1])
            except:
                metrics['mci_ad_auc'] = 0.0
        
        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds_3way)
        
        return metrics
    
    def train(self):
        """完整训练流程"""
        cfg = self.cfg['training']
        
        for epoch in range(1, cfg['num_epochs'] + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{cfg['num_epochs']}")
            print(f"{'='*60}")
            
            # 训练
            train_losses = self.train_epoch(epoch)
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 验证
            val_metrics = self.validate(self.val_loader, 'ADNI-Val')
            
            # ========== 详细的指标输出 ==========
            print(f"\n{'='*60}")
            print(f"Training Results - Epoch {epoch}")
            print(f"{'='*60}")
            
            # 训练损失
            print(f"\n[Train Losses]")
            print(f"  Total Loss: {train_losses['loss_total']:.4f}")
            print(f"  3-way Loss: {train_losses.get('loss_fused', 0.0):.4f}")
            print(f"  AD Loss: {train_losses.get('loss_ad', 0.0):.4f}")
            print(f"  NM Loss: {train_losses.get('loss_nm', 0.0):.4f}")
            
            # 三分类性能
            print(f"\n[Validation - 3-way Classification]")
            print(f"  Accuracy:  {val_metrics['3way_acc']:.4f}")
            print(f"  F1-Macro:  {val_metrics['3way_f1_macro']:.4f}")
            print(f"  F1-Weighted: {val_metrics['3way_f1_weighted']:.4f}")
            print(f"  AUC-Macro: {val_metrics.get('3way_auc_macro', 0.0):.4f}")
            
            print(f"\n  Per-class metrics:")
            print(f"    NC  - F1: {val_metrics['3way_f1_nc']:.4f}, AUC: {val_metrics.get('3way_auc_nc', 0.0):.4f}")
            print(f"    MCI - F1: {val_metrics['3way_f1_mci']:.4f}, AUC: {val_metrics.get('3way_auc_mci', 0.0):.4f}")
            print(f"    AD  - F1: {val_metrics['3way_f1_ad']:.4f}, AUC: {val_metrics.get('3way_auc_ad', 0.0):.4f}")
            
            # 三分类混淆矩阵
            cm = val_metrics['confusion_matrix']
            print(f"\n  Confusion Matrix:")
            print(f"                Predicted")
            print(f"                NC    MCI   AD")
            print(f"    Actual NC   {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}")
            print(f"           MCI  {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}")
            print(f"           AD   {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}")
            
            # NC vs AD
            if 'nc_ad_acc' in val_metrics:
                print(f"\n[Validation - NC vs AD]")
                print(f"  Accuracy:    {val_metrics['nc_ad_acc']:.4f}")
                print(f"  F1:          {val_metrics['nc_ad_f1']:.4f}")
                print(f"  AUC:         {val_metrics.get('nc_ad_auc', 0.0):.4f}")
                print(f"  Sensitivity: {val_metrics['nc_ad_sensitivity']:.4f}")
                print(f"  Specificity: {val_metrics['nc_ad_specificity']:.4f}")
            
            # NC vs MCI
            if 'nc_mci_acc' in val_metrics:
                print(f"\n[Validation - NC vs MCI]")
                print(f"  Accuracy:    {val_metrics['nc_mci_acc']:.4f}")
                print(f"  F1:          {val_metrics['nc_mci_f1']:.4f}")
                print(f"  AUC:         {val_metrics.get('nc_mci_auc', 0.0):.4f}")
                print(f"  Sensitivity: {val_metrics['nc_mci_sensitivity']:.4f}")
                print(f"  Specificity: {val_metrics['nc_mci_specificity']:.4f}")
            
            # MCI vs AD
            if 'mci_ad_acc' in val_metrics:
                print(f"\n[Validation - MCI vs AD]")
                print(f"  Accuracy:    {val_metrics['mci_ad_acc']:.4f}")
                print(f"  F1:          {val_metrics['mci_ad_f1']:.4f}")
                print(f"  AUC:         {val_metrics.get('mci_ad_auc', 0.0):.4f}")
                print(f"  Sensitivity: {val_metrics['mci_ad_sensitivity']:.4f}")
                print(f"  Specificity: {val_metrics['mci_ad_specificity']:.4f}")
            
            print(f"{'='*60}\n")
            # ======================================
            
            # 记录到wandb
            if self.cfg.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    **{f'train/{k}': v for k, v in train_losses.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items() 
                    if k != 'confusion_matrix'}
                })
            
            # 保存最佳模型
            if val_metrics['3way_f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['3way_f1_macro']
                self.best_epoch = epoch
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'val_metrics': val_metrics,
                    'cfg': self.cfg
                }, self.save_dir / 'best_model.pth')
                
                print(f"✓ Saved best model (F1={self.best_val_f1:.4f})")
            
            # 定期保存检查点
            if epoch % cfg['save_freq'] == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
                }, self.save_dir / f'checkpoint_epoch{epoch}.pth')
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best epoch: {self.best_epoch}, Best F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}")
        
        # 外部验证
        self._external_validation()
    
    def _external_validation(self):
        """外部验证（PUSH和SMHC）"""
        print(f"\n{'='*60}")
        print(f"External Validation")
        print(f"{'='*60}")
        
        # 加载最佳模型
        checkpoint = torch.load(self.save_dir / 'best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # PUSH
        print(f"\n[PUSH Dataset]")
        push_metrics = self.validate(self.test_push_loader, 'PUSH')
        
        print(f"3-way: ACC={push_metrics['3way_acc']:.4f}, "
            f"F1={push_metrics['3way_f1_macro']:.4f}, "
            f"AUC={push_metrics.get('3way_auc_macro', 0.0):.4f}")
        print(f"F1 per class: NC={push_metrics['3way_f1_nc']:.4f}, "
            f"MCI={push_metrics['3way_f1_mci']:.4f}, "
            f"AD={push_metrics['3way_f1_ad']:.4f}")
        
        # ✓ 修复：检查键是否存在
        if 'nc_ad_f1' in push_metrics:
            print(f"NC vs AD: F1={push_metrics['nc_ad_f1']:.4f}, "
                f"AUC={push_metrics.get('nc_ad_auc', 0.0):.4f}")
        
        print(f"Confusion Matrix:\n{push_metrics['confusion_matrix']}")
        
        # SMHC (只评估NC vs MCI)
        print(f"\n[SMHC Dataset - NC vs MCI only]")
        smhc_metrics = self.validate(self.test_smhc_loader, 'SMHC')
        
        # ✓ 修复：使用正确的键名
        if 'nc_mci_f1' in smhc_metrics:
            print(f"NC vs MCI: ACC={smhc_metrics.get('nc_mci_acc', 0.0):.4f}, "
                f"F1={smhc_metrics['nc_mci_f1']:.4f}, "
                f"AUC={smhc_metrics.get('nc_mci_auc', 0.0):.4f}")
        
        # 保存结果
        results = {
            'PUSH': push_metrics,
            'SMHC': smhc_metrics
        }
        
        torch.save(results, self.save_dir / 'external_validation_results.pth')
        print(f"\n✓ Saved external validation results")


def main():
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 设置随机种子
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    
    # 创建训练器
    trainer = Trainer(cfg)
    
    # 训练
    trainer.train()


if __name__ == "__main__":
    main()
