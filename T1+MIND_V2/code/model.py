import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import numpy as np

class TriLevelPriorFusionNetwork(nn.Module):
    """
    三层级先验融合网络（Tri-Level Prior Fusion Network, TPFN）
    
    架构设计：
    1. Text-Guided Slice Selection (文本引导的切片选择)
    2. Prior-Enhanced Graph Encoding (先验增强的图编码)
    3. Individual-Aware Feature Fusion (个体感知的特征融合)
    4. Multi-Task Prediction (多任务预测)
    
    创新点：
    - 使用GPT临床描述引导视觉注意力
    - 基于MIND组间统计分析的图网络调制
    - 个体MIND和人口学信息的深度融合
    - 跨模态注意力机制
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        
        # ================== Level 1: 文本先验模块 ==================
        self.text_encoder = TextPriorEncoder(
            clip_model_name=cfg.get("clip_model", "openai/clip-vit-base-patch32"),
            dim_out=256
        )
        
        # ================== Stage 1: 文本引导的切片选择 ==================
        self.slice_selector = TextGuidedSliceSelector(
            backbone=self._build_backbone(cfg),
            dim_slice=cfg.get("dim_slice", 256),
            num_slices=cfg.get("num_slices", 64),
            num_selected=cfg.get("num_selected", 16),
            dim_text=256,
            num_heads=8
        )
        
        # ================== Level 2: 组间形态学先验 ==================
        # 这个先验是预计算的，作为模型的固定参数
        self.register_buffer(
            'group_prior', 
            torch.zeros(360, 360)  # 将在训练前加载
        )
        
        # ================== Stage 2: 先验增强的图编码 ==================
        self.mind_encoder = PriorEnhancedMindEncoder(
            dim_node=360,  # MIND矩阵的行作为节点特征
            dim_hidden=256,
            num_roi=360,
            num_heads=8,
            num_layers=3
        )
        
        # ================== Stage 3: 个体先验增强 ==================
        self.individual_prior_encoder = IndividualPriorEncoder(
            dim_mind=360,
            dim_demo=cfg.get("dim_demo", 10),
            dim_out=128
        )
        
        # 皮下ROI处理
        self.subcortex_encoder = SubcortexEncoder(
            dim_slice=256,
            num_subcortex=66,
            dim_out=64
        )
        
        # ================== Stage 4: 跨模态融合 ==================
        self.cross_modal_fusion = CrossModalAttentionFusion(
            dim_img=256,
            dim_graph=256,
            dim_subcortex=64,
            dim_individual=128,
            dim_out=512,
            num_heads=8
        )
        
        # ================== Stage 5: 多任务预测头 ==================
        self.prediction_head = MultiTaskPredictionHead(
            dim_input=512,
            num_classes=3
        )
        
        # ================== 不确定性加权多任务学习 ==================
        self.task_uncertainty = nn.Parameter(torch.zeros(5))  # 5个任务
        # 不确定性权重（可学习参数）
        # 8个任务：3way, nc_ad, nc_mci, mci_ad, ad_vs_non, mci_vs_non, text, sparse
        self.log_vars = nn.Parameter(torch.zeros(8))
        
    def forward(self, batch):
        """
        前向传播
        
        Args:
            batch: dict包含
                - 'slices': [B, S, C, H, W] - 切片堆栈
                - 'mind': [B, 360, 360] - 个体MIND矩阵
                - 'demographics': [B, D] - 人口学信息
                - 'text_class': [B] - 类别标签（用于索引文本先验）
                - 'cover_ctx': [S, 360] - 切片-皮层ROI覆盖矩阵
                - 'cover_sub': [S, 66] - 切片-皮下ROI覆盖矩阵
                
        Returns:
            out: dict包含所有预测和中间表示
        """
        B = batch['slices'].shape[0]
        device = batch['slices'].device
        
        out = {}
        
        # ========== Level 1: 提取文本先验 ==========
        text_embeddings = self.text_encoder(batch['text_class'])  # [B, 256]
        out['text_embeddings'] = text_embeddings
        
        # ========== Stage 1: 文本引导的切片选择 ==========
        slice_out = self.slice_selector(
            batch['slices'], 
            text_prior=text_embeddings
        )
        selected_features = slice_out['features']  # [B, K, D]
        selected_indices = slice_out['selected_indices']  # [B, K]
        slice_attention = slice_out['attention_scores']  # [B, S]
        
        out.update({
            'selected_features': selected_features,
            'selected_indices': selected_indices,
            'slice_attention': slice_attention
        })
        
        # ========== 从选中的切片提取ROI特征 ==========
        # 皮层ROI
        roi_features_ctx = self._extract_roi_from_selected(
            selected_features, 
            selected_indices,
            batch['cover_ctx'],
            num_roi=360
        )  # [B, 360, D]
        
        # 皮下ROI
        roi_features_sub = self._extract_roi_from_selected(
            selected_features,
            selected_indices,
            batch['cover_sub'],
            num_roi=66
        )  # [B, 66, D]
        
        # ========== Stage 2: 先验增强的图编码 ==========
        graph_out = self.mind_encoder(
            batch['mind'],  # 个体MIND矩阵
            roi_features_ctx,  # 从影像提取的ROI特征
            self.group_prior  # 组间统计先验
        )
        
        z_graph = graph_out['graph_embedding']  # [B, 256]
        node_attention = graph_out['node_attention']  # [B, 360]
        
        out.update({
            'z_graph': z_graph,
            'node_attention': node_attention
        })
        
        # ========== Stage 3: 个体先验编码 ==========
        z_individual = self.individual_prior_encoder(
            batch['mind'],
            batch['demographics']
        )  # [B, 128]
        
        # 皮下编码
        z_subcortex = self.subcortex_encoder(roi_features_sub)  # [B, 64]
        
        # 影像级表示（从选中的切片聚合）
        z_img = selected_features.mean(dim=1)  # [B, D]
        
        out.update({
            'z_img': z_img,
            'z_individual': z_individual,
            'z_subcortex': z_subcortex
        })
        
        # ========== Stage 4: 跨模态融合 ==========
        fusion_out = self.cross_modal_fusion(
            z_img=z_img,
            z_graph=z_graph,
            z_subcortex=z_subcortex,
            z_individual=z_individual
        )
        
        h_fused = fusion_out['fused_features']  # [B, 512]
        modal_weights = fusion_out['modal_weights']  # [B, 4]
        
        out.update({
            'h_fused': h_fused,
            'modal_weights': modal_weights
        })
        
        # ========== Stage 5: 多任务预测 ==========
        predictions = self.prediction_head(h_fused)
        out.update(predictions)
        
        return out
    
    def _extract_roi_from_selected(self, selected_features, selected_indices, cover_matrix, num_roi):
        """
        从选中的切片中提取ROI特征
        
        Args:
            selected_features: [B, K, D]
            selected_indices: [B, K]
            cover_matrix: [S, R] - 切片-ROI覆盖矩阵
            num_roi: R
            
        Returns:
            roi_features: [B, R, D]
        """
        B, K, D = selected_features.shape
        S, R = cover_matrix.shape
        
        # 1. 根据选中的索引，提取对应的覆盖关系
        batch_indices = torch.arange(B, device=selected_features.device).unsqueeze(1).expand(-1, K)
        selected_cover = cover_matrix[selected_indices]  # [B, K, R]
        
        # 2. 归一化覆盖权重
        selected_cover = selected_cover / (selected_cover.sum(dim=1, keepdim=True) + 1e-8)
        
        # 3. 加权求和
        roi_features = torch.einsum('bkd,bkr->brd', selected_features, selected_cover)
        
        return roi_features
    
    def _build_backbone(self, cfg):
        """构建切片编码器的backbone"""
        backbone_type = cfg.get("backbone", "resnet18")
        
        if backbone_type == "resnet18":
            from torchvision.models import resnet18
            backbone = resnet18(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-2])  # 移除全连接层
            backbone.out_dim = 512
        elif backbone_type == "resnet50":
            from torchvision.models import resnet50
            backbone = resnet50(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone.out_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone_type}")
        
        return backbone
    
    def compute_loss(self, out, labels, cfg):
        """
        计算多任务损失（带不确定性加权）
        
        Args:
            out: 模型输出
            labels: [B] - 类别标签 (0=NC, 1=MCI, 2=AD)
            cfg: 配置
            
        Returns:
            loss_dict: 各项损失
        """
        # 1. 主任务：三分类
        loss_fused = self._focal_loss(
            out['logits_3way'], 
            labels,
            gamma=cfg.get("focal_gamma", 2.0)
        )
        
        # 2. 辅助任务1：AD vs 非AD
        labels_ad = (labels == 2).long()  # AD=1, 其他=0
        loss_ad = F.cross_entropy(out['logits_ad_vs_non'], labels_ad)
        
        # 3. 辅助任务2：NC vs MCI（只在非AD样本上）
        mask_non_ad = (labels != 2)
        if mask_non_ad.sum() > 0:
            labels_nm = labels[mask_non_ad]  # 0=NC, 1=MCI
            logits_nm = out['logits_nc_vs_mci'][mask_non_ad]
            loss_nm = F.cross_entropy(logits_nm, labels_nm)
        else:
            loss_nm = torch.tensor(0.0, device=labels.device)
        
        # 4. 文本对齐损失
        loss_text = self._text_alignment_loss(
            out['text_embeddings'],
            out['z_img'],
            labels
        )
        
        # 5. 稀疏性损失（鼓励切片选择的空间连续性）
        loss_sparse = self._sparse_loss(
            out['slice_attention'],
            out['selected_indices']
        )
        
        # 不确定性加权
        precision = torch.exp(-self.task_uncertainty)
        loss_total = (
            precision[0] * loss_fused + self.task_uncertainty[0] +
            precision[1] * loss_ad + self.task_uncertainty[1] +
            precision[2] * loss_nm + self.task_uncertainty[2] +
            precision[3] * loss_text + self.task_uncertainty[3] +
            precision[4] * loss_sparse + self.task_uncertainty[4]
        )
        
        return {
            'loss_total': loss_total,
            'loss_fused': loss_fused,
            'loss_ad': loss_ad,
            'loss_nm': loss_nm,
            'loss_text': loss_text,
            'loss_sparse': loss_sparse,
            'task_weights': precision.detach()
        }
    
    def _focal_loss(self, logits, labels, gamma=2.0):
        """Focal Loss"""
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** gamma) * ce_loss
        return focal_loss.mean()
    
    def _text_alignment_loss(self, text_emb, img_emb, labels):
        """文本-影像对齐损失（对比学习）"""
        # 归一化
        text_emb = F.normalize(text_emb, dim=-1)
        img_emb = F.normalize(img_emb, dim=-1)
        
        # 余弦相似度
        similarity = torch.mm(img_emb, text_emb.t())  # [B, B]
        
        # 对比损失：同类相似度高，异类相似度低
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        
        pos_sim = similarity[labels_eq].mean()
        neg_sim = similarity[~labels_eq].mean()
        
        loss = torch.clamp(0.5 - (pos_sim - neg_sim), min=0)
        
        return loss
    
    def _sparse_loss(self, attention_scores, selected_indices):
        """稀疏性损失：鼓励选中的切片在空间上连续"""
        # 对选中的索引排序
        indices_sorted = torch.sort(selected_indices, dim=-1).values
        
        # 计算相邻切片的间隔
        gaps = indices_sorted[:, 1:] - indices_sorted[:, :-1]
        
        # 希望间隔尽可能小（连续）
        continuity_loss = gaps.float().mean()
        
        # 熵正则化：希望注意力集中
        entropy = -torch.sum(
            attention_scores * torch.log(attention_scores + 1e-8),
            dim=-1
        ).mean()
        
        return 0.1 * continuity_loss + 0.01 * entropy


# ==================== 子模块实现 ====================

class TextPriorEncoder(nn.Module):
    """文本先验编码器（使用CLIP）"""
    
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", dim_out=256):
        super().__init__()
        
        # 加载CLIP模型
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # 冻结CLIP参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # 投影层
        clip_dim = self.clip_model.config.projection_dim
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, dim_out)
        )
        
        # 预定义的文本描述（你提供的GPT-5.1描述）
        self.text_descriptions = {
            0: "This MR image of the brain is diagnostic of a cognitively normal individual. The hippocampus appears intact with no visible atrophy. There is no thinning in the cortical regions, and the brain structures are symmetric with well-preserved sulcal and gyral patterns.",
            1: "This MR image of the brain is diagnostic of mild cognitive impairment. Mild atrophy is visible in the hippocampus, particularly in the posterior and medial regions. Subtle thinning is noted in the temporal and parietal cortices. There are no significant white matter changes, but early signs of cortical volume reduction are present.",
            2: "This MR image of the brain is diagnostic of Alzheimer's disease. Significant atrophy is observed in the hippocampus, particularly in the anterior and medial regions. Widespread cortical thinning is seen, especially in the temporal and parietal cortices. Enlarged lateral ventricles and widening of the sulci indicate cerebral atrophy."
        }
        
        # 预编码文本
        self._precompute_text_embeddings()
        
    def _precompute_text_embeddings(self):
        """预计算文本嵌入"""
        text_list = [self.text_descriptions[i] for i in range(3)]
        
        # CLIP编码
        inputs = self.processor(
            text=text_list,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        
        # 存储为buffer
        self.register_buffer('text_embeddings_raw', text_features)
        
    def forward(self, class_labels):
        """
        根据类别标签获取对应的文本嵌入
        
        Args:
            class_labels: [B] - 类别标签 (0, 1, 2)
            
        Returns:
            text_emb: [B, dim_out] - 文本嵌入
        """
        B = class_labels.shape[0]
        
        # 索引预计算的文本嵌入
        text_emb_raw = self.text_embeddings_raw[class_labels]  # [B, clip_dim]
        
        # 投影
        text_emb = self.proj(text_emb_raw)  # [B, dim_out]
        
        return text_emb


class TextGuidedSliceSelector(nn.Module):
    """
    文本引导的切片选择模块
    
    创新点：
    1. 使用文本先验初始化注意力
    2. Top-K硬选择（可解释）
    3. 空间位置编码
    """
    
    def __init__(self, backbone, dim_slice=256, num_slices=64, 
                 num_selected=16, dim_text=256, num_heads=8):
        super().__init__()
        
        self.backbone = backbone
        self.num_slices = num_slices
        self.num_selected = num_selected
        
        # 特征投影
        self.slice_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone.out_dim, dim_slice)
        )
        
        # 位置编码（轴向位置）
        self.pos_encoding = nn.Parameter(torch.randn(num_slices, dim_slice) * 0.02)
        
        # 文本引导的注意力初始化
        self.text_to_attention = nn.Sequential(
            nn.Linear(dim_text, dim_slice),
            nn.ReLU(),
            nn.Linear(dim_slice, num_slices),
            nn.Softmax(dim=-1)
        )
        
        # Transformer编码器（切片间交互）
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_slice,
                nhead=num_heads,
                dim_feedforward=dim_slice * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-LN架构，更稳定
            ),
            num_layers=2
        )
        
        # 注意力评分头
        self.attention_head = nn.Sequential(
            nn.Linear(dim_slice, dim_slice // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim_slice // 2, 1)
        )
        
        # 压缩层（可选）
        self.compress = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_slice,
                nhead=num_heads,
                dim_feedforward=dim_slice * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=1
        )
        
    def forward(self, x, text_prior):
        """
        Args:
            x: [B, S, C, H, W] - 切片堆栈
            text_prior: [B, D_text] - 文本先验嵌入
            
        Returns:
            dict with:
                - features: [B, K, D] - 选中的切片特征
                - attention_scores: [B, S] - 所有切片的注意力分数
                - selected_indices: [B, K] - 选中的切片索引
                - topk_values: [B, K] - 选中切片的注意力分数
        """
        B, S, C, H, W = x.shape
        
        # Stage 1: 提取每个切片的特征
        x = x.view(B * S, C, H, W)
        slice_features = self.backbone(x)  # [B*S, C', H', W']
        slice_features = self.slice_proj(slice_features)  # [B*S, D]
        slice_features = slice_features.view(B, S, -1)  # [B, S, D]
        
        # 加上位置编码
        slice_features = slice_features + self.pos_encoding.unsqueeze(0)
        
        # Stage 2: 文本引导的注意力初始化
        text_attention_prior = self.text_to_attention(text_prior)  # [B, S]
        
        # Stage 3: Transformer编码（切片间交互）
        contextualized = self.transformer(slice_features)  # [B, S, D]
        
        # Stage 4: 计算注意力分数（结合数据驱动和文本先验）
        data_attention_logits = self.attention_head(contextualized).squeeze(-1)  # [B, S]
        
        # 融合文本先验和数据驱动的注意力
        # 使用加权和：70%数据驱动 + 30%文本先验
        combined_logits = data_attention_logits + 0.5 * torch.log(text_attention_prior + 1e-8)
        attention_scores = torch.softmax(combined_logits, dim=-1)  # [B, S]
        
        # Stage 5: Top-K选择
        topk_values, topk_indices = torch.topk(
            attention_scores,
            k=self.num_selected,
            dim=-1
        )  # [B, K]
        
        # 收集选中的切片特征
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, self.num_selected)
        selected_features = contextualized[batch_indices, topk_indices]  # [B, K, D]
        
        # Stage 6: 压缩（可选）
        compressed_features = self.compress(selected_features)  # [B, K, D]
        
        return {
            'features': compressed_features,
            'attention_scores': attention_scores,
            'selected_indices': topk_indices,
            'topk_values': topk_values
        }


class PriorEnhancedMindEncoder(nn.Module):
    """
    先验增强的MIND图编码器
    
    创新点：
    1. 使用组间统计先验调制图边权重
    2. 融合个体MIND矩阵和影像ROI特征
    3. 层次化图注意力
    """
    
    def __init__(self, dim_node=360, dim_hidden=256, num_roi=360, 
                 num_heads=8, num_layers=3):
        super().__init__()
        
        self.num_roi = num_roi
        
        # 将MIND矩阵的每一行作为节点特征
        self.node_proj = nn.Linear(dim_node, dim_hidden)
        
        # 图注意力层
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(
                dim_in=dim_hidden,
                dim_out=dim_hidden,
                num_heads=num_heads,
                dropout=0.1,
                concat=True if i < num_layers - 1 else False
            )
            for i in range(num_layers)
        ])
        
        # 先验调制网络
        self.prior_modulator = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 融合影像ROI特征和MIND特征
        self.roi_fusion = nn.Sequential(
            nn.Linear(dim_hidden * 2, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 全局池化（图级表示）
        self.readout = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim_hidden)
        )
        
    def forward(self, mind_matrix, roi_features_img, group_prior):
        """
        Args:
            mind_matrix: [B, 360, 360] - 个体MIND矩阵
            roi_features_img: [B, 360, D] - 从影像提取的ROI特征
            group_prior: [360, 360] - 组间统计先验（归一化到[0,1]）
            
        Returns:
            dict with:
                - graph_embedding: [B, D] - 图级表示
                - node_features: [B, 360, D] - 节点特征
                - node_attention: [B, 360] - 节点重要性
        """
        B = mind_matrix.shape[0]
        device = mind_matrix.device
        
        # Stage 1: 节点特征初始化（MIND矩阵的每一行）
        node_features_mind = self.node_proj(mind_matrix)  # [B, 360, D]
        
        # Stage 2: 融合影像ROI特征
        node_features = torch.cat([node_features_mind, roi_features_img], dim=-1)
        node_features = self.roi_fusion(node_features)  # [B, 360, D]
        
        # Stage 3: 构建邻接矩阵（基于MIND距离）
        # MIND是距离矩阵，转换为相似度矩阵
        tau = 1.0
        adj_base = torch.exp(-mind_matrix / tau)  # [B, 360, 360]
        
        # Stage 4: 使用组间先验调制邻接矩阵
        # 先验高的边会被增强，先验低的边会被抑制
        prior_modulation = self.prior_modulator(group_prior.unsqueeze(-1)).squeeze(-1)  # [360, 360]
        prior_modulation = prior_modulation.unsqueeze(0).expand(B, -1, -1)  # [B, 360, 360]
        
        adj_modulated = adj_base * (1.0 + prior_modulation)  # [B, 360, 360]
        
        # 归一化（行归一化）
        adj_norm = adj_modulated / (adj_modulated.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Stage 5: 图注意力网络
        h = node_features
        for gat in self.gat_layers:
            h = gat(h, adj_norm)  # [B, 360, D]
        
        # Stage 6: 节点重要性（用于可解释性）
        node_attention = torch.softmax(
            h.norm(dim=-1),  # 使用特征的L2范数作为重要性
            dim=-1
        )  # [B, 360]
        
        # Stage 7: 图级表示（加权求和）
        graph_embedding = torch.einsum('bn,bnd->bd', node_attention, h)  # [B, D]
        graph_embedding = self.readout(graph_embedding)
        
        return {
            'graph_embedding': graph_embedding,
            'node_features': h,
            'node_attention': node_attention
        }


class GraphAttentionLayer(nn.Module):
    """图注意力层（GAT）"""
    
    def __init__(self, dim_in, dim_out, num_heads=8, dropout=0.1, concat=True):
        super().__init__()
        
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.concat = concat
        
        if concat:
            assert dim_out % num_heads == 0
            self.dim_head = dim_out // num_heads
        else:
            self.dim_head = dim_out
        
        # 多头注意力
        self.W = nn.Linear(dim_in, self.dim_head * num_heads, bias=False)
        self.a = nn.Parameter(torch.randn(num_heads, 2 * self.dim_head))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        if not concat:
            self.proj = nn.Linear(self.dim_head * num_heads, dim_out)
        
    def forward(self, h, adj):
        """
        Args:
            h: [B, N, D_in] - 节点特征
            adj: [B, N, N] - 邻接矩阵
            
        Returns:
            h_out: [B, N, D_out] - 更新后的节点特征
        """
        B, N, _ = h.shape
        
        # 线性变换
        h_transformed = self.W(h)  # [B, N, num_heads * dim_head]
        h_transformed = h_transformed.view(B, N, self.num_heads, self.dim_head)
        
        # 计算注意力系数
        # [B, N, num_heads, dim_head] -> [B, N, 1, num_heads, dim_head]
        h_i = h_transformed.unsqueeze(2)
        # [B, N, num_heads, dim_head] -> [B, 1, N, num_heads, dim_head]
        h_j = h_transformed.unsqueeze(1)
        
        # 拼接
        h_concat = torch.cat([
            h_i.expand(-1, -1, N, -1, -1),
            h_j.expand(-1, N, -1, -1, -1)
        ], dim=-1)  # [B, N, N, num_heads, 2*dim_head]
        
        # 注意力分数
        e = torch.einsum('bnmhd,hd->bnmh', h_concat, self.a)  # [B, N, N, num_heads]
        e = self.leaky_relu(e)
        
        # 使用邻接矩阵mask（只考虑有边的节点对）
        mask = (adj.unsqueeze(-1) > 0)  # [B, N, N, 1]
        e = e.masked_fill(~mask, float('-inf'))
        
        # Softmax
        alpha = torch.softmax(e, dim=2)  # [B, N, N, num_heads]
        alpha = self.dropout(alpha)
        
        # 聚合邻居特征
        h_transformed = h_transformed.unsqueeze(1).expand(-1, N, -1, -1, -1)  # [B, N, N, num_heads, dim_head]
        h_aggregated = torch.einsum('bnmh,bnmhd->bnhd', alpha, h_transformed)  # [B, N, num_heads, dim_head]
        
        if self.concat:
            # 拼接多头
            h_out = h_aggregated.reshape(B, N, -1)  # [B, N, num_heads * dim_head]
        else:
            # 平均多头
            h_out = h_aggregated.mean(dim=2)  # [B, N, dim_head]
            h_out = self.proj(h_aggregated.reshape(B, N, -1))
        
        return h_out


class IndividualPriorEncoder(nn.Module):
    """
    个体先验编码器
    
    融合个体MIND矩阵和人口学信息
    """
    
    def __init__(self, dim_mind=360, dim_demo=13, dim_out=128):
        super().__init__()
        
        # MIND矩阵编码（使用CNN或MLP）
        self.mind_encoder = nn.Sequential(
            # 将360x360矩阵视为图像
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 180x180
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 90x90
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, dim_out)
        )
        
        # 人口学编码
        self.demo_encoder = nn.Sequential(
            nn.Linear(dim_demo, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, dim_out)
        )
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(dim_out * 2, dim_out),
            nn.LayerNorm(dim_out),
            nn.GELU()
        )
        
    def forward(self, mind_matrix, demographics):
        """
        Args:
            mind_matrix: [B, 360, 360]
            demographics: [B, D_demo]
            
        Returns:
            individual_features: [B, dim_out]
        """
        # 编码MIND
        mind_matrix = mind_matrix.unsqueeze(1)  # [B, 1, 360, 360]
        mind_features = self.mind_encoder(mind_matrix)  # [B, dim_out]
        
        # 编码人口学
        demo_features = self.demo_encoder(demographics)  # [B, dim_out]
        
        # 融合
        individual_features = torch.cat([mind_features, demo_features], dim=-1)
        individual_features = self.fusion(individual_features)
        
        return individual_features


class SubcortexEncoder(nn.Module):
    """皮下ROI编码器"""
    
    def __init__(self, dim_slice=256, num_subcortex=66, dim_out=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(dim_slice, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, dim_out)
        )
        
        # 池化
        self.pool = nn.Sequential(
            nn.Linear(num_subcortex * dim_out, 256),
            nn.GELU(),
            nn.Linear(256, dim_out)
        )
        
    def forward(self, roi_features_sub):
        """
        Args:
            roi_features_sub: [B, 66, D]
            
        Returns:
            subcortex_features: [B, dim_out]
        """
        # 编码每个皮下ROI
        encoded = self.encoder(roi_features_sub)  # [B, 66, dim_out]
        
        # 池化
        encoded_flat = encoded.flatten(1)  # [B, 66 * dim_out]
        subcortex_features = self.pool(encoded_flat)
        
        return subcortex_features


class CrossModalAttentionFusion(nn.Module):
    """
    跨模态注意力融合
    
    创新点：
    1. 多模态相互查询（Cross-Attention）
    2. 门控机制（动态调整模态权重）
    3. 残差连接
    """
    
    def __init__(self, dim_img=256, dim_graph=256, dim_subcortex=64, 
                 dim_individual=128, dim_out=512, num_heads=8):
        super().__init__()
        
        # 统一维度
        self.proj_img = nn.Linear(dim_img, 256)
        self.proj_graph = nn.Linear(dim_graph, 256)
        self.proj_subcortex = nn.Linear(dim_subcortex, 256)
        self.proj_individual = nn.Linear(dim_individual, 256)
        
        # 跨模态注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(256 * 4, 256),
            nn.GELU(),
            nn.Linear(256, 4),
            nn.Softmax(dim=-1)
        )
        
        # 融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, dim_out)
        )
        
    def forward(self, z_img, z_graph, z_subcortex, z_individual):
        """
        Args:
            z_img: [B, D_img]
            z_graph: [B, D_graph]
            z_subcortex: [B, D_subcortex]
            z_individual: [B, D_individual]
            
        Returns:
            dict with:
                - fused_features: [B, dim_out]
                - modal_weights: [B, 4] - 每个模态的权重
        """
        B = z_img.shape[0]
        
        # 投影到统一维度
        z_img = self.proj_img(z_img).unsqueeze(1)  # [B, 1, 256]
        z_graph = self.proj_graph(z_graph).unsqueeze(1)
        z_subcortex = self.proj_subcortex(z_subcortex).unsqueeze(1)
        z_individual = self.proj_individual(z_individual).unsqueeze(1)
        
        # 堆叠所有模态
        z_stack = torch.cat([z_img, z_graph, z_subcortex, z_individual], dim=1)  # [B, 4, 256]
        
        # 自注意力（模态间交互）
        z_attended, _ = self.cross_attn(z_stack, z_stack, z_stack)  # [B, 4, 256]
        
        # 残差连接
        z_attended = z_attended + z_stack
        
        # 门控加权
        z_concat = z_attended.flatten(1)  # [B, 4 * 256]
        modal_weights = self.gate_net(z_concat)  # [B, 4]
        
        # 加权求和
        z_weighted = torch.einsum('bm,bmd->bd', modal_weights, z_attended)  # [B, 256]
        
        # 最终融合
        fused_features = self.fusion_mlp(z_weighted)  # [B, dim_out]
        
        return {
            'fused_features': fused_features,
            'modal_weights': modal_weights
        }


class MultiTaskPredictionHead(nn.Module):
    """
    多任务预测头 - 支持六个任务
    
    主任务：
    1. 三分类 (NC/MCI/AD)
    
    辅助任务（二分类）：
    2. NC vs AD
    3. NC vs MCI  
    4. MCI vs AD
    5. AD vs 非AD
    6. MCI vs 非MCI
    """
    
    def __init__(self, dim_input=512, num_classes=3):
        super().__init__()
        
        # 共享特征提取
        self.shared_encoder = nn.Sequential(
            nn.Linear(dim_input, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 主任务：三分类
        self.head_3way = nn.Linear(256, num_classes)
        
        # 辅助任务1：NC vs AD
        self.head_nc_ad = nn.Linear(256, 2)
        
        # 辅助任务2：NC vs MCI
        self.head_nc_mci = nn.Linear(256, 2)
        
        # 辅助任务3：MCI vs AD
        self.head_mci_ad = nn.Linear(256, 2)
        
        # 辅助任务4：AD vs 非AD
        self.head_ad_vs_non = nn.Linear(256, 2)
        
        # 辅助任务5：MCI vs 非MCI
        self.head_mci_vs_non = nn.Linear(256, 2)
        
    def forward(self, h):
        """
        Args:
            h: [B, dim_input] - 融合特征
            
        Returns:
            dict with logits for all tasks
        """
        # 共享编码
        features = self.shared_encoder(h)  # [B, 256]
        
        return {
            'logits_3way': self.head_3way(features),
            'logits_nc_ad': self.head_nc_ad(features),
            'logits_nc_mci': self.head_nc_mci(features),
            'logits_mci_ad': self.head_mci_ad(features),
            'logits_ad_vs_non': self.head_ad_vs_non(features),
            'logits_mci_vs_non': self.head_mci_vs_non(features),
        }


# model.py - 修改损失函数计算
def compute_loss(self, outputs, labels, cfg):
    """
    计算多任务损失
    
    Args:
        outputs: 模型输出字典
        labels: [B] - 真实标签 (0=NC, 1=MCI, 2=AD)
        cfg: 配置字典
    """
    B = labels.shape[0]
    device = labels.device
    
    # 1. 主任务：三分类 (Focal Loss)
    loss_3way = self.focal_loss(outputs['logits_3way'], labels)
    
    # 2. NC vs AD (只用NC和AD样本)
    mask_nc_ad = (labels == 0) | (labels == 2)
    if mask_nc_ad.sum() > 0:
        labels_nc_ad = (labels[mask_nc_ad] == 2).long()  # NC=0, AD=1
        logits_nc_ad = outputs['logits_nc_ad'][mask_nc_ad]
        loss_nc_ad = F.cross_entropy(logits_nc_ad, labels_nc_ad)
    else:
        loss_nc_ad = torch.tensor(0.0, device=device)
    
    # 3. NC vs MCI (只用NC和MCI样本)
    mask_nc_mci = (labels == 0) | (labels == 1)
    if mask_nc_mci.sum() > 0:
        labels_nc_mci = (labels[mask_nc_mci] == 1).long()  # NC=0, MCI=1
        logits_nc_mci = outputs['logits_nc_mci'][mask_nc_mci]
        loss_nc_mci = F.cross_entropy(logits_nc_mci, labels_nc_mci)
    else:
        loss_nc_mci = torch.tensor(0.0, device=device)
    
    # 4. MCI vs AD (只用MCI和AD样本)
    mask_mci_ad = (labels == 1) | (labels == 2)
    if mask_mci_ad.sum() > 0:
        labels_mci_ad = (labels[mask_mci_ad] == 2).long()  # MCI=0, AD=1
        logits_mci_ad = outputs['logits_mci_ad'][mask_mci_ad]
        loss_mci_ad = F.cross_entropy(logits_mci_ad, labels_mci_ad)
    else:
        loss_mci_ad = torch.tensor(0.0, device=device)
    
    # 5. AD vs 非AD (所有样本)
    labels_ad = (labels == 2).long()
    loss_ad_vs_non = F.cross_entropy(outputs['logits_ad_vs_non'], labels_ad)
    
    # 6. MCI vs 非MCI (所有样本)
    labels_mci = (labels == 1).long()
    loss_mci_vs_non = F.cross_entropy(outputs['logits_mci_vs_non'], labels_mci)
    
    # 7. 文本对齐损失
    if 'text_embeddings' in outputs and 'z_img' in outputs:
        loss_text = self.compute_text_alignment_loss(
            outputs['text_embeddings'],
            outputs['z_img'],
            labels
        )
    else:
        loss_text = torch.tensor(0.0, device=device)
    
    # 8. 切片选择稀疏性损失
    if 'selected_indices' in outputs:
        loss_sparse = self.compute_sparsity_loss(outputs['selected_indices'])
    else:
        loss_sparse = torch.tensor(0.0, device=device)
    
    # 9. 不确定性加权的总损失
    # 使用可学习的log(σ²)参数自动平衡各任务
    precision_3way = torch.exp(-self.log_vars[0])
    precision_nc_ad = torch.exp(-self.log_vars[1])
    precision_nc_mci = torch.exp(-self.log_vars[2])
    precision_mci_ad = torch.exp(-self.log_vars[3])
    precision_ad = torch.exp(-self.log_vars[4])
    precision_mci = torch.exp(-self.log_vars[5])
    precision_text = torch.exp(-self.log_vars[6])
    precision_sparse = torch.exp(-self.log_vars[7])
    
    loss_total = (
        precision_3way * loss_3way + self.log_vars[0] +
        precision_nc_ad * loss_nc_ad + self.log_vars[1] +
        precision_nc_mci * loss_nc_mci + self.log_vars[2] +
        precision_mci_ad * loss_mci_ad + self.log_vars[3] +
        precision_ad * loss_ad_vs_non + self.log_vars[4] +
        precision_mci * loss_mci_vs_non + self.log_vars[5] +
        precision_text * loss_text + self.log_vars[6] +
        precision_sparse * loss_sparse + self.log_vars[7]
    )
    
    return {
        'loss_total': loss_total,
        'loss_3way': loss_3way,
        'loss_nc_ad': loss_nc_ad,
        'loss_nc_mci': loss_nc_mci,
        'loss_mci_ad': loss_mci_ad,
        'loss_ad_vs_non': loss_ad_vs_non,
        'loss_mci_vs_non': loss_mci_vs_non,
        'loss_text': loss_text,
        'loss_sparse': loss_sparse
    }