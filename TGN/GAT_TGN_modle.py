#!/usr/bin/env python

"""
基于GAT的时序图神经网络(TGN)模型
GAT-based Temporal Graph Networks for Link Quality Prediction

主要组件：
1. HeterogeneousGAT: 异构图注意力网络
2. TemporalEncoder: 时序编码器  
3. Memory Module: 记忆更新模块
4. LinkPredictor: 链路质量预测器
5. MultiHorizonPredictor: 多步预测器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse, to_undirected
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class HeterogeneousGAT(nn.Module):
    """异构图注意力网络"""
    
    def __init__(self, 
                 node_feature_dims: Dict[str, int],
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super(HeterogeneousGAT, self).__init__()
        
        self.node_types = list(node_feature_dims.keys())
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 节点类型嵌入
        self.node_type_embedding = nn.Embedding(len(self.node_types), hidden_dim)
        
        # 异构节点特征投影
        self.node_projections = nn.ModuleDict()
        for node_type, feature_dim in node_feature_dims.items():
            self.node_projections[node_type] = nn.Linear(feature_dim, hidden_dim)
        
        # 多层GAT
        self.gat_layers = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = hidden_dim if layer == 0 else hidden_dim * num_heads
            output_dim = hidden_dim
            
            self.gat_layers.append(
                GATConv(
                    input_dim, 
                    output_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=(layer < num_layers - 1)  # 最后一层不concat
                )
            )
        
        # 层标准化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * num_heads if i < num_layers - 1 else hidden_dim)
            for i in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                node_features: torch.Tensor,
                node_types: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_features: [N, F] 节点特征
            node_types: [N] 节点类型索引
            edge_index: [2, E] 边索引
            edge_attr: [E, edge_dim] 边特征
            batch: [N] 批次索引
        
        Returns:
            node_embeddings: [N, hidden_dim] 节点嵌入
        """
        
        # 类型嵌入
        type_embeddings = self.node_type_embedding(node_types)
        
        # 特征投影（这里简化处理，实际应该根据node_types分别投影）
        x = self.node_projections['default'](node_features) + type_embeddings
        
        # 多层GAT处理
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            residual = x if i > 0 else None
            
            # GAT卷积
            x = gat_layer(x, edge_index)
            x = layer_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # 残差连接
            if residual is not None and residual.shape == x.shape:
                x = x + residual
        
        return x

class TemporalEncoder(nn.Module):
    """时序编码器"""
    
    def __init__(self, 
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super(TemporalEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # LSTM用于时序建模
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                temporal_embeddings: torch.Tensor,
                temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        时序编码
        
        Args:
            temporal_embeddings: [B, T, hidden_dim] 时序嵌入
            temporal_mask: [B, T] 时序mask
        
        Returns:
            encoded_sequence: [B, T, hidden_dim] 编码后的时序
        """
        
        # 位置编码
        x = self.pos_encoding(temporal_embeddings.transpose(0, 1)).transpose(0, 1)
        
        # LSTM编码
        lstm_out, _ = self.lstm(x)
        
        # 自注意力
        attn_out, _ = self.self_attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=temporal_mask
        )
        
        # 残差连接和层标准化
        x = self.layer_norm1(lstm_out + self.dropout(attn_out))
        
        return x

class MemoryModule(nn.Module):
    """记忆模块"""
    
    def __init__(self, 
                 memory_dim: int = 64,
                 num_nodes: int = 100,
                 update_method: str = 'gru'):
        super(MemoryModule, self).__init__()
        
        self.memory_dim = memory_dim
        self.num_nodes = num_nodes
        self.update_method = update_method
        
        # 初始化节点记忆
        self.register_buffer('node_memory', torch.zeros(num_nodes, memory_dim))
        self.register_buffer('last_update_time', torch.zeros(num_nodes))
        
        # 记忆更新网络
        if update_method == 'gru':
            self.memory_updater = nn.GRUCell(memory_dim, memory_dim)
        elif update_method == 'lstm':
            self.memory_updater = nn.LSTMCell(memory_dim, memory_dim)
        else:
            self.memory_updater = nn.Linear(memory_dim * 2, memory_dim)
        
        # 时间编码
        self.time_encoder = nn.Linear(1, memory_dim)
        
    def forward(self, 
                node_ids: torch.Tensor,
                node_embeddings: torch.Tensor,
                timestamps: torch.Tensor) -> torch.Tensor:
        """
        记忆更新和检索
        
        Args:
            node_ids: [N] 节点ID
            node_embeddings: [N, memory_dim] 当前节点嵌入
            timestamps: [N] 时间戳
        
        Returns:
            updated_memory: [N, memory_dim] 更新后的记忆
        """
        
        # 获取当前记忆
        current_memory = self.node_memory[node_ids]
        
        # 时间编码
        time_deltas = timestamps.unsqueeze(-1) - self.last_update_time[node_ids].unsqueeze(-1)
        time_encoding = self.time_encoder(time_deltas)
        
        # 记忆更新
        if self.update_method == 'gru':
            updated_memory = self.memory_updater(
                node_embeddings + time_encoding, 
                current_memory
            )
        else:
            combined = torch.cat([current_memory, node_embeddings + time_encoding], dim=-1)
            updated_memory = torch.tanh(self.memory_updater(combined))
        
        # 更新全局记忆
        self.node_memory[node_ids] = updated_memory.detach()
        self.last_update_time[node_ids] = timestamps.detach()
        
        return updated_memory
    
    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """获取节点记忆"""
        return self.node_memory[node_ids]

class LinkPredictor(nn.Module):
    """链路质量预测器"""
    
    def __init__(self, 
                 node_dim: int = 64,
                 edge_dim: int = 32,
                 hidden_dim: int = 128,
                 output_dim: int = 5):  # RSSI, SNR, Bandwidth, Latency, Packet_loss
        super(LinkPredictor, self).__init__()
        
        self.output_dim = output_dim
        
        # 边特征编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # 节点对特征融合
        self.node_fusion = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 链路质量预测头
        self.link_quality_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 二分类头（链路存在性）
        self.link_existence_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                src_embeddings: torch.Tensor,
                dst_embeddings: torch.Tensor,
                edge_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        链路质量预测
        
        Args:
            src_embeddings: [E, node_dim] 源节点嵌入
            dst_embeddings: [E, node_dim] 目标节点嵌入  
            edge_features: [E, edge_dim] 边特征
        
        Returns:
            predictions: 包含质量预测和存在性预测的字典
        """
        
        # 边特征编码
        edge_encoded = self.edge_encoder(edge_features)
        
        # 节点对特征融合
        node_pairs = torch.cat([src_embeddings, dst_embeddings], dim=-1)
        node_fused = self.node_fusion(node_pairs)
        
        # 组合特征
        combined_features = torch.cat([node_fused, edge_encoded], dim=-1)
        
        # 预测
        link_quality = self.link_quality_predictor(combined_features)
        link_existence = self.link_existence_predictor(combined_features)
        
        return {
            'link_quality': link_quality,
            'link_existence': link_existence
        }

class MultiHorizonPredictor(nn.Module):
    """多步预测器"""
    
    def __init__(self, 
                 input_dim: int = 64,
                 hidden_dim: int = 128,
                 prediction_horizons: List[int] = [1, 3, 5, 10],
                 output_features: int = 5):
        super(MultiHorizonPredictor, self).__init__()
        
        self.prediction_horizons = prediction_horizons
        self.output_features = output_features
        
        # 为每个预测步长创建专门的预测头
        self.horizon_predictors = nn.ModuleDict()
        
        for horizon in prediction_horizons:
            self.horizon_predictors[f'horizon_{horizon}'] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_features)
            )
        
        # 共享的特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        多步预测
        
        Args:
            features: [B, input_dim] 输入特征
        
        Returns:
            predictions: 每个步长的预测结果
        """
        
        # 共享特征提取
        shared_features = self.feature_extractor(features)
        
        predictions = {}
        for horizon in self.prediction_horizons:
            predictor = self.horizon_predictors[f'horizon_{horizon}']
            predictions[f'horizon_{horizon}'] = predictor(shared_features)
        
        return predictions

class TGN_GAT(nn.Module):
    """基于GAT的时序图神经网络主模型"""
    
    def __init__(self, 
                 node_feature_dim: int = 10,
                 edge_feature_dim: int = 7,
                 hidden_dim: int = 64,
                 memory_dim: int = 64,
                 num_gat_heads: int = 4,
                 num_gat_layers: int = 2,
                 num_temporal_layers: int = 2,
                 prediction_horizons: List[int] = [1, 3, 5, 10],
                 max_nodes: int = 100,
                 dropout: float = 0.1):
        super(TGN_GAT, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.prediction_horizons = prediction_horizons
        
        # 节点类型定义
        node_feature_dims = {
            'vehicle': node_feature_dim,
            'uav': node_feature_dim,
            'base_station': node_feature_dim,
            'default': node_feature_dim
        }
        
        # 核心组件
        self.gat_encoder = HeterogeneousGAT(
            node_feature_dims=node_feature_dims,
            hidden_dim=hidden_dim,
            num_heads=num_gat_heads,
            num_layers=num_gat_layers,
            dropout=dropout
        )
        
        self.temporal_encoder = TemporalEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_temporal_layers,
            dropout=dropout
        )
        
        self.memory_module = MemoryModule(
            memory_dim=memory_dim,
            num_nodes=max_nodes,
            update_method='gru'
        )
        
        self.link_predictor = LinkPredictor(
            node_dim=hidden_dim + memory_dim,
            edge_dim=edge_feature_dim,
            hidden_dim=hidden_dim * 2,
            output_dim=5
        )
        
        self.multi_horizon_predictor = MultiHorizonPredictor(
            input_dim=hidden_dim + memory_dim,
            hidden_dim=hidden_dim * 2,
            prediction_horizons=prediction_horizons,
            output_features=5
        )
        
        # 图级别的聚合
        self.graph_aggregator = nn.Sequential(
            nn.Linear(hidden_dim + memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
    def forward(self, 
                batch_data: Dict,
                predict_horizons: bool = True) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            batch_data: 批次数据
            predict_horizons: 是否进行多步预测
        
        Returns:
            predictions: 预测结果字典
        """
        
        # 提取批次数据
        node_features = batch_data['node_features']  # [B*T*N, F]
        edge_features = batch_data['edge_features']  # [E, edge_F]
        edge_index = batch_data['edge_index']        # [2, E]
        node_types = batch_data.get('node_types', torch.zeros(node_features.shape[0], dtype=torch.long))
        node_ids = batch_data.get('node_ids', torch.arange(node_features.shape[0]))
        timestamps = batch_data.get('timestamps', torch.zeros(node_features.shape[0]))
        batch_info = batch_data.get('batch_info', {})
        
        # GAT编码
        node_embeddings = self.gat_encoder(
            node_features=node_features,
            node_types=node_types,
            edge_index=edge_index,
            edge_attr=edge_features
        )
        
        # 记忆更新
        memory_embeddings = self.memory_module(
            node_ids=node_ids,
            node_embeddings=node_embeddings,
            timestamps=timestamps
        )
        
        # 融合节点嵌入和记忆
        enhanced_node_embeddings = torch.cat([node_embeddings, memory_embeddings], dim=-1)
        
        # 时序编码（如果有时序信息）
        if 'sequence_length' in batch_info:
            seq_len = batch_info['sequence_length']
            batch_size = batch_info['batch_size']
            
            # 重塑为时序格式
            temporal_embeddings = enhanced_node_embeddings.view(batch_size, seq_len, -1, enhanced_node_embeddings.shape[-1])
            temporal_embeddings = temporal_embeddings.mean(dim=2)  # 平均池化节点维度
            
            # 时序编码
            temporal_encoded = self.temporal_encoder(temporal_embeddings)
            
            # 取最后一个时间步的嵌入
            final_embeddings = temporal_encoded[:, -1, :]
        else:
            # 图级别聚合
            final_embeddings = global_mean_pool(enhanced_node_embeddings, 
                                               batch_info.get('batch_indices', torch.zeros(enhanced_node_embeddings.shape[0], dtype=torch.long)))
        
        predictions = {}
        
        # 链路质量预测
        if edge_index.shape[1] > 0:
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]
            
            src_embeddings = enhanced_node_embeddings[src_nodes]
            dst_embeddings = enhanced_node_embeddings[dst_nodes]
            
            link_predictions = self.link_predictor(
                src_embeddings=src_embeddings,
                dst_embeddings=dst_embeddings,
                edge_features=edge_features
            )
            
            predictions.update(link_predictions)
        
        # 多步预测
        if predict_horizons:
            # 使用图级别嵌入进行多步预测
            if final_embeddings.dim() == 1:
                final_embeddings = final_embeddings.unsqueeze(0)
            
            horizon_predictions = self.multi_horizon_predictor(final_embeddings)
            predictions.update(horizon_predictions)
        
        return predictions
    
    def predict_link_quality(self, 
                           src_node_id: int,
                           dst_node_id: int,
                           current_features: torch.Tensor) -> torch.Tensor:
        """
        预测特定链路的质量
        
        Args:
            src_node_id: 源节点ID
            dst_node_id: 目标节点ID
            current_features: 当前特征
        
        Returns:
            predicted_quality: 预测的链路质量
        """
        
        with torch.no_grad():
            # 获取节点记忆
            src_memory = self.memory_module.get_memory(torch.tensor([src_node_id]))
            dst_memory = self.memory_module.get_memory(torch.tensor([dst_node_id]))
            
            # 简化的预测（实际使用时需要完整的图结构）
            combined_memory = torch.cat([src_memory, dst_memory], dim=-1)
            
            # 使用链路预测器
            dummy_edge_features = torch.zeros(1, 7)  # 需要实际的边特征
            predictions = self.link_predictor(
                src_embeddings=src_memory,
                dst_embeddings=dst_memory,
                edge_features=dummy_edge_features
            )
            
            return predictions['link_quality']

def create_model_config() -> Dict:
    """创建模型配置"""
    return {
        'node_feature_dim': 10,      # 节点特征维度
        'edge_feature_dim': 7,       # 边特征维度
        'hidden_dim': 64,            # 隐藏层维度
        'memory_dim': 64,            # 记忆维度
        'num_gat_heads': 4,          # GAT注意力头数
        'num_gat_layers': 2,         # GAT层数
        'num_temporal_layers': 2,    # 时序编码器层数
        'prediction_horizons': [1, 3, 5, 10],  # 预测步长
        'max_nodes': 100,            # 最大节点数
        'dropout': 0.1,              # Dropout率
        'learning_rate': 0.001,      # 学习率
        'weight_decay': 1e-5,        # 权重衰减
        'batch_size': 32,            # 批次大小
        'num_epochs': 100,           # 训练轮数
        'early_stopping_patience': 10,  # 早停耐心值
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

# 示例使用
if __name__ == '__main__':
    # 创建模型
    config = create_model_config()
    model = TGN_GAT(**{k: v for k, v in config.items() if k in TGN_GAT.__init__.__code__.co_varnames})
    
    print("TGN-GAT Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 创建示例数据
    batch_data = {
        'node_features': torch.randn(50, 10),   # 50个节点，10维特征
        'edge_features': torch.randn(100, 7),   # 100条边，7维特征
        'edge_index': torch.randint(0, 50, (2, 100)),  # 边索引
        'node_types': torch.randint(0, 3, (50,)),       # 节点类型
        'node_ids': torch.arange(50),                   # 节点ID
        'timestamps': torch.randn(50),                  # 时间戳
        'batch_info': {
            'batch_size': 1,
            'sequence_length': 1
        }
    }
    
    # 前向传播测试
    with torch.no_grad():
        predictions = model(batch_data)
        print("Forward pass successful!")
        for key, value in predictions.items():
            print(f"{key}: {value.shape}")