#!/usr/bin/env python

"""
TGN模型训练脚本
Training Script for TGN Model

功能：
1. 数据加载和批处理
2. 多任务损失函数
3. 训练和验证循环
4. 模型保存和加载
5. 性能评估和可视化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
import pickle
import json
import os
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import wandb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入之前定义的模型
from tgn_gat_model import TGN_GAT, create_model_config

class NetworkDataset(Dataset):
    """网络数据集"""
    
    def __init__(self, 
                 data_file: str,
                 scalers_file: str,
                 transform: Optional[callable] = None):
        
        # 加载数据
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        # 加载特征缩放器
        with open(scalers_file, 'rb') as f:
            self.scalers = pickle.load(f)
        
        self.transform = transform
        print(f"Loaded {len(self.data)} samples from {data_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 转换为PyTorch Geometric数据格式
        graph_data = self._convert_to_pyg_data(sample)
        
        if self.transform:
            graph_data = self.transform(graph_data)
        
        return graph_data
    
    def _convert_to_pyg_data(self, sample: Dict) -> Dict:
        """转换为PyTorch Geometric数据格式"""
        
        sequence = sample['input_sequence']
        targets = sample['targets']
        
        # 处理最后一个时间步的图数据
        last_snapshot = sequence[-1]
        
        # 节点特征
        nodes = last_snapshot['nodes']
        node_ids = list(nodes.keys())
        node_id_map = {node_id: i for i, node_id in enumerate(node_ids)}
        
        node_features = []
        node_types = []
        node_positions = []
        
        for node_id in node_ids:
            node_data = nodes[node_id]
            
            # 使用标准化后的特征
            if 'normalized_features' in node_data:
                features = node_data['normalized_features']
            else:
                # 回退到原始特征处理
                features = self._extract_node_features(node_data)
            
            node_features.append(features)
            
            # 节点类型编码
            type_map = {'vehicle': 0, 'uav': 1, 'base_station': 2}
            node_types.append(type_map.get(node_data['type'], 0))
            
            # 位置信息
            node_positions.append(node_data['position'][:3])
        
        # 边特征和索引
        edges = last_snapshot['edges']
        edge_index = []
        edge_features = []
        edge_targets = []
        
        for edge in edges:
            src_id = edge['src']
            dst_id = edge['dst']
            
            if src_id in node_id_map and dst_id in node_id_map:
                src_idx = node_id_map[src_id]
                dst_idx = node_id_map[dst_id]
                
                edge_index.append([src_idx, dst_idx])
                
                # 边特征
                if 'normalized_features' in edge:
                    edge_feat = edge['normalized_features']
                else:
                    edge_feat = self._extract_edge_features(edge)
                
                edge_features.append(edge_feat)
                
                # 边的预测目标
                link_id = f"{src_id}-{dst_id}"
                if 'link_qualities' in targets and link_id in targets['link_qualities']:
                    target_quality = targets['link_qualities'][link_id]
                    edge_target = [
                        target_quality.get('rssi', -70),
                        target_quality.get('snr', 10),
                        target_quality.get('bandwidth', 20),
                        target_quality.get('latency', 10),
                        target_quality.get('packet_loss', 0.01)
                    ]
                    edge_targets.append(edge_target)
                else:
                    # 使用当前值作为目标（自监督）
                    edge_targets.append(edge_feat[:5])
        
        # 转换为张量
        node_features = torch.tensor(node_features, dtype=torch.float32)
        node_types = torch.tensor(node_types, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
        edge_features = torch.tensor(edge_features, dtype=torch.float32) if edge_features else torch.empty((0, 7), dtype=torch.float32)
        edge_targets = torch.tensor(edge_targets, dtype=torch.float32) if edge_targets else torch.empty((0, 5), dtype=torch.float32)
        
        # 多步预测目标
        horizon_targets = {}
        for horizon_key, horizon_data in targets.items():
            if horizon_key.startswith('horizon_'):
                # 简化处理：使用网络级别的平均指标作为目标
                if 'link_qualities' in horizon_data and horizon_data['link_qualities']:
                    qualities = list(horizon_data['link_qualities'].values())
                    avg_quality = [
                        np.mean([q.get('rssi', -70) for q in qualities]),
                        np.mean([q.get('snr', 10) for q in qualities]),
                        np.mean([q.get('bandwidth', 20) for q in qualities]),
                        np.mean([q.get('latency', 10) for q in qualities]),
                        np.mean([q.get('packet_loss', 0.01) for q in qualities])
                    ]
                    horizon_targets[horizon_key] = torch.tensor(avg_quality, dtype=torch.float32)
        
        return {
            'node_features': node_features,
            'node_types': node_types,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'edge_targets': edge_targets,
            'horizon_targets': horizon_targets,
            'node_ids': torch.arange(len(node_ids)),
            'timestamps': torch.tensor([last_snapshot['timestamp']] * len(node_ids)),
            'batch_info': {
                'batch_size': 1,
                'sequence_length': 1,
                'num_nodes': len(node_ids),
                'num_edges': len(edge_index[0]) if len(edge_index) > 0 else 0
            }
        }
    
    def _extract_node_features(self, node_data: Dict) -> List[float]:
        """提取节点特征（回退方法）"""
        features = [
            node_data['position'][0],
            node_data['position'][1], 
            node_data['position'][2],
            node_data.get('velocity', 0),
            node_data.get('energy', 100),
            node_data.get('load', 0),
            node_data.get('connectivity', 0),
            {'vehicle': 0, 'uav': 1, 'base_station': 2}.get(node_data['type'], 0),
            node_data.get('battery', 100) if node_data['type'] == 'uav' else node_data.get('speed', 0),
            node_data.get('altitude', 0) if node_data['type'] == 'uav' else node_data.get('direction', 0)
        ]
        return features
    
    def _extract_edge_features(self, edge_data: Dict) -> List[float]:
        """提取边特征（回退方法）"""
        return [
            edge_data.get('rssi', -70),
            edge_data.get('snr', 10),
            edge_data.get('bandwidth', 20),
            edge_data.get('latency', 10),
            edge_data.get('packet_loss', 0.01),
            edge_data.get('distance', 100),
            edge_data.get('los_probability', 0.8)
        ]

def collate_fn(batch: List[Dict]) -> Dict:
    """批处理函数"""
    
    # 合并所有样本的数据
    batch_node_features = []
    batch_edge_features = []
    batch_edge_indices = []
    batch_edge_targets = []
    batch_node_types = []
    batch_node_ids = []
    batch_timestamps = []
    batch_indices = []
    
    horizon_targets_batch = {f'horizon_{h}': [] for h in [1, 3, 5, 10]}
    
    node_offset = 0
    
    for i, sample in enumerate(batch):
        # 节点数据
        batch_node_features.append(sample['node_features'])
        batch_node_types.append(sample['node_types'])
        batch_node_ids.append(sample['node_ids'] + node_offset)
        batch_timestamps.append(sample['timestamps'])
        
        # 创建批次索引
        num_nodes = sample['node_features'].shape[0]
        batch_indices.extend([i] * num_nodes)
        
        # 边数据
        if sample['edge_index'].shape[1] > 0:
            # 调整边索引
            adjusted_edge_index = sample['edge_index'] + node_offset
            batch_edge_indices.append(adjusted_edge_index)
            batch_edge_features.append(sample['edge_features'])
            batch_edge_targets.append(sample['edge_targets'])
        
        # 多步预测目标
        for horizon_key in horizon_targets_batch.keys():
            if horizon_key in sample['horizon_targets']:
                horizon_targets_batch[horizon_key].append(sample['horizon_targets'][horizon_key])
            else:
                # 填充默认值
                horizon_targets_batch[horizon_key].append(torch.zeros(5))
        
        node_offset += num_nodes
    
    # 拼接数据
    batched_data = {
        'node_features': torch.cat(batch_node_features, dim=0),
        'node_types': torch.cat(batch_node_types, dim=0),
        'node_ids': torch.cat(batch_node_ids, dim=0),
        'timestamps': torch.cat(batch_timestamps, dim=0),
        'batch_info': {
            'batch_size': len(batch),
            'sequence_length': 1,
            'batch_indices': torch.tensor(batch_indices, dtype=torch.long)
        }
    }
    
    # 处理边数据
    if batch_edge_indices:
        batched_data['edge_index'] = torch.cat(batch_edge_indices, dim=1)
        batched_data['edge_features'] = torch.cat(batch_edge_features, dim=0)
        batched_data['edge_targets'] = torch.cat(batch_edge_targets, dim=0)
    else:
        batched_data['edge_index'] = torch.empty((2, 0), dtype=torch.long)
        batched_data['edge_features'] = torch.empty((0, 7), dtype=torch.float32)
        batched_data['edge_targets'] = torch.empty((0, 5), dtype=torch.float32)
    
    # 处理多步预测目标
    for horizon_key, targets in horizon_targets_batch.items():
        if targets:
            batched_data[f'{horizon_key}_targets'] = torch.stack(targets, dim=0)
        else:
            batched_data[f'{horizon_key}_targets'] = torch.empty((len(batch), 5))
    
    return batched_data

class MultiTaskLoss(nn.Module):
    """多任务损失函数"""
    
    def __init__(self, 
                 link_quality_weight: float = 1.0,
                 link_existence_weight: float = 0.5,
                 horizon_weights: Dict[str, float] = None):
        super(MultiTaskLoss, self).__init__()
        
        self.link_quality_weight = link_quality_weight
        self.link_existence_weight = link_existence_weight
        
        if horizon_weights is None:
            self.horizon_weights = {
                'horizon_1': 1.0,
                'horizon_3': 0.8,
                'horizon_5': 0.6,
                'horizon_10': 0.4
            }
        else:
            self.horizon_weights = horizon_weights
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, predictions: Dict, targets: Dict) -> Dict:
        """计算多任务损失"""
        
        losses = {}
        total_loss = 0.0
        
        # 链路质量损失
        if 'link_quality' in predictions and 'edge_targets' in targets:
            edge_targets = targets['edge_targets']
            if edge_targets.shape[0] > 0:
                link_quality_pred = predictions['link_quality']
                
                quality_loss = self.mse_loss(link_quality_pred, edge_targets)
                losses['link_quality_loss'] = quality_loss
                total_loss += self.link_quality_weight * quality_loss
        
        # 链路存在性损失
        if 'link_existence' in predictions and 'edge_targets' in targets:
            edge_targets = targets['edge_targets']
            if edge_targets.shape[0] > 0:
                # 基于RSSI阈值创建存在性标签
                existence_labels = (edge_targets[:, 0] > -80).float().unsqueeze(1)
                existence_pred = predictions['link_existence']
                
                existence_loss = self.bce_loss(existence_pred, existence_labels)
                losses['link_existence_loss'] = existence_loss
                total_loss += self.link_existence_weight * existence_loss
        
        # 多步预测损失
        for horizon in ['horizon_1', 'horizon_3', 'horizon_5', 'horizon_10']:
            if horizon in predictions and f'{horizon}_targets' in targets:
                horizon_pred = predictions[horizon]
                horizon_target = targets[f'{horizon}_targets']
                
                horizon_loss = self.mse_loss(horizon_pred, horizon_target)
                losses[f'{horizon}_loss'] = horizon_loss
                total_loss += self.horizon_weights[horizon] * horizon_loss
        
        losses['total_loss'] = total_loss
        return losses

class TGNTrainer:
    """TGN训练器"""
    
    def __init__(self, 
                 model: TGN_GAT,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 设备
        self.device = torch.device(config['device'])
        self.model.to(self.device)
        
        # 优化器和学习率调度器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 损失函数
        self.criterion = MultiTaskLoss()
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        
        # Wandb初始化
        if config.get('use_wandb', False):
            wandb.init(
                project='tgn-link-prediction',
                config=config,
                name=f"tgn_gat_{time.strftime('%Y%m%d_%H%M%S')}"
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'link_quality_loss': 0.0,
            'link_existence_loss': 0.0,
            'horizon_1_loss': 0.0,
            'horizon_3_loss': 0.0,
            'horizon_5_loss': 0.0,
            'horizon_10_loss': 0.0
        }
        
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(tqdm(self.train_loader, desc="Training")):
            # 移动数据到设备
            batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch_data.items()}
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(batch_data)
            
            # 计算损失
            losses = self.criterion(predictions, batch_data)
            
            # 反向传播
            losses['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 累计损失
            for loss_name, loss_value in losses.items():
                if loss_name in epoch_losses:
                    epoch_losses[loss_name] += loss_value.item()
            
            num_batches += 1
            
            # 记录到wandb
            if self.config.get('use_wandb', False) and batch_idx % 10 == 0:
                wandb.log({f'batch_{k}': v.item() for k, v in losses.items()})
        
        # 平均损失
        for loss_name in epoch_losses:
            epoch_losses[loss_name] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        
        self.model.eval()
        epoch_losses = {
            'total_loss': 0.0,
            'link_quality_loss': 0.0,
            'link_existence_loss': 0.0,
            'horizon_1_loss': 0.0,
            'horizon_3_loss': 0.0,
            'horizon_5_loss': 0.0,
            'horizon_10_loss': 0.0
        }
        
        predictions_list = []
        targets_list = []
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validating"):
                # 移动数据到设备
                batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch_data.items()}
                
                # 前向传播
                predictions = self.model(batch_data)
                
                # 计算损失
                losses = self.criterion(predictions, batch_data)
                
                # 累计损失
                for loss_name, loss_value in losses.items():
                    if loss_name in epoch_losses:
                        epoch_losses[loss_name] += loss_value.item()
                
                # 保存预测结果用于计算指标
                if 'link_quality' in predictions:
                    predictions_list.append(predictions['link_quality'].cpu())
                    targets_list.append(batch_data['edge_targets'].cpu())
                
                num_batches += 1
        
        # 平均损失
        for loss_name in epoch_losses:
            epoch_losses[loss_name] /= num_batches
        
        # 计算评估指标
        if predictions_list and targets_list:
            all_predictions = torch.cat(predictions_list, dim=0).numpy()
            all_targets = torch.cat(targets_list, dim=0).numpy()
            
            metrics = self._compute_metrics(all_predictions, all_targets)
            epoch_losses.update(metrics)
        
        return epoch_losses
    
    def _compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        
        metrics = {}
        
        # 对每个输出维度计算指标
        feature_names = ['RSSI', 'SNR', 'Bandwidth', 'Latency', 'Packet_Loss']
        
        for i, feature_name in enumerate(feature_names):
            pred_i = predictions[:, i]
            target_i = targets[:, i]
            
            # 回归指标
            mse = mean_squared_error(target_i, pred_i)
            mae = mean_absolute_error(target_i, pred_i)
            r2 = r2_score(target_i, pred_i)
            
            metrics[f'{feature_name}_MSE'] = mse
            metrics[f'{feature_name}_MAE'] = mae
            metrics[f'{feature_name}_R2'] = r2
        
        # 总体指标
        overall_mse = mean_squared_error(targets.flatten(), predictions.flatten())
        overall_mae = mean_absolute_error(targets.flatten(), predictions.flatten())
        
        metrics['Overall_MSE'] = overall_mse
        metrics['Overall_MAE'] = overall_mae
        
        return metrics
    
    def train(self):
        """完整训练流程"""
        
        print(f"Starting training for {self.config['num_epochs']} epochs...")
        
        for epoch in range(self.config['num_epochs']):
            start_time = time.time()
            
            # 训练
            train_losses = self.train_epoch()
            
            # 验证
            val_losses = self.validate_epoch()
            
            # 学习率调度
            self.scheduler.step(val_losses['total_loss'])
            
            # 保存损失历史
            self.train_losses.append(train_losses)
            self.val_losses.append(val_losses)
            
            epoch_time = time.time() - start_time
            
            # 打印进度
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} ({epoch_time:.2f}s)")
            print(f"Train Loss: {train_losses['total_loss']:.6f}")
            print(f"Val Loss: {val_losses['total_loss']:.6f}")
            
            # 记录到wandb
            if self.config.get('use_wandb', False):
                log_data = {f'train_{k}': v for k, v in train_losses.items()}
                log_data.update({f'val_{k}': v for k, v in val_losses.items()})
                log_data['epoch'] = epoch
                log_data['learning_rate'] = self.optimizer.param_groups[0]['lr']
                wandb.log(log_data)
            
            # 早停检查
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.patience_counter = 0
                
                # 保存最佳模型
                self.save_model('best_model.pth')
                print("Saved best model!")
                
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            print("-" * 50)
        
        print("Training completed!")
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def save_model(self, filename: str):
        """保存模型"""
        
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        os.makedirs('models', exist_ok=True)
        torch.save(model_state, os.path.join('models', filename))
    
    def load_model(self, filename: str):
        """加载模型"""
        
        model_state = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(model_state['model_state_dict'])
        self.optimizer.load_state_dict(model_state['optimizer_state_dict'])
        
        self.train_losses = model_state.get('train_losses', [])
        self.val_losses = model_state.get('val_losses', [])
        self.best_val_loss = model_state.get('best_val_loss', float('inf'))
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        
        if not self.train_losses or not self.val_losses:
            return
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 总损失曲线
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 3, 1)
        train_total = [loss['total_loss'] for loss in self.train_losses]
        val_total = [loss['total_loss'] for loss in self.val_losses]
        plt.plot(epochs, train_total, 'b-', label='Train')
        plt.plot(epochs, val_total, 'r-', label='Validation')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 链路质量损失
        plt.subplot(2, 3, 2)
        train_link = [loss['link_quality_loss'] for loss in self.train_losses]
        val_link = [loss['link_quality_loss'] for loss in self.val_losses]
        plt.plot(epochs, train_link, 'b-', label='Train')
        plt.plot(epochs, val_link, 'r-', label='Validation')
        plt.title('Link Quality Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 多步预测损失
        horizons = ['horizon_1', 'horizon_3', 'horizon_5', 'horizon_10']
        colors = ['green', 'orange', 'purple', 'brown']
        
        for i, (horizon, color) in enumerate(zip(horizons, colors)):
            plt.subplot(2, 3, 3 + i)
            train_horizon = [loss[f'{horizon}_loss'] for loss in self.train_losses]
            val_horizon = [loss[f'{horizon}_loss'] for loss in self.val_losses]
            plt.plot(epochs, train_horizon, f'{color[0]}-', label='Train')
            plt.plot(epochs, val_horizon, f'{color[0]}--', label='Validation')
            plt.title(f'{horizon.title()} Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主训练函数"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Train TGN Model')
    parser.add_argument('--data_dir', required=True, help='Directory containing preprocessed data')
    parser.add_argument('--config', help='Training config file')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--resume', help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # 加载配置
    config = create_model_config()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            user_config = json.load(f)
            config.update(user_config)
    
    config['use_wandb'] = args.use_wandb
    
    # 创建数据集
    train_dataset = NetworkDataset(
        data_file=os.path.join(args.data_dir, 'train_data.pkl'),
        scalers_file=os.path.join(args.data_dir, 'feature_scalers.pkl')
    )
    
    val_dataset = NetworkDataset(
        data_file=os.path.join(args.data_dir, 'val_data.pkl'),
        scalers_file=os.path.join(args.data_dir, 'feature_scalers.pkl')
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # 创建模型
    model_params = {k: v for k, v in config.items() if k in TGN_GAT.__init__.__code__.co_varnames}
    model = TGN_GAT(**model_params)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 创建训练器
    trainer = TGNTrainer(model, train_loader, val_loader, config)
    
    # 恢复训练（如果指定）
    if args.resume and os.path.exists(args.resume):
        trainer.load_model(args.resume)
        print(f"Resumed training from {args.resume}")
    
    # 开始训练
    trainer.train()

if __name__ == '__main__':
    main()