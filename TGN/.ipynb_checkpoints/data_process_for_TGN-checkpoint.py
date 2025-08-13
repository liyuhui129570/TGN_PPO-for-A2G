#!/usr/bin/env python

"""
仿真数据预处理器 - 将网络仿真数据转换为TGN训练格式
Data Preprocessor for TGN Training

主要功能：
1. 加载仿真数据
2. 特征工程和标准化
3. 构建时序图序列
4. 生成训练/验证/测试数据集
"""

import numpy as np
import pandas as pd
import pickle
import json
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict, deque
import networkx as nx
from typing import List, Dict, Tuple, Optional
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NetworkDataPreprocessor:
    """网络数据预处理器"""
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        
        # 特征缩放器
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.global_scaler = StandardScaler()
        
        # 节点类型映射
        self.node_type_mapping = {
            'vehicle': 0,
            'uav': 1, 
            'base_station': 2
        }
        
        # 数据统计
        self.data_stats = {}
        
    def _load_config(self, config_file: str) -> dict:
        """加载预处理配置"""
        default_config = {
            'sequence_length': 20,          # 时序窗口长度
            'prediction_horizons': [1, 3, 5, 10],  # 预测步长
            'min_link_quality': 0.1,       # 最小链路质量阈值
            'max_nodes': 100,              # 最大节点数
            'feature_engineering': {
                'temporal_features': True,   # 时序特征
                'spatial_features': True,    # 空间特征
                'mobility_features': True,   # 移动性特征
                'communication_features': True  # 通信特征
            },
            'data_split': {
                'train': 0.7,
                'val': 0.15,
                'test': 0.15
            },
            'normalization': 'standard'     # 'standard', 'minmax', 'robust'
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def load_simulation_data(self, data_file: str) -> dict:
        """加载仿真数据"""
        print(f"Loading simulation data from {data_file}...")
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded {len(data['network_snapshots'])} network snapshots")
        print(f"Time span: {self._calculate_time_span(data['network_snapshots']):.2f} seconds")
        
        return data
    
    def _calculate_time_span(self, snapshots: List[dict]) -> float:
        """计算数据时间跨度"""
        if len(snapshots) < 2:
            return 0.0
        return snapshots[-1]['timestamp'] - snapshots[0]['timestamp']
    
    def preprocess_data(self, simulation_data: dict) -> dict:
        """主预处理流程"""
        print("Starting data preprocessing...")
        
        # 1. 清理和过滤数据
        cleaned_data = self._clean_data(simulation_data['network_snapshots'])
        
        # 2. 特征工程
        enriched_data = self._feature_engineering(cleaned_data)
        
        # 3. 构建时序图序列
        graph_sequences = self._build_temporal_sequences(enriched_data)
        
        # 4. 创建训练数据
        train_data = self._create_training_data(graph_sequences)
        
        # 5. 数据分割
        datasets = self._split_datasets(train_data)
        
        # 6. 特征标准化
        normalized_datasets = self._normalize_features(datasets)
        
        print("Data preprocessing completed!")
        return normalized_datasets
    
    def _clean_data(self, snapshots: List[dict]) -> List[dict]:
        """数据清理和过滤"""
        print("Cleaning and filtering data...")
        
        cleaned_snapshots = []
        
        for snapshot in snapshots:
            # 过滤无效数据
            if not snapshot.get('nodes') or not snapshot.get('edges'):
                continue
                
            # 过滤节点数量过少的快照
            if len(snapshot['nodes']) < 3:
                continue
            
            # 清理节点数据
            cleaned_nodes = {}
            for node_id, node_data in snapshot['nodes'].items():
                if self._is_valid_node(node_data):
                    cleaned_nodes[node_id] = self._clean_node_data(node_data)
            
            # 清理边数据
            cleaned_edges = []
            valid_node_ids = set(cleaned_nodes.keys())
            
            for edge in snapshot['edges']:
                if (edge['src'] in valid_node_ids and 
                    edge['dst'] in valid_node_ids and
                    self._is_valid_edge(edge)):
                    cleaned_edges.append(self._clean_edge_data(edge))
            
            if len(cleaned_nodes) >= 3 and len(cleaned_edges) >= 1:
                cleaned_snapshot = {
                    'timestamp': snapshot['timestamp'],
                    'graph_id': snapshot.get('graph_id', 0),
                    'nodes': cleaned_nodes,
                    'edges': cleaned_edges,
                    'global_features': snapshot.get('global_features', {})
                }
                cleaned_snapshots.append(cleaned_snapshot)
        
        print(f"Cleaned data: {len(cleaned_snapshots)} valid snapshots")
        return cleaned_snapshots
    
    def _is_valid_node(self, node_data: dict) -> bool:
        """检查节点数据有效性"""
        required_fields = ['type', 'position']
        return all(field in node_data for field in required_fields)
    
    def _is_valid_edge(self, edge_data: dict) -> bool:
        """检查边数据有效性"""
        required_fields = ['src', 'dst', 'rssi']
        if not all(field in edge_data for field in required_fields):
            return False
        
        # 检查链路质量阈值
        rssi = edge_data.get('rssi', -100)
        return rssi > -80  # 过滤信号过弱的链路
    
    def _clean_node_data(self, node_data: dict) -> dict:
        """清理节点数据"""
        cleaned = {
            'type': node_data['type'],
            'position': node_data['position'][:3] if len(node_data['position']) >= 3 else node_data['position'] + [0],
            'velocity': node_data.get('velocity', 0),
            'energy': node_data.get('energy', 100),
            'load': node_data.get('load', 0),
            'connectivity': node_data.get('connectivity', 0)
        }
        
        # 添加类型特定特征
        if node_data['type'] == 'uav':
            cleaned['battery'] = node_data.get('battery', 100)
            cleaned['altitude'] = cleaned['position'][2]
        elif node_data['type'] == 'vehicle':
            cleaned['speed'] = node_data.get('velocity', 0)
            cleaned['direction'] = node_data.get('direction', 0)
        elif node_data['type'] == 'base_station':
            cleaned['coverage_radius'] = node_data.get('coverage_radius', 500)
            cleaned['capacity'] = node_data.get('capacity', 1000)
        
        return cleaned
    
    def _clean_edge_data(self, edge_data: dict) -> dict:
        """清理边数据"""
        return {
            'src': edge_data['src'],
            'dst': edge_data['dst'],
            'rssi': edge_data.get('rssi', -70),
            'snr': edge_data.get('snr', 10),
            'bandwidth': edge_data.get('bandwidth', 20),
            'latency': edge_data.get('latency', 10),
            'packet_loss': edge_data.get('packet_loss', 0.01),
            'distance': edge_data.get('distance', 100),
            'los_probability': edge_data.get('los_probability', 0.8)
        }
    
    def _feature_engineering(self, snapshots: List[dict]) -> List[dict]:
        """特征工程"""
        print("Performing feature engineering...")
        
        enriched_snapshots = []
        
        for i, snapshot in enumerate(snapshots):
            enriched_snapshot = snapshot.copy()
            
            # 计算时序特征
            if self.config['feature_engineering']['temporal_features']:
                enriched_snapshot = self._add_temporal_features(enriched_snapshot, i, snapshots)
            
            # 计算空间特征
            if self.config['feature_engineering']['spatial_features']:
                enriched_snapshot = self._add_spatial_features(enriched_snapshot)
            
            # 计算移动性特征
            if self.config['feature_engineering']['mobility_features'] and i > 0:
                enriched_snapshot = self._add_mobility_features(enriched_snapshot, snapshots[i-1])
            
            # 计算通信特征
            if self.config['feature_engineering']['communication_features']:
                enriched_snapshot = self._add_communication_features(enriched_snapshot)
            
            enriched_snapshots.append(enriched_snapshot)
        
        return enriched_snapshots
    
    def _add_temporal_features(self, snapshot: dict, index: int, all_snapshots: List[dict]) -> dict:
        """添加时序特征"""
        # 时间戳特征
        snapshot['temporal_features'] = {
            'relative_time': index / len(all_snapshots),
            'time_of_day': (snapshot['timestamp'] % 86400) / 86400,  # 一天中的时间
            'sequence_position': index
        }
        
        # 历史统计特征
        if index > 0:
            recent_snapshots = all_snapshots[max(0, index-5):index]
            snapshot['temporal_features'].update({
                'avg_node_count': np.mean([len(s['nodes']) for s in recent_snapshots]),
                'avg_edge_count': np.mean([len(s['edges']) for s in recent_snapshots])
            })
        
        return snapshot
    
    def _add_spatial_features(self, snapshot: dict) -> dict:
        """添加空间特征"""
        nodes = snapshot['nodes']
        positions = np.array([node['position'] for node in nodes.values()])
        
        # 全局空间统计
        spatial_features = {
            'network_center': positions.mean(axis=0).tolist(),
            'network_spread': positions.std(axis=0).tolist(),
            'min_coordinates': positions.min(axis=0).tolist(),
            'max_coordinates': positions.max(axis=0).tolist()
        }
        
        # 节点密度分析
        if len(positions) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(positions[:, :2])  # 只考虑x,y坐标
            spatial_features.update({
                'avg_node_distance': distances.mean(),
                'min_node_distance': distances.min(),
                'max_node_distance': distances.max()
            })
        
        snapshot['spatial_features'] = spatial_features
        return snapshot
    
    def _add_mobility_features(self, current_snapshot: dict, previous_snapshot: dict) -> dict:
        """添加移动性特征"""
        mobility_features = {}
        
        current_nodes = current_snapshot['nodes']
        previous_nodes = previous_snapshot['nodes']
        
        # 计算节点移动性
        velocities = []
        for node_id in current_nodes:
            if node_id in previous_nodes:
                curr_pos = np.array(current_nodes[node_id]['position'])
                prev_pos = np.array(previous_nodes[node_id]['position'])
                
                time_diff = current_snapshot['timestamp'] - previous_snapshot['timestamp']
                if time_diff > 0:
                    velocity = np.linalg.norm(curr_pos - prev_pos) / time_diff
                    velocities.append(velocity)
        
        if velocities:
            mobility_features = {
                'avg_velocity': np.mean(velocities),
                'max_velocity': np.max(velocities),
                'velocity_std': np.std(velocities)
            }
        
        current_snapshot['mobility_features'] = mobility_features
        return current_snapshot
    
    def _add_communication_features(self, snapshot: dict) -> dict:
        """添加通信特征"""
        edges = snapshot['edges']
        
        if not edges:
            snapshot['communication_features'] = {}
            return snapshot
        
        # 链路质量统计
        rssi_values = [edge['rssi'] for edge in edges]
        snr_values = [edge['snr'] for edge in edges]
        latencies = [edge['latency'] for edge in edges]
        
        communication_features = {
            'avg_rssi': np.mean(rssi_values),
            'min_rssi': np.min(rssi_values),
            'max_rssi': np.max(rssi_values),
            'avg_snr': np.mean(snr_values),
            'avg_latency': np.mean(latencies),
            'link_density': len(edges) / max(len(snapshot['nodes']) * (len(snapshot['nodes']) - 1) // 2, 1)
        }
        
        snapshot['communication_features'] = communication_features
        return snapshot
    
    def _build_temporal_sequences(self, snapshots: List[dict]) -> List[dict]:
        """构建时序图序列"""
        print("Building temporal graph sequences...")
        
        sequence_length = self.config['sequence_length']
        sequences = []
        
        for i in range(len(snapshots) - sequence_length + 1):
            sequence = {
                'snapshots': snapshots[i:i + sequence_length],
                'sequence_id': i,
                'start_time': snapshots[i]['timestamp'],
                'end_time': snapshots[i + sequence_length - 1]['timestamp']
            }
            sequences.append(sequence)
        
        print(f"Created {len(sequences)} temporal sequences")
        return sequences
    
    def _create_training_data(self, sequences: List[dict]) -> List[dict]:
        """创建训练数据"""
        print("Creating training data...")
        
        training_samples = []
        prediction_horizons = self.config['prediction_horizons']
        
        for seq_id, sequence in enumerate(sequences):
            snapshots = sequence['snapshots']
            
            # 输入序列（除了最后几个时间步）
            input_length = len(snapshots) - max(prediction_horizons)
            if input_length < 5:  # 确保有足够的输入数据
                continue
            
            input_snapshots = snapshots[:input_length]
            
            # 为每个预测步长创建目标
            targets = {}
            for horizon in prediction_horizons:
                if input_length + horizon - 1 < len(snapshots):
                    target_snapshot = snapshots[input_length + horizon - 1]
                    targets[f'horizon_{horizon}'] = self._extract_prediction_targets(target_snapshot)
            
            if targets:  # 只有当有有效目标时才添加样本
                sample = {
                    'sequence_id': seq_id,
                    'input_sequence': input_snapshots,
                    'targets': targets,
                    'metadata': {
                        'start_time': sequence['start_time'],
                        'input_length': input_length
                    }
                }
                training_samples.append(sample)
        
        print(f"Created {len(training_samples)} training samples")
        return training_samples
    
    def _extract_prediction_targets(self, snapshot: dict) -> dict:
        """提取预测目标"""
        targets = {
            'link_qualities': {},
            'node_positions': {},
            'network_metrics': {}
        }
        
        # 链路质量目标
        for edge in snapshot['edges']:
            link_id = f"{edge['src']}-{edge['dst']}"
            targets['link_qualities'][link_id] = {
                'rssi': edge['rssi'],
                'snr': edge['snr'],
                'bandwidth': edge['bandwidth'],
                'latency': edge['latency'],
                'packet_loss': edge['packet_loss']
            }
        
        # 节点位置目标（主要用于移动节点）
        for node_id, node_data in snapshot['nodes'].items():
            if node_data['type'] in ['vehicle', 'uav']:
                targets['node_positions'][node_id] = node_data['position']
        
        # 网络级指标
        targets['network_metrics'] = snapshot.get('global_features', {})
        
        return targets
    
    def _split_datasets(self, training_data: List[dict]) -> dict:
        """数据集分割"""
        print("Splitting datasets...")
        
        split_config = self.config['data_split']
        
        # 按时间顺序分割（而不是随机分割）
        total_samples = len(training_data)
        train_end = int(total_samples * split_config['train'])
        val_end = int(total_samples * (split_config['train'] + split_config['val']))
        
        datasets = {
            'train': training_data[:train_end],
            'val': training_data[train_end:val_end],
            'test': training_data[val_end:]
        }
        
        print(f"Dataset split - Train: {len(datasets['train'])}, "
              f"Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")
        
        return datasets
    
    def _normalize_features(self, datasets: dict) -> dict:
        """特征标准化"""
        print("Normalizing features...")
        
        # 收集所有训练数据的特征
        train_node_features = []
        train_edge_features = []
        train_global_features = []
        
        for sample in datasets['train']:
            for snapshot in sample['input_sequence']:
                # 节点特征
                for node_data in snapshot['nodes'].values():
                    node_features = self._extract_node_features(node_data)
                    train_node_features.append(node_features)
                
                # 边特征
                for edge_data in snapshot['edges']:
                    edge_features = self._extract_edge_features(edge_data)
                    train_edge_features.append(edge_features)
                
                # 全局特征
                global_features = self._extract_global_features(snapshot)
                if global_features:
                    train_global_features.append(global_features)
        
        # 拟合缩放器
        if train_node_features:
            self.node_scaler.fit(train_node_features)
        if train_edge_features:
            self.edge_scaler.fit(train_edge_features)
        if train_global_features:
            self.global_scaler.fit(train_global_features)
        
        # 应用标准化
        normalized_datasets = {}
        for split_name, split_data in datasets.items():
            normalized_datasets[split_name] = self._apply_normalization(split_data)
        
        return normalized_datasets
    
    def _extract_node_features(self, node_data: dict) -> List[float]:
        """提取节点特征向量"""
        features = [
            node_data['position'][0],  # x
            node_data['position'][1],  # y
            node_data['position'][2],  # z
            node_data.get('velocity', 0),
            node_data.get('energy', 100),
            node_data.get('load', 0),
            node_data.get('connectivity', 0),
            self.node_type_mapping.get(node_data['type'], 0)
        ]
        
        # 添加类型特定特征
        if node_data['type'] == 'uav':
            features.extend([
                node_data.get('battery', 100),
                node_data.get('altitude', 50)
            ])
        elif node_data['type'] == 'vehicle':
            features.extend([
                node_data.get('speed', 0),
                node_data.get('direction', 0)
            ])
        elif node_data['type'] == 'base_station':
            features.extend([
                node_data.get('coverage_radius', 500),
                node_data.get('capacity', 1000)
            ])
        else:
            features.extend([0, 0])  # 填充
        
        return features
    
    def _extract_edge_features(self, edge_data: dict) -> List[float]:
        """提取边特征向量"""
        return [
            edge_data['rssi'],
            edge_data['snr'],
            edge_data['bandwidth'],
            edge_data['latency'],
            edge_data['packet_loss'],
            edge_data['distance'],
            edge_data['los_probability']
        ]
    
    def _extract_global_features(self, snapshot: dict) -> List[float]:
        """提取全局特征向量"""
        global_feat = snapshot.get('global_features', {})
        spatial_feat = snapshot.get('spatial_features', {})
        comm_feat = snapshot.get('communication_features', {})
        
        features = [
            global_feat.get('total_nodes', 0),
            global_feat.get('vehicle_count', 0),
            global_feat.get('uav_count', 0),
            global_feat.get('bs_count', 0),
            global_feat.get('network_density', 0),
            comm_feat.get('avg_rssi', -70),
            comm_feat.get('avg_snr', 10),
            comm_feat.get('link_density', 0)
        ]
        
        return features
    
    def _apply_normalization(self, split_data: List[dict]) -> List[dict]:
        """应用标准化到数据集"""
        normalized_data = []
        
        for sample in split_data:
            normalized_sample = sample.copy()
            normalized_sequence = []
            
            for snapshot in sample['input_sequence']:
                normalized_snapshot = snapshot.copy()
                
                # 标准化节点特征
                normalized_nodes = {}
                for node_id, node_data in snapshot['nodes'].items():
                    node_features = self._extract_node_features(node_data)
                    normalized_features = self.node_scaler.transform([node_features])[0]
                    
                    normalized_node = node_data.copy()
                    normalized_node['normalized_features'] = normalized_features.tolist()
                    normalized_nodes[node_id] = normalized_node
                
                normalized_snapshot['nodes'] = normalized_nodes
                
                # 标准化边特征
                normalized_edges = []
                for edge_data in snapshot['edges']:
                    edge_features = self._extract_edge_features(edge_data)
                    normalized_features = self.edge_scaler.transform([edge_features])[0]
                    
                    normalized_edge = edge_data.copy()
                    normalized_edge['normalized_features'] = normalized_features.tolist()
                    normalized_edges.append(normalized_edge)
                
                normalized_snapshot['edges'] = normalized_edges
                normalized_sequence.append(normalized_snapshot)
            
            normalized_sample['input_sequence'] = normalized_sequence
            normalized_data.append(normalized_sample)
        
        return normalized_data
    
    def save_processed_data(self, processed_data: dict, output_dir: str):
        """保存处理后的数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存数据集
        for split_name, split_data in processed_data.items():
            output_file = os.path.join(output_dir, f'{split_name}_data.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"Saved {split_name} data to {output_file}")
        
        # 保存缩放器
        scalers = {
            'node_scaler': self.node_scaler,
            'edge_scaler': self.edge_scaler,
            'global_scaler': self.global_scaler
        }
        
        scaler_file = os.path.join(output_dir, 'feature_scalers.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"Saved feature scalers to {scaler_file}")
        
        # 保存配置和统计信息
        config_file = os.path.join(output_dir, 'preprocessing_config.json')
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        stats_file = os.path.join(output_dir, 'data_statistics.json')
        stats = self._compute_dataset_statistics(processed_data)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Data preprocessing completed! Output saved to {output_dir}")
    
    def _compute_dataset_statistics(self, datasets: dict) -> dict:
        """计算数据集统计信息"""
        stats = {}
        
        for split_name, split_data in datasets.items():
            total_samples = len(split_data)
            total_snapshots = sum(len(sample['input_sequence']) for sample in split_data)
            
            # 节点和边的统计
            node_counts = []
            edge_counts = []
            
            for sample in split_data:
                for snapshot in sample['input_sequence']:
                    node_counts.append(len(snapshot['nodes']))
                    edge_counts.append(len(snapshot['edges']))
            
            stats[split_name] = {
                'total_samples': total_samples,
                'total_snapshots': total_snapshots,
                'avg_nodes_per_snapshot': np.mean(node_counts),
                'avg_edges_per_snapshot': np.mean(edge_counts),
                'max_nodes': np.max(node_counts) if node_counts else 0,
                'max_edges': np.max(edge_counts) if edge_counts else 0
            }
        
        return stats

def main():
    """主函数 - 数据预处理流程"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Network Data Preprocessor for TGN')
    parser.add_argument('--input', required=True, help='Input simulation data file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--config', help='Preprocessing config file')
    
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = NetworkDataPreprocessor(args.config)
    
    # 加载仿真数据
    simulation_data = preprocessor.load_simulation_data(args.input)
    
    # 预处理数据
    processed_data = preprocessor.preprocess_data(simulation_data)
    
    # 保存处理后的数据
    preprocessor.save_processed_data(processed_data, args.output)

if __name__ == '__main__':
    main()