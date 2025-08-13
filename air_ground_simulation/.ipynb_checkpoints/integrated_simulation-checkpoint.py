#!/usr/bin/env python

"""
增强版空地一体化网络仿真系统 - 为TGN+PPO路由优化设计
Enhanced Air-Ground Integrated Network Simulation for TGN+PPO Routing

主要改进：
1. 增强数据收集和存储
2. 支持自定义SUMO路网和CoppeliaSim场景
3. 实时网络状态监控
4. 为TGN训练生成时序图数据
"""

import sys
import os
import time
import threading
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict, deque
import pickle

sys.path.append('/mininet-wifi/mininet/')
sys.path.append('/mininet-wifi')

from mininet.log import setLogLevel, info
from mn_wifi.cli import CLI
from mn_wifi.net import Mininet_wifi
from mn_wifi.link import wmediumd, adhoc, ITSLink
from mn_wifi.wmediumdConnector import interference
from mn_wifi.telemetry import telemetry
from mn_wifi.sumo.runner import sumo
import traci

class NetworkDataCollector:
    """网络数据收集器 - 为TGN训练生成数据"""
    
    def __init__(self, data_dir="./network_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # 数据存储
        self.network_snapshots = deque(maxlen=1000)  # 保存最近1000个快照
        self.link_quality_history = defaultdict(deque)
        self.mobility_data = defaultdict(list)
        
        # 时序图数据
        self.temporal_graphs = []
        self.current_graph_id = 0
        
        # 实时监控数据
        self.current_network_state = {}
        self.data_lock = threading.Lock()
        
    def collect_network_snapshot(self, vehicles, uavs, base_stations):
        """收集完整的网络快照"""
        timestamp = time.time()
        
        snapshot = {
            'timestamp': timestamp,
            'graph_id': self.current_graph_id,
            'nodes': self._collect_node_data(vehicles, uavs, base_stations),
            'edges': self._collect_edge_data(vehicles, uavs, base_stations),
            'global_features': self._collect_global_features(vehicles, uavs, base_stations)
        }
        
        with self.data_lock:
            self.network_snapshots.append(snapshot)
            self.current_network_state = snapshot
            
        # 每100个快照保存一次
        if len(self.network_snapshots) % 100 == 0:
            self._save_snapshots_batch()
            
        self.current_graph_id += 1
        return snapshot
    
    def _collect_node_data(self, vehicles, uavs, base_stations):
        """收集节点数据"""
        nodes = {}
        
        # 收集车辆数据
        for vehicle in vehicles:
            try:
                if hasattr(vehicle, 'position'):
                    pos = vehicle.position
                else:
                    pos = (0, 0, 0)  # 默认位置
                
                # 尝试获取SUMO数据
                sumo_data = self._get_sumo_vehicle_data(vehicle.name)
                
                nodes[vehicle.name] = {
                    'type': 'vehicle',
                    'position': pos,
                    'velocity': sumo_data.get('speed', 0),
                    'direction': sumo_data.get('angle', 0),
                    'interfaces': self._get_interface_info(vehicle),
                    'load': self._calculate_node_load(vehicle),
                    'energy': 100,  # 车辆假设能量充足
                    'connectivity': self._calculate_connectivity(vehicle, vehicles + uavs)
                }
            except Exception as e:
                info(f"*** Error collecting vehicle data for {vehicle.name}: {e}\n")
        
        # 收集无人机数据
        for uav in uavs:
            try:
                if hasattr(uav, 'position'):
                    pos = uav.position
                else:
                    pos = (100, 100, 50)  # 默认位置
                
                nodes[uav.name] = {
                    'type': 'uav',
                    'position': pos,
                    'velocity': self._estimate_uav_velocity(uav),
                    'battery': self._get_uav_battery(uav),
                    'mission_state': 'patrol',  # 任务状态
                    'coverage_area': self._calculate_coverage_area(uav),
                    'relay_capacity': self._calculate_relay_capacity(uav),
                    'connectivity': self._calculate_connectivity(uav, vehicles + uavs)
                }
            except Exception as e:
                info(f"*** Error collecting UAV data for {uav.name}: {e}\n")
        
        # 收集基站数据
        for bs in base_stations:
            try:
                if hasattr(bs, 'position'):
                    pos = bs.position
                else:
                    pos = (2700, 3400, 0)  # 默认位置
                
                nodes[bs.name] = {
                    'type': 'base_station',
                    'position': pos,
                    'load': self._calculate_bs_load(bs),
                    'capacity': 1000,  # Mbps
                    'coverage_radius': 500,  # 米
                    'connected_devices': self._count_connected_devices(bs, vehicles + uavs),
                    'backhaul_quality': 0.95  # 回程链路质量
                }
            except Exception as e:
                info(f"*** Error collecting BS data for {bs.name}: {e}\n")
        
        return nodes
    
    def _collect_edge_data(self, vehicles, uavs, base_stations):
        """收集边（链路）数据"""
        edges = []
        all_nodes = vehicles + uavs + base_stations
        
        for i, node1 in enumerate(all_nodes):
            for node2 in all_nodes[i+1:]:
                # 计算链路质量
                link_quality = self._calculate_link_quality(node1, node2)
                
                if link_quality > 0:  # 只保存有效链路
                    edge = {
                        'src': node1.name,
                        'dst': node2.name,
                        'rssi': link_quality['rssi'],
                        'snr': link_quality['snr'],
                        'bandwidth': link_quality['bandwidth'],
                        'latency': link_quality['latency'],
                        'packet_loss': link_quality['packet_loss'],
                        'interference': link_quality['interference'],
                        'distance': link_quality['distance'],
                        'los_probability': link_quality['los_prob']  # 视距概率
                    }
                    edges.append(edge)
                    
                    # 保存链路质量历史
                    link_id = f"{node1.name}-{node2.name}"
                    self.link_quality_history[link_id].append({
                        'timestamp': time.time(),
                        'quality': link_quality
                    })
        
        return edges
    
    def _calculate_link_quality(self, node1, node2):
        """计算两节点间的链路质量"""
        try:
            # 获取节点位置
            pos1 = getattr(node1, 'position', (0, 0, 0))
            pos2 = getattr(node2, 'position', (0, 0, 0))
            
            # 计算距离
            distance = np.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))
            
            if distance > 500:  # 超出通信范围
                return 0
            
            # 基础信号强度计算（简化的路径损耗模型）
            path_loss = 32.45 + 20 * np.log10(distance/1000) + 20 * np.log10(2400)  # 2.4GHz
            rssi = 20 - path_loss  # 假设发射功率20dBm
            
            # 信噪比计算
            noise_floor = -90  # dBm
            interference = self._calculate_interference(node1, node2)
            snr = rssi - noise_floor - interference
            
            # 视距概率（考虑高度差）
            height_diff = abs(pos1[2] - pos2[2]) if len(pos1) > 2 and len(pos2) > 2 else 0
            los_prob = min(1.0, np.exp(-distance/200) + height_diff/100)
            
            # 带宽和延迟估算
            if snr > 20:
                bandwidth = 54  # Mbps
                latency = distance / 300000000 * 1000  # ms (光速传播)
            elif snr > 10:
                bandwidth = 24
                latency = distance / 300000000 * 1000 + 5
            else:
                bandwidth = 11
                latency = distance / 300000000 * 1000 + 10
            
            # 丢包率估算
            packet_loss = max(0, (20 - snr) / 100) if snr < 20 else 0
            
            return {
                'rssi': rssi,
                'snr': snr,
                'bandwidth': bandwidth,
                'latency': latency,
                'packet_loss': packet_loss,
                'interference': interference,
                'distance': distance,
                'los_prob': los_prob
            }
            
        except Exception as e:
            info(f"*** Error calculating link quality: {e}\n")
            return 0
    
    def _calculate_interference(self, node1, node2):
        """计算干扰水平"""
        # 简化的干扰计算
        base_interference = 5  # dBm
        return base_interference + np.random.normal(0, 2)
    
    def _get_sumo_vehicle_data(self, vehicle_name):
        """从SUMO获取车辆数据"""
        try:
            if traci.isLoaded():
                if vehicle_name in traci.vehicle.getIDList():
                    return {
                        'speed': traci.vehicle.getSpeed(vehicle_name),
                        'angle': traci.vehicle.getAngle(vehicle_name),
                        'position': traci.vehicle.getPosition(vehicle_name)
                    }
        except:
            pass
        return {}
    
    def _collect_global_features(self, vehicles, uavs, base_stations):
        """收集全局网络特征"""
        total_nodes = len(vehicles) + len(uavs) + len(base_stations)
        
        return {
            'total_nodes': total_nodes,
            'vehicle_count': len(vehicles),
            'uav_count': len(uavs),
            'bs_count': len(base_stations),
            'network_density': self._calculate_network_density(vehicles + uavs + base_stations),
            'avg_connectivity': self._calculate_avg_connectivity(vehicles + uavs + base_stations),
            'network_load': self._calculate_network_load(vehicles + uavs + base_stations)
        }
    
    def generate_temporal_graph_data(self, window_size=10):
        """生成TGN训练用的时序图数据"""
        if len(self.network_snapshots) < window_size:
            return None
        
        # 取最近window_size个快照
        recent_snapshots = list(self.network_snapshots)[-window_size:]
        
        temporal_graph = {
            'graph_sequence': recent_snapshots,
            'num_snapshots': len(recent_snapshots),
            'time_span': recent_snapshots[-1]['timestamp'] - recent_snapshots[0]['timestamp'],
            'prediction_targets': self._generate_prediction_targets(recent_snapshots)
        }
        
        return temporal_graph
    
    def _generate_prediction_targets(self, snapshots):
        """生成预测目标（下一个时间步的链路质量）"""
        if len(snapshots) < 2:
            return {}
        
        last_snapshot = snapshots[-1]
        targets = {}
        
        for edge in last_snapshot['edges']:
            link_id = f"{edge['src']}-{edge['dst']}"
            targets[link_id] = {
                'future_rssi': edge['rssi'],
                'future_bandwidth': edge['bandwidth'],
                'future_latency': edge['latency']
            }
        
        return targets
    
    def save_training_data(self, filename=None):
        """保存TGN训练数据"""
        if filename is None:
            filename = f"tgn_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        filepath = os.path.join(self.data_dir, filename)
        
        training_data = {
            'network_snapshots': list(self.network_snapshots),
            'link_quality_history': dict(self.link_quality_history),
            'temporal_graphs': self.temporal_graphs
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(training_data, f)
        
        info(f"*** Training data saved to {filepath}\n")
        return filepath

class EnhancedIntegratedSimulation:
    """增强版集成仿真系统"""
    
    def __init__(self, config_file=None):
        self.net = None
        self.cars = []
        self.drones = []
        self.base_stations = []
        
        # 数据收集器
        self.data_collector = NetworkDataCollector()
        
        # 仿真配置
        self.config = self._load_config(config_file)
        
        # 运行状态
        self.simulation_running = False
        self.data_collection_thread = None
        
    def _load_config(self, config_file):
        """加载仿真配置"""
        default_config = {
            'num_cars': 20,
            'num_drones': 3,
            'num_base_stations': 6,
            'simulation_time': 3600,  # 1小时
            'data_collection_interval': 1.0,  # 1秒
            'sumo_config': './sumo/grid.sumocfg',  # 用户自定义SUMO配置
            'coppelia_scene': './coppelia/uav_scenario.ttt',  # 用户自定义场景
            'output_dir': './simulation_output'
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def start_external_simulators(self):
        """启动外部仿真器（支持自定义配置）"""
        info("*** Starting external simulators with custom configurations\n")
        
        # 启动SUMO（使用用户自定义路网）
        sumo_config = self.config['sumo_config']
        if os.path.exists(sumo_config):
            info(f"*** Starting SUMO with config: {sumo_config}\n")
            self.net.useExternalProgram(
                program=sumo,
                port=8813,
                config_file=sumo_config,
                extra_params=["--start", "--delay", "100", "--step-length", "1.0"],
                clients=1,
                exec_order=0
            )
        else:
            info(f"*** SUMO config file not found: {sumo_config}, using default\n")
            # 使用默认配置...
        
        # 启动CoppeliaSim（使用用户自定义场景）
        self._start_coppelia_sim_custom()
        
        # 启动socket服务器
        self.net.socketServer(ip='127.0.0.1', port=12345)
    
    def _start_coppelia_sim_custom(self):
        """启动CoppeliaSim（支持自定义场景文件）"""
        def start_coppelia():
            scene_file = self.config['coppelia_scene']
            
            if not os.path.exists(scene_file):
                info(f"*** CoppeliaSim scene file not found: {scene_file}\n")
                scene_file = "./integrated_simulation.ttt"  # 回退到默认场景
            
            info(f"*** Starting CoppeliaSim with scene: {scene_file}\n")
            
            # 检查环境并选择启动方式
            if os.environ.get('DISPLAY'):
                cmd = f"DISPLAY={os.environ.get('DISPLAY')} /opt/CoppeliaSim/coppeliaSim.sh -s 100000 {scene_file}"
            elif os.system('which xvfb-run >/dev/null 2>&1') == 0:
                cmd = f"xvfb-run -a /opt/CoppeliaSim/coppeliaSim.sh -s 100000 {scene_file}"
            else:
                cmd = f"/opt/CoppeliaSim/coppeliaSim.sh -h"
            
            os.system(cmd + " &")
        
        coppelia_thread = threading.Thread(target=start_coppelia)
        coppelia_thread.daemon = True
        coppelia_thread.start()
    
    def start_data_collection(self):
        """启动数据收集线程"""
        def collect_data():
            self.simulation_running = True
            start_time = time.time()
            
            while self.simulation_running:
                try:
                    # 收集网络快照
                    snapshot = self.data_collector.collect_network_snapshot(
                        self.cars, self.drones, self.base_stations
                    )
                    
                    # 生成时序图数据（每10个快照生成一次）
                    if len(self.data_collector.network_snapshots) % 10 == 0:
                        temporal_graph = self.data_collector.generate_temporal_graph_data()
                        if temporal_graph:
                            self.data_collector.temporal_graphs.append(temporal_graph)
                    
                    # 检查是否达到仿真时间
                    if time.time() - start_time > self.config['simulation_time']:
                        info("*** Simulation time limit reached\n")
                        break
                    
                    time.sleep(self.config['data_collection_interval'])
                    
                except Exception as e:
                    info(f"*** Error in data collection: {e}\n")
                    time.sleep(1.0)
        
        self.data_collection_thread = threading.Thread(target=collect_data)
        self.data_collection_thread.daemon = True
        self.data_collection_thread.start()
    
    def run_simulation(self):
        """运行完整仿真"""
        try:
            # 创建网络拓扑
            self.create_network_topology()
            
            # 配置网络
            self.configure_network()
            
            # 启动外部仿真器
            self.start_external_simulators()
            
            # 构建网络
            self.net.build()
            time.sleep(10)  # 等待网络稳定
            
            # 启动基站
            for bs in self.base_stations:
                bs.start([])
            
            # 配置IP地址
            self.configure_ip_addressing()
            
            # 启动数据收集
            self.start_data_collection()
            
            # 显示信息并启动CLI
            self.print_network_info()
            info("*** Type 'save_data' to save training data\n")
            info("*** Type 'show_stats' to show network statistics\n")
            
            # 自定义CLI命令
            original_do_save_data = CLI.do_save_data
            original_do_show_stats = CLI.do_show_stats
            
            def do_save_data(self_cli, line):
                filename = self.data_collector.save_training_data()
                info(f"*** Data saved to {filename}\n")
            
            def do_show_stats(self_cli, line):
                stats = self.get_network_statistics()
                for key, value in stats.items():
                    info(f"*** {key}: {value}\n")
            
            CLI.do_save_data = do_save_data
            CLI.do_show_stats = do_show_stats
            
            CLI(self.net)
            
        except Exception as e:
            info(f"*** Error during simulation: {e}\n")
        finally:
            self.cleanup()
    
    def get_network_statistics(self):
        """获取网络统计信息"""
        if not self.data_collector.current_network_state:
            return {"error": "No data collected yet"}
        
        current_state = self.data_collector.current_network_state
        
        stats = {
            "Total Snapshots": len(self.data_collector.network_snapshots),
            "Current Nodes": len(current_state.get('nodes', {})),
            "Current Edges": len(current_state.get('edges', [])),
            "Temporal Graphs": len(self.data_collector.temporal_graphs),
            "Data Collection Time": f"{time.time() - (self.data_collector.network_snapshots[0]['timestamp'] if self.data_collector.network_snapshots else time.time()):.1f}s"
        }
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        info("*** Cleaning up simulation\n")
        
        self.simulation_running = False
        
        # 保存最终数据
        final_data_file = self.data_collector.save_training_data("final_training_data.pkl")
        
        # 生成数据报告
        self._generate_data_report(final_data_file)
        
        # 停止网络
        if self.net:
            self.net.stop()
    
    def _generate_data_report(self, data_file):
        """生成数据收集报告"""
        report = {
            "simulation_config": self.config,
            "data_file": data_file,
            "statistics": self.get_network_statistics(),
            "collection_summary": {
                "total_snapshots": len(self.data_collector.network_snapshots),
                "temporal_graphs": len(self.data_collector.temporal_graphs),
                "unique_links": len(self.data_collector.link_quality_history),
                "data_size_mb": os.path.getsize(data_file) / (1024*1024) if os.path.exists(data_file) else 0
            }
        }
        
        report_file = os.path.join(self.config['output_dir'], "simulation_report.json")
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        info(f"*** Simulation report saved to {report_file}\n")

def main():
    """主函数"""
    setLogLevel('info')
    
    info("*** Enhanced Air-Ground Integrated Network Simulation for TGN+PPO\n")
    
    # 检查配置文件
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    if config_file and not os.path.exists(config_file):
        info(f"*** Config file not found: {config_file}\n")
        config_file = None
    
    # 运行仿真
    simulation = EnhancedIntegratedSimulation(config_file)
    simulation.run_simulation()

if __name__ == '__main__':
    main()