#!/usr/bin/env python

"""
Air-Ground Coordination Script
空地协调通信脚本

实现车辆、无人机、基站之间的协调通信和数据交换

Author: AI Assistant
"""

import sys
import time
import math
import socket
import json
import threading
import subprocess
from collections import defaultdict
from mininet.log import setLogLevel, info

class AirGroundCoordinator:
    def __init__(self):
        self.running = True
        self.vehicle_data = {}
        self.uav_data = {}
        self.base_station_data = {}
        
        # 通信参数
        self.socket_host = '127.0.0.1'
        self.socket_port = 12346  # 不同于UAV sync的端口
        
        # 协调策略参数
        self.coordination_interval = 2.0  # 协调间隔（秒）
        self.emergency_threshold = 5.0     # 紧急情况阈值
        
        # 消息队列
        self.message_queue = []
        self.message_lock = threading.Lock()

    def initialize_coordination(self):
        """初始化协调系统"""
        info("*** Initializing air-ground coordination system\n")
        
        # 启动各个功能线程
        self.start_data_collection_thread()
        self.start_coordination_thread()
        self.start_communication_thread()
        self.start_emergency_monitoring_thread()

    def start_data_collection_thread(self):
        """启动数据收集线程"""
        def collect_data():
            while self.running:
                try:
                    # 收集车辆数据
                    self.collect_vehicle_data()
                    
                    # 收集无人机数据
                    self.collect_uav_data()
                    
                    # 收集基站数据
                    self.collect_base_station_data()
                    
                    time.sleep(1.0)
                    
                except Exception as e:
                    info(f"*** Error in data collection: {e}\n")
                    time.sleep(1.0)
        
        thread = threading.Thread(target=collect_data)
        thread.daemon = True
        thread.start()

    def collect_vehicle_data(self):
        """收集车辆数据"""
        try:
            # 使用SUMO TraCI API获取车辆信息
            import traci
            
            if traci.isLoaded():
                vehicle_ids = traci.vehicle.getIDList()
                
                for veh_id in vehicle_ids:
                    position = traci.vehicle.getPosition(veh_id)
                    speed = traci.vehicle.getSpeed(veh_id)
                    angle = traci.vehicle.getAngle(veh_id)
                    
                    self.vehicle_data[veh_id] = {
                        'position': position,
                        'speed': speed,
                        'angle': angle,
                        'timestamp': time.time(),
                        'status': 'normal'
                    }
        except Exception as e:
            # 如果SUMO不可用，生成模拟数据
            self.generate_simulated_vehicle_data()

    def generate_simulated_vehicle_data(self):
        """生成模拟车辆数据"""
        for i in range(20):  # 假设20辆车
            veh_id = f'car{i+1}'
            current_time = time.time()
            
            # 模拟车辆在道路上的移动
            base_x = 2700 + (i % 5) * 100
            base_y = 3400 + (i // 5) * 50
            
            # 添加时间相关的移动
            offset_x = 50 * abs(math.sin(current_time * 0.05 + i))
            offset_y = 20 * abs(math.cos(current_time * 0.05 + i))
            
            self.vehicle_data[veh_id] = {
                'position': (base_x + offset_x, base_y + offset_y),
                'speed': 30 + 10 * abs(math.sin(current_time * 0.1 + i)),
                'angle': 90 + 30 * math.sin(current_time * 0.02 + i),
                'timestamp': current_time,
                'status': 'normal'
            }

    def collect_uav_data(self):
        """收集无人机数据"""
        # 从文件或共享内存读取UAV位置数据
        try:
            # 这里可以读取UAV position sync脚本生成的数据
            for i in range(3):  # 假设3架无人机
                uav_id = f'drone{i+1}'
                current_time = time.time()
                
                # 模拟无人机数据
                base_x = 150 + i * 100
                base_y = 150 + i * 100
                base_z = 60 + i * 10
                
                offset = 50 * abs(math.sin(current_time * 0.1 + i))
                
                self.uav_data[uav_id] = {
                    'position': (base_x + offset, base_y + offset, base_z),
                    'battery': 85 - i * 5,  # 模拟电池电量
                    'mission_status': 'patrol',
                    'timestamp': current_time,
                    'communication_range': 200,
                    'connected_vehicles': []
                }
                
        except Exception as e:
            info(f"*** Error collecting UAV data: {e}\n")

    def collect_base_station_data(self):
        """收集基站数据"""
        bs_positions = [
            (2600, 3500), (2800, 3500), (3000, 3500),
            (2600, 3300), (2800, 3300), (3000, 3300)
        ]
        
        for i, pos in enumerate(bs_positions):
            bs_id = f'bs{i+1}'
            self.base_station_data[bs_id] = {
                'position': pos,
                'connected_devices': 0,
                'load': 50 + 20 * abs(math.sin(time.time() * 0.05 + i)),
                'status': 'active',
                'timestamp': time.time()
            }

    def start_coordination_thread(self):
        """启动协调线程"""
        def coordinate():
            while self.running:
                try:
                    # 执行主要协调逻辑
                    self.perform_coordination()
                    time.sleep(self.coordination_interval)
                    
                except Exception as e:
                    info(f"*** Error in coordination: {e}\n")
                    time.sleep(self.coordination_interval)
        
        thread = threading.Thread(target=coordinate)
        thread.daemon = True
        thread.start()

    def perform_coordination(self):
        """执行协调任务"""
        # 1. 分析当前网络状态
        network_state = self.analyze_network_state()
        
        # 2. 识别需要协调的情况
        coordination_tasks = self.identify_coordination_tasks(network_state)
        
        # 3. 执行协调策略
        for task in coordination_tasks:
            self.execute_coordination_task(task)

    def analyze_network_state(self):
        """分析网络状态"""
        state = {
            'vehicle_density': self.calculate_vehicle_density(),
            'uav_coverage': self.calculate_uav_coverage(),
            'bs_load_distribution': self.calculate_bs_load_distribution(),
            'communication_quality': self.assess_communication_quality()
        }
        return state

    def calculate_vehicle_density(self):
        """计算车辆密度"""
        if not self.vehicle_data:
            return 0
        
        # 计算每个区域的车辆密度
        regions = defaultdict(int)
        for veh_id, data in self.vehicle_data.items():
            pos = data['position']
            region_x = int(pos[0] // 100)
            region_y = int(pos[1] // 100)
            regions[(region_x, region_y)] += 1
        
        return dict(regions)

    def calculate_uav_coverage(self):
        """计算无人机覆盖情况"""
        coverage = {}
        for uav_id, data in self.uav_data.items():
            pos = data['position']
            comm_range = data['communication_range']
            
            # 计算此无人机覆盖的车辆数量
            covered_vehicles = 0
            for veh_id, veh_data in self.vehicle_data.items():
                veh_pos = veh_data['position']
                distance = ((pos[0] - veh_pos[0])**2 + (pos[1] - veh_pos[1])**2)**0.5
                if distance <= comm_range:
                    covered_vehicles += 1
            
            coverage[uav_id] = {
                'covered_vehicles': covered_vehicles,
                'efficiency': covered_vehicles / max(len(self.vehicle_data), 1)
            }
        
        return coverage

    def calculate_bs_load_distribution(self):
        """计算基站负载分布"""
        loads = {}
        for bs_id, data in self.base_station_data.items():
            loads[bs_id] = data['load']
        return loads

    def assess_communication_quality(self):
        """评估通信质量"""
        # 简化的通信质量评估
        quality_scores = {}
        
        # 基于距离和干扰评估V2V通信质量
        for veh_id in self.vehicle_data:
            quality_scores[veh_id] = {
                'v2v_quality': 0.8 + 0.2 * abs(math.sin(time.time() * 0.1)),
                'v2i_quality': 0.7 + 0.3 * abs(math.cos(time.time() * 0.1))
            }
        
        return quality_scores

    def identify_coordination_tasks(self, network_state):
        """识别需要协调的任务"""
        tasks = []
        
        # 任务1：优化无人机位置
        uav_coverage = network_state['uav_coverage']
        for uav_id, coverage in uav_coverage.items():
            if coverage['efficiency'] < 0.3:  # 覆盖效率低
                tasks.append({
                    'type': 'optimize_uav_position',
                    'uav_id': uav_id,
                    'priority': 'medium'
                })
        
        # 任务2：负载均衡
        bs_loads = network_state['bs_load_distribution']
        max_load = max(bs_loads.values()) if bs_loads else 0
        min_load = min(bs_loads.values()) if bs_loads else 0
        if max_load - min_load > 30:  # 负载差异大
            tasks.append({
                'type': 'load_balancing',
                'max_bs': max(bs_loads, key=bs_loads.get),
                'min_bs': min(bs_loads, key=bs_loads.get),
                'priority': 'high'
            })
        
        # 任务3：通信质量优化
        comm_quality = network_state['communication_quality']
        for node_id, quality in comm_quality.items():
            if quality['v2v_quality'] < 0.5:
                tasks.append({
                    'type': 'improve_communication',
                    'node_id': node_id,
                    'priority': 'medium'
                })
        
        return tasks

    def execute_coordination_task(self, task):
        """执行协调任务"""
        if task['type'] == 'optimize_uav_position':
            self.optimize_uav_position(task['uav_id'])
        elif task['type'] == 'load_balancing':
            self.perform_load_balancing(task['max_bs'], task['min_bs'])
        elif task['type'] == 'improve_communication':
            self.improve_node_communication(task['node_id'])

    def optimize_uav_position(self, uav_id):
        """优化无人机位置"""
        info(f"*** Optimizing position for {uav_id}\n")
        
        # 计算最佳位置（基于车辆分布）
        if not self.vehicle_data:
            return
        
        # 找到车辆密集区域
        vehicle_positions = [data['position'] for data in self.vehicle_data.values()]
        center_x = sum(pos[0] for pos in vehicle_positions) / len(vehicle_positions)
        center_y = sum(pos[1] for pos in vehicle_positions) / len(vehicle_positions)
        
        # 发送位置调整命令
        new_position = (center_x, center_y, 70)  # 固定高度70米
        self.send_uav_position_command(uav_id, new_position)

    def send_uav_position_command(self, uav_id, position):
        """发送无人机位置调整命令"""
        command = {
            'type': 'uav_position_command',
            'uav_id': uav_id,
            'target_position': position,
            'timestamp': time.time()
        }
        
        with self.message_lock:
            self.message_queue.append(command)

    def perform_load_balancing(self, max_bs, min_bs):
        """执行负载均衡"""
        info(f"*** Performing load balancing: {max_bs} -> {min_bs}\n")
        
        # 这里可以实现将部分连接从高负载基站转移到低负载基站
        # 在实际实现中，这需要与网络控制器交互

    def improve_node_communication(self, node_id):
        """改善节点通信质量"""
        info(f"*** Improving communication for {node_id}\n")
        
        # 可能的策略：
        # 1. 调整传输功率
        # 2. 更改信道
        # 3. 启用中继通信

    def start_communication_thread(self):
        """启动通信线程"""
        def handle_communication():
            while self.running:
                try:
                    with self.message_lock:
                        messages_to_process = self.message_queue[:]
                        self.message_queue.clear()
                    
                    for message in messages_to_process:
                        self.process_message(message)
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    info(f"*** Error in communication thread: {e}\n")
                    time.sleep(0.5)
        
        thread = threading.Thread(target=handle_communication)
        thread.daemon = True
        thread.start()

    def process_message(self, message):
        """处理消息"""
        msg_type = message.get('type')
        
        if msg_type == 'uav_position_command':
            # 转发给UAV position sync脚本
            info(f"*** Processing UAV position command for {message['uav_id']}\n")
        elif msg_type == 'emergency_alert':
            self.handle_emergency(message)

    def start_emergency_monitoring_thread(self):
        """启动紧急情况监控线程"""
        def monitor_emergency():
            while self.running:
                try:
                    self.check_emergency_situations()
                    time.sleep(1.0)
                    
                except Exception as e:
                    info(f"*** Error in emergency monitoring: {e}\n")
                    time.sleep(1.0)
        
        thread = threading.Thread(target=monitor_emergency)
        thread.daemon = True
        thread.start()

    def check_emergency_situations(self):
        """检查紧急情况"""
        # 检查车辆紧急情况
        for veh_id, data in self.vehicle_data.items():
            if data['speed'] > 80:  # 超速
                self.trigger_emergency('speeding', veh_id, data)
        
        # 检查无人机紧急情况
        for uav_id, data in self.uav_data.items():
            if data['battery'] < 20:  # 低电量
                self.trigger_emergency('low_battery', uav_id, data)

    def trigger_emergency(self, emergency_type, node_id, data):
        """触发紧急响应"""
        info(f"*** EMERGENCY: {emergency_type} detected for {node_id}\n")
        
        emergency_msg = {
            'type': 'emergency_alert',
            'emergency_type': emergency_type,
            'node_id': node_id,
            'data': data,
            'timestamp': time.time()
        }
        
        with self.message_lock:
            self.message_queue.append(emergency_msg)

    def handle_emergency(self, emergency_msg):
        """处理紧急情况"""
        emergency_type = emergency_msg['emergency_type']
        node_id = emergency_msg['node_id']
        
        if emergency_type == 'speeding':
            info(f"*** Handling speeding emergency for {node_id}\n")
            # 可以通知附近的基站或无人机
            
        elif emergency_type == 'low_battery':
            info(f"*** Handling low battery emergency for {node_id}\n")
            # 指导无人机返回充电站

    def print_status_report(self):
        """打印状态报告"""
        info("*** Air-Ground Coordination Status Report ***\n")
        info(f"*** Active Vehicles: {len(self.vehicle_data)}\n")
        info(f"*** Active UAVs: {len(self.uav_data)}\n")
        info(f"*** Active Base Stations: {len(self.base_station_data)}\n")
        info(f"*** Messages in Queue: {len(self.message_queue)}\n")

    def run(self):
        """运行协调器"""
        try:
            self.initialize_coordination()
            
            info("*** Air-Ground Coordinator started successfully\n")
            
            # 主循环
            while self.running:
                time.sleep(10)
                self.print_status_report()
                
        except KeyboardInterrupt:
            info("*** Stopping Air-Ground Coordinator\n")
            self.running = False
        except Exception as e:
            info(f"*** Error in coordinator: {e}\n")

def main():
    setLogLevel('info')
    
    coordinator = AirGroundCoordinator()
    coordinator.run()

if __name__ == '__main__':
    main()