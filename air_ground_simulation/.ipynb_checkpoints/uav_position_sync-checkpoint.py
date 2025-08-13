#!/usr/bin/env python

"""
UAV Position Synchronization Script
无人机位置同步脚本

从CoppeliaSim获取无人机位置数据并同步到mininet-wifi网络

Author: AI Assistant
"""

import sys
import time
import math
import socket
import json
import threading
from mininet.log import setLogLevel, info

class UAVPositionSync:
    def __init__(self, drone_names):
        self.drone_names = drone_names
        self.socket_host = '127.0.0.1'
        self.socket_port = 12345
        self.running = True
        self.positions = {}
        
        # CoppeliaSim连接参数
        self.coppelia_host = '127.0.0.1'
        self.coppelia_port = 19997  # CoppeliaSim remote API port

    def connect_to_coppelia(self):
        """连接到CoppeliaSim"""
        try:
            import sim
            
            # 关闭之前的连接
            sim.simxFinish(-1)
            
            # 建立新连接
            self.client_id = sim.simxStart(self.coppelia_host, self.coppelia_port, True, True, 5000, 5)
            
            if self.client_id != -1:
                info("*** Connected to CoppeliaSim\n")
                return True
            else:
                info("*** Failed to connect to CoppeliaSim\n")
                return False
                
        except ImportError:
            info("*** CoppeliaSim Python API not available, using socket fallback\n")
            return self.connect_to_socket()

    def connect_to_socket(self):
        """使用socket连接"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.socket_host, self.socket_port))
            info("*** Connected to mininet-wifi socket server\n")
            return True
        except Exception as e:
            info(f"*** Socket connection failed: {e}\n")
            return False

    def get_drone_positions_from_coppelia(self):
        """从CoppeliaSim获取无人机位置"""
        positions = {}
        
        try:
            import sim
            
            for drone_name in self.drone_names:
                # 获取无人机对象句柄
                ret_code, handle = sim.simxGetObjectHandle(
                    self.client_id, 
                    drone_name, 
                    sim.simx_opmode_blocking
                )
                
                if ret_code == sim.simx_return_ok:
                    # 获取位置
                    ret_code, position = sim.simxGetObjectPosition(
                        self.client_id, 
                        handle, 
                        -1,  # 相对于世界坐标系
                        sim.simx_opmode_blocking
                    )
                    
                    if ret_code == sim.simx_return_ok:
                        # CoppeliaSim位置单位通常是米，需要转换为mininet-wifi坐标
                        x = position[0] * 100  # 转换为mininet坐标系
                        y = position[1] * 100
                        z = position[2] * 100
                        
                        positions[drone_name] = {
                            'x': x,
                            'y': y,
                            'z': z,
                            'timestamp': time.time()
                        }
                        
        except ImportError:
            # 如果没有CoppeliaSim API，使用模拟数据
            positions = self.generate_simulated_positions()
        except Exception as e:
            info(f"*** Error getting positions from CoppeliaSim: {e}\n")
            positions = self.generate_simulated_positions()
            
        return positions

    def generate_simulated_positions(self):
        """生成模拟的无人机位置数据（当CoppeliaSim不可用时）"""
        positions = {}
        current_time = time.time()
        
        # 模拟无人机的移动轨迹
        for i, drone_name in enumerate(self.drone_names):
            # 基础位置
            base_x = 100 + i * 100
            base_y = 100 + i * 100
            base_z = 50 + i * 10
            
            # 添加周期性移动
            offset = 50 * abs(math.sin(current_time * 0.1 + i))
            
            positions[drone_name] = {
                'x': base_x + offset,
                'y': base_y + offset,
                'z': base_z,
                'timestamp': current_time
            }
            
        return positions

    def update_mininet_positions(self, positions):
        """更新mininet-wifi中的无人机位置"""
        try:
            for drone_name, pos in positions.items():
                # 构造位置更新命令
                position_str = f"{pos['x']},{pos['y']},{pos['z']}"
                
                # 发送位置更新到socket服务器
                update_msg = {
                    'type': 'position_update',
                    'node': drone_name,
                    'position': position_str,
                    'timestamp': pos['timestamp']
                }
                
                message = json.dumps(update_msg) + '\n'
                self.sock.send(message.encode())
                
        except Exception as e:
            info(f"*** Error updating mininet positions: {e}\n")

    def run_sync_loop(self):
        """运行同步循环"""
        info("*** Starting UAV position synchronization loop\n")
        
        while self.running:
            try:
                # 从CoppeliaSim获取位置
                positions = self.get_drone_positions_from_coppelia()
                
                if positions:
                    # 更新本地位置缓存
                    self.positions.update(positions)
                    
                    # 更新mininet-wifi位置
                    self.update_mininet_positions(positions)
                    
                    # 打印位置信息（调试用）
                    for drone_name, pos in positions.items():
                        info(f"*** {drone_name}: ({pos['x']:.1f}, {pos['y']:.1f}, {pos['z']:.1f})\n")
                
                # 等待下次更新
                time.sleep(1.0)  # 1Hz更新频率
                
            except KeyboardInterrupt:
                info("*** Stopping UAV position sync\n")
                break
            except Exception as e:
                info(f"*** Error in sync loop: {e}\n")
                time.sleep(1.0)

    def start_monitoring_thread(self):
        """启动监控线程"""
        monitor_thread = threading.Thread(target=self.monitor_network_changes)
        monitor_thread.daemon = True
        monitor_thread.start()

    def monitor_network_changes(self):
        """监控网络拓扑变化"""
        while self.running:
            try:
                # 检查无人机间的连通性
                self.check_uav_connectivity()
                time.sleep(5.0)  # 每5秒检查一次
                
            except Exception as e:
                info(f"*** Error in network monitoring: {e}\n")
                time.sleep(5.0)

    def check_uav_connectivity(self):
        """检查无人机间的连通性"""
        # 基于位置计算无人机间的距离
        drone_positions = list(self.positions.items())
        
        for i, (drone1, pos1) in enumerate(drone_positions):
            for j, (drone2, pos2) in enumerate(drone_positions[i+1:], i+1):
                distance = self.calculate_distance(pos1, pos2)
                
                # 如果距离在通信范围内（例如200米）
                if distance < 200:
                    # 可以在这里触发网络拓扑更新
                    pass

    def calculate_distance(self, pos1, pos2):
        """计算两点间的3D距离"""
        dx = pos1['x'] - pos2['x']
        dy = pos1['y'] - pos2['y']
        dz = pos1['z'] - pos2['z']
        return (dx*dx + dy*dy + dz*dz) ** 0.5

    def cleanup(self):
        """清理资源"""
        self.running = False
        
        try:
            import sim
            sim.simxFinish(self.client_id)
        except:
            pass
        
        try:
            self.sock.close()
        except:
            pass

def main():
    if len(sys.argv) < 2:
        print("Usage: python uav_position_sync.py <drone1> <drone2> ... <droneN>")
        sys.exit(1)
    
    drone_names = sys.argv[1:]
    
    setLogLevel('info')
    info(f"*** Starting UAV position sync for: {', '.join(drone_names)}\n")
    
    sync = UAVPositionSync(drone_names)
    
    try:
        # 连接到仿真环境
        if sync.connect_to_coppelia():
            # 启动监控线程
            sync.start_monitoring_thread()
            
            # 运行同步循环
            sync.run_sync_loop()
        else:
            info("*** Failed to connect to simulation environment\n")
            
    except KeyboardInterrupt:
        info("*** Interrupted by user\n")
    finally:
        sync.cleanup()

if __name__ == '__main__':
    main()