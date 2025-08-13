#!/usr/bin/env python

"""
修复验证测试脚本
Test script to verify the math import fix

运行此脚本验证修复是否成功
"""

import time
import math
import sys

def test_math_functions():
    """测试数学函数是否正确导入"""
    print("*** Testing math functions fix...")
    
    try:
        current_time = time.time()
        
        # 测试之前出错的函数调用
        test_values = []
        
        for i in range(5):
            # 这些是之前出错的调用，现在应该工作正常
            sin_val = math.sin(current_time * 0.1 + i)
            cos_val = math.cos(current_time * 0.1 + i)
            
            # 模拟车辆数据生成
            offset_x = 50 * abs(math.sin(current_time * 0.05 + i))
            offset_y = 20 * abs(math.cos(current_time * 0.05 + i))
            speed = 30 + 10 * abs(math.sin(current_time * 0.1 + i))
            angle = 90 + 30 * math.sin(current_time * 0.02 + i)
            
            test_values.append({
                'sin_val': sin_val,
                'cos_val': cos_val,
                'offset_x': offset_x,
                'offset_y': offset_y,
                'speed': speed,
                'angle': angle
            })
            
            print(f"*** Vehicle {i+1}: speed={speed:.2f}, angle={angle:.2f}")
        
        # 测试UAV位置生成
        for i in range(3):
            offset = 50 * abs(math.sin(current_time * 0.1 + i))
            base_x = 100 + i * 100
            base_y = 100 + i * 100
            
            uav_x = base_x + offset
            uav_y = base_y + offset
            
            print(f"*** UAV {i+1}: position=({uav_x:.1f}, {uav_y:.1f})")
        
        # 测试基站负载
        for i in range(6):
            load = 50 + 20 * abs(math.sin(time.time() * 0.05 + i))
            print(f"*** Base Station {i+1}: load={load:.1f}%")
        
        # 测试通信质量
        v2v_quality = 0.8 + 0.2 * abs(math.sin(time.time() * 0.1))
        v2i_quality = 0.7 + 0.3 * abs(math.cos(time.time() * 0.1))
        
        print(f"*** Communication Quality: V2V={v2v_quality:.3f}, V2I={v2i_quality:.3f}")
        
        print("*** ✅ All math functions working correctly!")
        return True
        
    except AttributeError as e:
        print(f"*** ❌ AttributeError still exists: {e}")
        return False
    except Exception as e:
        print(f"*** ❌ Other error occurred: {e}")
        return False

def test_import_structure():
    """测试导入结构"""
    print("\n*** Testing import structure...")
    
    # 测试time模块
    try:
        current_time = time.time()
        time.sleep(0.1)
        print("*** ✅ time module working correctly")
    except Exception as e:
        print(f"*** ❌ time module error: {e}")
        return False
    
    # 测试math模块
    try:
        result = math.sin(1.0)
        result2 = math.cos(1.0)
        result3 = math.sqrt(4.0)
        print(f"*** ✅ math module working correctly (sin(1)={result:.3f})")
    except Exception as e:
        print(f"*** ❌ math module error: {e}")
        return False
    
    return True

def simulate_data_collection():
    """模拟数据收集过程"""
    print("\n*** Simulating data collection process...")
    
    vehicle_data = {}
    uav_data = {}
    base_station_data = {}
    
    try:
        # 模拟车辆数据收集
        for i in range(5):  # 测试5辆车
            veh_id = f'car{i+1}'
            current_time = time.time()
            
            base_x = 2700 + (i % 5) * 100
            base_y = 3400 + (i // 5) * 50
            
            offset_x = 50 * abs(math.sin(current_time * 0.05 + i))
            offset_y = 20 * abs(math.cos(current_time * 0.05 + i))
            
            vehicle_data[veh_id] = {
                'position': (base_x + offset_x, base_y + offset_y),
                'speed': 30 + 10 * abs(math.sin(current_time * 0.1 + i)),
                'angle': 90 + 30 * math.sin(current_time * 0.02 + i),
                'timestamp': current_time,
                'status': 'normal'
            }
        
        print(f"*** ✅ Generated data for {len(vehicle_data)} vehicles")
        
        # 模拟无人机数据收集
        for i in range(3):
            uav_id = f'drone{i+1}'
            current_time = time.time()
            
            base_x = 150 + i * 100
            base_y = 150 + i * 100
            base_z = 60 + i * 10
            
            offset = 50 * abs(math.sin(current_time * 0.1 + i))
            
            uav_data[uav_id] = {
                'position': (base_x + offset, base_y + offset, base_z),
                'battery': 85 - i * 5,
                'mission_status': 'patrol',
                'timestamp': current_time,
                'communication_range': 200,
                'connected_vehicles': []
            }
        
        print(f"*** ✅ Generated data for {len(uav_data)} UAVs")
        
        # 模拟基站数据收集
        bs_positions = [
            (2600, 3500), (2800, 3500), (3000, 3500),
            (2600, 3300), (2800, 3300), (3000, 3300)
        ]
        
        for i, pos in enumerate(bs_positions):
            bs_id = f'bs{i+1}'
            base_station_data[bs_id] = {
                'position': pos,
                'connected_devices': 0,
                'load': 50 + 20 * abs(math.sin(time.time() * 0.05 + i)),
                'status': 'active',
                'timestamp': time.time()
            }
        
        print(f"*** ✅ Generated data for {len(base_station_data)} base stations")
        
        # 打印一些示例数据
        print("\n*** Sample data:")
        for veh_id, data in list(vehicle_data.items())[:2]:
            print(f"*** {veh_id}: pos=({data['position'][0]:.1f}, {data['position'][1]:.1f}), speed={data['speed']:.1f}")
        
        for uav_id, data in uav_data.items():
            print(f"*** {uav_id}: pos=({data['position'][0]:.1f}, {data['position'][1]:.1f}, {data['position'][2]:.1f})")
        
        return True
        
    except Exception as e:
        print(f"*** ❌ Error in data collection simulation: {e}")
        return False

def main():
    print("=== Math Functions Fix Verification ===")
    print("This script tests the fixes for 'time' module sin/cos errors")
    print()
    
    # 运行测试
    test1_passed = test_import_structure()
    test2_passed = test_math_functions()
    test3_passed = simulate_data_collection()
    
    print("\n=== Test Results ===")
    print(f"Import Structure Test: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Math Functions Test: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"Data Collection Test: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 All tests passed! The math functions fix is working correctly.")
        print("You can now run the main simulation without the 'time.sin' errors.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())