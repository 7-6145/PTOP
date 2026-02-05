"""
快速测试脚本
用于验证所有模块功能，避免长时间等待
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, Any

from data_preprocessing import DataProcessor
from spatial_index import SpatialIndex, PopulationCoverageCalculator
from reward_function import RewardFunction
from bus_optimization_env import BusStopOptimizationEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickOptimizer:
    """
    快速优化器 - 用于快速测试和演示
    使用简化的启发式算法替代复杂的PPO训练
    """
    
    def __init__(self, population_csv_path: str, bus_stops_shp_path: str):
        """初始化快速优化器"""
        self.population_csv_path = population_csv_path
        self.bus_stops_shp_path = bus_stops_shp_path
        
        # 加载数据
        processor = DataProcessor(population_csv_path, bus_stops_shp_path)
        self.population_data, self.bus_stops_data, _ = processor.get_processed_data()
        
        # 初始化奖励函数
        self.reward_function = RewardFunction(self.population_data, self.bus_stops_data)
        
        # 原始站点位置
        self.original_positions = self.bus_stops_data[['longitude', 'latitude']].values
        
        logger.info(f"快速优化器初始化完成，{len(self.original_positions)} 个站点")
    
    def gradient_based_optimization(self, max_iterations: int = 50, 
                                   step_size: float = 0.0001) -> Dict[str, Any]:
        """
        基于梯度的快速优化（简化版本）
        
        Args:
            max_iterations: 最大迭代次数
            step_size: 步长
            
        Returns:
            优化结果
        """
        logger.info("开始基于梯度的快速优化...")
        start_time = time.time()
        
        current_positions = self.original_positions.copy()
        best_positions = current_positions.copy()
        
        # 计算初始奖励
        initial_reward = self.reward_function.calculate_reward(current_positions)
        best_reward = initial_reward['total_reward']
        
        logger.info(f"初始奖励: {best_reward:.6f}")
        logger.info(f"初始覆盖率: {initial_reward['coverage_ratio']:.4f}")
        
        history = []
        
        for iteration in range(max_iterations):
            improved = False
            
            # 对每个站点尝试小幅移动
            for i in range(len(current_positions)):
                original_pos = current_positions[i].copy()
                
                # 尝试8个方向的移动
                directions = [
                    [step_size, 0], [-step_size, 0], [0, step_size], [0, -step_size],
                    [step_size, step_size], [step_size, -step_size], 
                    [-step_size, step_size], [-step_size, -step_size]
                ]
                
                best_move = None
                best_local_reward = best_reward
                
                for direction in directions:
                    # 尝试移动
                    test_positions = current_positions.copy()
                    test_positions[i] = original_pos + np.array(direction)
                    
                    # 检查边界
                    bounds_lower, bounds_upper = self.reward_function.get_optimization_bounds()
                    if (test_positions[i, 0] < bounds_lower[i, 0] or 
                        test_positions[i, 0] > bounds_upper[i, 0] or
                        test_positions[i, 1] < bounds_lower[i, 1] or 
                        test_positions[i, 1] > bounds_upper[i, 1]):
                        continue
                    
                    # 计算奖励
                    reward_dict = self.reward_function.calculate_reward(test_positions)
                    test_reward = reward_dict['total_reward']
                    
                    if test_reward > best_local_reward:
                        best_local_reward = test_reward
                        best_move = direction
                        improved = True
                
                # 应用最佳移动
                if best_move is not None:
                    current_positions[i] = original_pos + np.array(best_move)
                    best_reward = best_local_reward
                    best_positions = current_positions.copy()
            
            # 记录历史
            current_reward_dict = self.reward_function.calculate_reward(current_positions)
            history.append({
                'iteration': iteration + 1,
                'reward': current_reward_dict['total_reward'],
                'coverage_ratio': current_reward_dict['coverage_ratio'],
                'coverage_improvement': current_reward_dict['coverage_improvement'],
                'total_movement': current_reward_dict['total_movement']
            })
            
            # 打印进度
            if (iteration + 1) % 10 == 0:
                logger.info(f"迭代 {iteration + 1}: 奖励 {best_reward:.6f}, "
                          f"覆盖率 {current_reward_dict['coverage_ratio']:.4f}")
            
            # 检查收敛
            if not improved:
                logger.info(f"算法收敛于迭代 {iteration + 1}")
                break
        
        optimization_time = time.time() - start_time
        
        # 最终质量评估
        final_quality = self.reward_function.evaluate_solution_quality(best_positions)
        
        result = {
            'method': 'gradient_based',
            'best_positions': best_positions,
            'original_positions': self.original_positions,
            'best_reward': best_reward,
            'initial_reward': initial_reward['total_reward'],
            'improvement': best_reward - initial_reward['total_reward'],
            'optimization_time': optimization_time,
            'iterations': len(history),
            'quality_report': final_quality,
            'history': history
        }
        
        logger.info(f"快速优化完成! 用时: {optimization_time:.2f}秒")
        logger.info(f"奖励改善: {result['improvement']:.6f}")
        logger.info(f"覆盖率改善: {final_quality['quality_metrics']['coverage_improvement_percent']:.2f}%")
        
        return result
    
    def random_search_optimization(self, max_iterations: int = 100, 
                                  search_radius: float = 0.002) -> Dict[str, Any]:
        """
        随机搜索优化（更快的基线方法）
        
        Args:
            max_iterations: 最大迭代次数
            search_radius: 搜索半径
            
        Returns:
            优化结果
        """
        logger.info("开始随机搜索优化...")
        start_time = time.time()
        
        best_positions = self.original_positions.copy()
        initial_reward = self.reward_function.calculate_reward(best_positions)
        best_reward = initial_reward['total_reward']
        
        history = []
        
        for iteration in range(max_iterations):
            # 生成随机扰动
            n_stations_to_move = max(1, len(self.original_positions) // 4)  # 移动25%的站点
            stations_to_move = np.random.choice(len(self.original_positions), 
                                              n_stations_to_move, replace=False)
            
            test_positions = best_positions.copy()
            
            for station_idx in stations_to_move:
                # 随机移动
                perturbation = np.random.normal(0, search_radius, 2)
                test_positions[station_idx] += perturbation
            
            # 检查边界
            bounds_lower, bounds_upper = self.reward_function.get_optimization_bounds()
            test_positions = np.clip(test_positions, bounds_lower, bounds_upper)
            
            # 计算奖励
            reward_dict = self.reward_function.calculate_reward(test_positions)
            test_reward = reward_dict['total_reward']
            
            # 更新最佳解
            if test_reward > best_reward:
                best_positions = test_positions.copy()
                best_reward = test_reward
                logger.info(f"迭代 {iteration + 1}: 发现更优解，奖励 {best_reward:.6f}")
            
            # 记录历史（每10次迭代）
            if (iteration + 1) % 10 == 0:
                current_quality = self.reward_function.calculate_reward(best_positions)
                history.append({
                    'iteration': iteration + 1,
                    'reward': best_reward,
                    'coverage_ratio': current_quality['coverage_ratio'],
                    'coverage_improvement': current_quality['coverage_improvement']
                })
        
        optimization_time = time.time() - start_time
        final_quality = self.reward_function.evaluate_solution_quality(best_positions)
        
        result = {
            'method': 'random_search',
            'best_positions': best_positions,
            'original_positions': self.original_positions,
            'best_reward': best_reward,
            'initial_reward': initial_reward['total_reward'],
            'improvement': best_reward - initial_reward['total_reward'],
            'optimization_time': optimization_time,
            'iterations': max_iterations,
            'quality_report': final_quality,
            'history': history
        }
        
        logger.info(f"随机搜索完成! 用时: {optimization_time:.2f}秒")
        logger.info(f"奖励改善: {result['improvement']:.6f}")
        
        return result


def test_all_modules_quickly():
    """快速测试所有模块"""
    logger.info("=== 开始快速模块测试 ===")
    
    try:
        # 1. 数据预处理测试
        logger.info("\n1. 测试数据预处理...")
        processor = DataProcessor(
            "./populaiton/温州_population_grid.csv",
            "./公交站点shp/0577温州.shp"
        )
        pop_data, bus_data, analysis = processor.get_processed_data()
        logger.info(f"✓ 数据加载成功: {len(pop_data)} 人口点, {len(bus_data)} 站点")
        
        # 2. 空间索引测试
        logger.info("\n2. 测试空间索引...")
        coverage_calc = PopulationCoverageCalculator(pop_data)
        test_positions = bus_data[['longitude', 'latitude']].values[:5]  # 只测试前5个站点
        coverage = coverage_calc.calculate_coverage_ratio_fast(test_positions)
        logger.info(f"✓ 空间索引正常，测试覆盖率: {coverage:.4f}")
        
        # 3. 奖励函数测试
        logger.info("\n3. 测试奖励函数...")
        reward_func = RewardFunction(pop_data, bus_data)
        original_positions = bus_data[['longitude', 'latitude']].values
        reward_dict = reward_func.calculate_reward(original_positions)
        logger.info(f"✓ 奖励函数正常，基线奖励: {reward_dict['total_reward']:.6f}")
        
        # 4. 环境测试（简化版本）
        logger.info("\n4. 测试强化学习环境...")
        env = BusStopOptimizationEnv(
            "./populaiton/温州_population_grid.csv",
            "./公交站点shp/0577温州.shp",
            max_episode_steps=5  # 只测试5步
        )
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        logger.info(f"✓ 环境测试正常，测试奖励: {reward:.6f}")
        env.close()
        
        # 5. 快速优化测试
        logger.info("\n5. 测试快速优化算法...")
        quick_optimizer = QuickOptimizer(
            "./populaiton/温州_population_grid.csv",
            "./公交站点shp/0577温州.shp"
        )
        
        # 梯度优化测试
        result1 = quick_optimizer.gradient_based_optimization(max_iterations=20)
        logger.info(f"✓ 梯度优化完成，改善: {result1['improvement']:.6f}")
        
        # 随机搜索测试  
        result2 = quick_optimizer.random_search_optimization(max_iterations=50)
        logger.info(f"✓ 随机搜索完成，改善: {result2['improvement']:.6f}")
        
        logger.info("\n=== 所有模块测试通过! ===")
        logger.info("系统已准备好进行完整训练")
        
        return {
            'data_test': True,
            'spatial_index_test': True,
            'reward_function_test': True,
            'environment_test': True,
            'gradient_optimization_result': result1,
            'random_search_result': result2
        }
        
    except Exception as e:
        logger.error(f"模块测试失败: {e}")
        raise


if __name__ == "__main__":
    test_all_modules_quickly()