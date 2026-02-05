"""
超高速优化器 - UltraThink级别
使用numba并行化 + 向量化操作 + 高效搜索算法
"""

import numpy as np
import pandas as pd
from numba import jit, prange
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, Any, Tuple

from data_preprocessing import DataProcessor
from acceleration_utils import (
    fast_population_coverage, 
    fast_batch_reward_calculation,
    fast_movement_distances,
    fast_boundary_check,
    fast_min_distance_check
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@jit(nopython=True, parallel=True, cache=True)
def parallel_genetic_algorithm(population: np.ndarray,
                              pop_points: np.ndarray,
                              pop_weights: np.ndarray,
                              original_positions: np.ndarray,
                              bounds_lower: np.ndarray,
                              bounds_upper: np.ndarray,
                              coverage_radius: float,
                              movement_weight: float,
                              mutation_rate: float,
                              generations: int) -> Tuple[np.ndarray, float]:
    """
    并行遗传算法 - 超高速优化核心
    """
    pop_size = population.shape[0]
    n_stops = population.shape[1]
    
    # 预分配内存
    fitness_scores = np.zeros(pop_size)
    new_population = np.zeros_like(population)
    
    best_individual = population[0].copy()
    best_fitness = -np.inf
    
    for generation in range(generations):
        # 并行计算适应度
        for i in prange(pop_size):
            individual = population[i]
            
            # 计算覆盖率
            coverage = fast_population_coverage(
                individual, pop_points, pop_weights, coverage_radius
            )
            
            # 计算移动成本
            movement_distances = np.zeros(n_stops)
            for j in range(n_stops):
                dx = individual[j, 0] - original_positions[j, 0]
                dy = individual[j, 1] - original_positions[j, 1]
                movement_distances[j] = np.sqrt(dx * dx + dy * dy)
            
            total_movement = np.sum(movement_distances)
            
            # 适应度 = 覆盖率 - 移动惩罚
            fitness_scores[i] = coverage - movement_weight * total_movement
            
            # 更新全局最优
            if fitness_scores[i] > best_fitness:
                best_fitness = fitness_scores[i]
                best_individual = individual.copy()
        
        # 选择、交叉、变异（并行）
        for i in prange(pop_size):
            # 锦标赛选择
            parent1_idx = tournament_selection(fitness_scores, 3)
            parent2_idx = tournament_selection(fitness_scores, 3)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # 交叉
            child = crossover(parent1, parent2)
            
            # 变异
            if np.random.random() < mutation_rate:
                child = mutate(child, bounds_lower, bounds_upper, 0.001)
            
            # 边界约束
            child = clip_to_bounds(child, bounds_lower, bounds_upper)
            
            new_population[i] = child
        
        # 更新种群
        population = new_population.copy()
    
    return best_individual, best_fitness

@jit(nopython=True, cache=True)
def tournament_selection(fitness_scores: np.ndarray, tournament_size: int) -> int:
    """锦标赛选择"""
    pop_size = len(fitness_scores)
    best_idx = np.random.randint(0, pop_size)
    best_fitness = fitness_scores[best_idx]
    
    for _ in range(tournament_size - 1):
        candidate_idx = np.random.randint(0, pop_size)
        if fitness_scores[candidate_idx] > best_fitness:
            best_idx = candidate_idx
            best_fitness = fitness_scores[candidate_idx]
    
    return best_idx

@jit(nopython=True, cache=True)
def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """均匀交叉"""
    child = np.zeros_like(parent1)
    
    for i in range(parent1.shape[0]):
        for j in range(parent1.shape[1]):
            if np.random.random() < 0.5:
                child[i, j] = parent1[i, j]
            else:
                child[i, j] = parent2[i, j]
    
    return child

@jit(nopython=True, cache=True)
def mutate(individual: np.ndarray, 
           bounds_lower: np.ndarray, 
           bounds_upper: np.ndarray, 
           mutation_strength: float) -> np.ndarray:
    """高斯变异"""
    mutated = individual.copy()
    
    for i in range(individual.shape[0]):
        for j in range(individual.shape[1]):
            if np.random.random() < 0.1:  # 10%的基因发生变异
                noise = np.random.normal(0, mutation_strength)
                mutated[i, j] = individual[i, j] + noise
    
    return mutated

@jit(nopython=True, cache=True)
def clip_to_bounds(positions: np.ndarray, 
                  bounds_lower: np.ndarray, 
                  bounds_upper: np.ndarray) -> np.ndarray:
    """边界约束"""
    clipped = positions.copy()
    
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            if clipped[i, j] < bounds_lower[i, j]:
                clipped[i, j] = bounds_lower[i, j]
            elif clipped[i, j] > bounds_upper[i, j]:
                clipped[i, j] = bounds_upper[i, j]
    
    return clipped

@jit(nopython=True, parallel=True, cache=True)
def parallel_simulated_annealing(initial_solution: np.ndarray,
                               pop_points: np.ndarray,
                               pop_weights: np.ndarray,
                               original_positions: np.ndarray,
                               bounds_lower: np.ndarray,
                               bounds_upper: np.ndarray,
                               coverage_radius: float,
                               movement_weight: float,
                               n_chains: int,
                               iterations_per_chain: int) -> Tuple[np.ndarray, float]:
    """
    并行模拟退火算法
    """
    n_stops = initial_solution.shape[0]
    
    # 并行运行多条退火链
    chain_results = np.zeros((n_chains, n_stops, 2))
    chain_scores = np.zeros(n_chains)
    
    for chain_idx in prange(n_chains):
        current_solution = initial_solution.copy()
        
        # 添加随机扰动初始化不同链
        for i in range(n_stops):
            for j in range(2):
                perturbation = np.random.normal(0, 0.0005)
                current_solution[i, j] += perturbation
        
        # 边界约束
        current_solution = clip_to_bounds(current_solution, bounds_lower, bounds_upper)
        
        # 计算初始能量
        current_coverage = fast_population_coverage(
            current_solution, pop_points, pop_weights, coverage_radius
        )
        movement_distances = np.zeros(n_stops)
        for i in range(n_stops):
            dx = current_solution[i, 0] - original_positions[i, 0]
            dy = current_solution[i, 1] - original_positions[i, 1]
            movement_distances[i] = np.sqrt(dx * dx + dy * dy)
        
        current_energy = current_coverage - movement_weight * np.sum(movement_distances)
        best_solution = current_solution.copy()
        best_energy = current_energy
        
        # 退火参数
        initial_temp = 0.1
        final_temp = 0.001
        cooling_rate = np.power(final_temp / initial_temp, 1.0 / iterations_per_chain)
        temperature = initial_temp
        
        # 退火迭代
        for iteration in range(iterations_per_chain):
            # 生成邻域解
            neighbor = current_solution.copy()
            
            # 随机选择一个站点进行移动
            station_idx = np.random.randint(0, n_stops)
            move_x = np.random.normal(0, 0.0002)
            move_y = np.random.normal(0, 0.0002)
            
            neighbor[station_idx, 0] += move_x
            neighbor[station_idx, 1] += move_y
            
            # 边界约束
            neighbor = clip_to_bounds(neighbor, bounds_lower, bounds_upper)
            
            # 计算邻域解能量
            neighbor_coverage = fast_population_coverage(
                neighbor, pop_points, pop_weights, coverage_radius
            )
            neighbor_movement_distances = np.zeros(n_stops)
            for i in range(n_stops):
                dx = neighbor[i, 0] - original_positions[i, 0]
                dy = neighbor[i, 1] - original_positions[i, 1]
                neighbor_movement_distances[i] = np.sqrt(dx * dx + dy * dy)
            
            neighbor_energy = neighbor_coverage - movement_weight * np.sum(neighbor_movement_distances)
            
            # 接受准则
            energy_diff = neighbor_energy - current_energy
            
            if energy_diff > 0 or np.random.random() < np.exp(energy_diff / temperature):
                current_solution = neighbor
                current_energy = neighbor_energy
                
                if neighbor_energy > best_energy:
                    best_solution = neighbor.copy()
                    best_energy = neighbor_energy
            
            # 降温
            temperature *= cooling_rate
        
        chain_results[chain_idx] = best_solution
        chain_scores[chain_idx] = best_energy
    
    # 找到所有链中的最佳解
    best_chain_idx = np.argmax(chain_scores)
    return chain_results[best_chain_idx], chain_scores[best_chain_idx]

class UltraFastOptimizer:
    """
    超高速优化器主类
    """
    
    def __init__(self, population_csv_path: str, bus_stops_shp_path: str):
        """初始化优化器"""
        logger.info("初始化超高速优化器...")
        
        # 加载数据
        processor = DataProcessor(population_csv_path, bus_stops_shp_path)
        self.population_data, self.bus_stops_data, _ = processor.get_processed_data()
        
        # 转换为numpy数组（numba友好）
        self.pop_points = self.population_data[['longitude', 'latitude']].values
        self.pop_weights = self.population_data['population'].values
        self.original_positions = self.bus_stops_data[['longitude', 'latitude']].values
        
        # 计算边界
        margin = 0.005
        min_lon = self.pop_points[:, 0].min() - margin
        max_lon = self.pop_points[:, 0].max() + margin
        min_lat = self.pop_points[:, 1].min() - margin
        max_lat = self.pop_points[:, 1].max() + margin
        
        n_stops = len(self.original_positions)
        self.bounds_lower = np.full((n_stops, 2), [min_lon, min_lat])
        self.bounds_upper = np.full((n_stops, 2), [max_lon, max_lat])
        
        # 优化参数
        self.coverage_radius = 0.005
        self.movement_weight = 0.3
        
        logger.info(f"优化器初始化完成: {n_stops} 站点, {len(self.pop_points)} 人口点")
    
    def genetic_algorithm_optimize(self, pop_size: int = 100, generations: int = 50) -> Dict[str, Any]:
        """遗传算法优化"""
        logger.info(f"开始遗传算法优化: 种群大小={pop_size}, 代数={generations}")
        start_time = time.time()
        
        # 初始化种群
        n_stops = len(self.original_positions)
        population = np.zeros((pop_size, n_stops, 2))
        
        # 种群初始化：在原位置附近生成
        for i in range(pop_size):
            for j in range(n_stops):
                noise = np.random.normal(0, 0.001, 2)
                population[i, j] = self.original_positions[j] + noise
        
        # 边界约束
        for i in range(pop_size):
            population[i] = clip_to_bounds(population[i], self.bounds_lower, self.bounds_upper)
        
        # 运行遗传算法
        best_positions, best_fitness = parallel_genetic_algorithm(
            population, self.pop_points, self.pop_weights, self.original_positions,
            self.bounds_lower, self.bounds_upper, self.coverage_radius, 
            self.movement_weight, 0.05, generations
        )
        
        optimization_time = time.time() - start_time
        
        # 计算详细结果
        final_coverage = fast_population_coverage(
            best_positions, self.pop_points, self.pop_weights, self.coverage_radius
        )
        
        movement_distances = fast_movement_distances(self.original_positions, best_positions)
        
        result = {
            'method': 'parallel_genetic_algorithm',
            'best_positions': best_positions,
            'best_fitness': best_fitness,
            'final_coverage': final_coverage,
            'total_movement_km': np.sum(movement_distances) * 111.32,
            'moved_stations': np.sum(movement_distances > 0.0001),
            'optimization_time': optimization_time,
            'generations': generations,
            'population_size': pop_size
        }
        
        logger.info(f"遗传算法完成: 用时 {optimization_time:.2f}秒")
        logger.info(f"最佳适应度: {best_fitness:.6f}")
        logger.info(f"覆盖率: {final_coverage:.4f}")
        
        return result
    
    def simulated_annealing_optimize(self, n_chains: int = 8, iterations_per_chain: int = 1000) -> Dict[str, Any]:
        """并行模拟退火优化"""
        logger.info(f"开始并行模拟退火: {n_chains} 条链, 每链 {iterations_per_chain} 迭代")
        start_time = time.time()
        
        # 运行并行模拟退火
        best_positions, best_energy = parallel_simulated_annealing(
            self.original_positions, self.pop_points, self.pop_weights,
            self.original_positions, self.bounds_lower, self.bounds_upper,
            self.coverage_radius, self.movement_weight, n_chains, iterations_per_chain
        )
        
        optimization_time = time.time() - start_time
        
        # 计算详细结果
        final_coverage = fast_population_coverage(
            best_positions, self.pop_points, self.pop_weights, self.coverage_radius
        )
        
        movement_distances = fast_movement_distances(self.original_positions, best_positions)
        
        result = {
            'method': 'parallel_simulated_annealing',
            'best_positions': best_positions,
            'best_energy': best_energy,
            'final_coverage': final_coverage,
            'total_movement_km': np.sum(movement_distances) * 111.32,
            'moved_stations': np.sum(movement_distances > 0.0001),
            'optimization_time': optimization_time,
            'chains': n_chains,
            'iterations_per_chain': iterations_per_chain
        }
        
        logger.info(f"模拟退火完成: 用时 {optimization_time:.2f}秒")
        logger.info(f"最佳能量: {best_energy:.6f}")
        logger.info(f"覆盖率: {final_coverage:.4f}")
        
        return result


def ultra_fast_test():
    """超高速测试"""
    logger.info("=== 超高速优化器测试 ===")
    
    try:
        # 创建优化器
        optimizer = UltraFastOptimizer(
            "./populaiton/温州_population_grid.csv",
            "./公交站点shp/0577温州.shp"
        )
        
        # 测试1: 遗传算法（小规模快速测试）
        logger.info("\n测试1: 并行遗传算法")
        ga_result = optimizer.genetic_algorithm_optimize(pop_size=50, generations=20)
        
        logger.info(f"遗传算法结果:")
        logger.info(f"  用时: {ga_result['optimization_time']:.2f}秒")
        logger.info(f"  覆盖率: {ga_result['final_coverage']:.4f}")
        logger.info(f"  移动总距离: {ga_result['total_movement_km']:.3f}km")
        logger.info(f"  移动站点数: {ga_result['moved_stations']}")
        
        # 测试2: 并行模拟退火
        logger.info("\n测试2: 并行模拟退火")
        sa_result = optimizer.simulated_annealing_optimize(n_chains=4, iterations_per_chain=500)
        
        logger.info(f"模拟退火结果:")
        logger.info(f"  用时: {sa_result['optimization_time']:.2f}秒")
        logger.info(f"  覆盖率: {sa_result['final_coverage']:.4f}")
        logger.info(f"  移动总距离: {sa_result['total_movement_km']:.3f}km")
        logger.info(f"  移动站点数: {sa_result['moved_stations']}")
        
        # 比较结果
        logger.info(f"\n=== 算法比较 ===")
        logger.info(f"遗传算法: 覆盖率 {ga_result['final_coverage']:.4f}, 用时 {ga_result['optimization_time']:.2f}秒")
        logger.info(f"模拟退火: 覆盖率 {sa_result['final_coverage']:.4f}, 用时 {sa_result['optimization_time']:.2f}秒")
        
        # 确定最佳算法
        if ga_result['final_coverage'] > sa_result['final_coverage']:
            logger.info("遗传算法表现更好!")
            best_result = ga_result
        else:
            logger.info("模拟退火表现更好!")
            best_result = sa_result
        
        logger.info("超高速优化器测试完成!")
        return best_result
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    ultra_fast_test()