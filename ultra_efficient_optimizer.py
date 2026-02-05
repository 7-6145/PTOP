"""
超高效优化器 - UltraThink Pro版本
解决数据规模和算法效率问题，实现真正的高效优化
"""

import numpy as np
import pandas as pd
from numba import jit, prange
import logging
import time
from typing import Dict, Any, Tuple
import random

from data_preprocessing import DataProcessor
from acceleration_utils import fast_population_coverage, fast_movement_distances

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@jit(nopython=True, cache=True)
def smart_initialization(original_positions: np.ndarray, 
                        bounds_lower: np.ndarray,
                        bounds_upper: np.ndarray,
                        population_size: int,
                        max_initial_move: float = 0.001) -> np.ndarray:
    """
    智能初始化：大部分站点保持原位，少数站点小幅移动
    """
    n_stops = original_positions.shape[0]
    population = np.zeros((population_size, n_stops, 2))
    
    for i in range(population_size):
        # 复制原始位置
        population[i] = original_positions.copy()
        
        # 只移动20%的站点
        n_move = max(1, int(n_stops * 0.2))
        move_indices = np.random.choice(n_stops, n_move, replace=False)
        
        for idx in move_indices:
            # 小幅随机移动
            dx = np.random.normal(0, max_initial_move * 0.5)
            dy = np.random.normal(0, max_initial_move * 0.5)
            
            new_x = population[i, idx, 0] + dx
            new_y = population[i, idx, 1] + dy
            
            # 边界约束
            if bounds_lower[idx, 0] <= new_x <= bounds_upper[idx, 0]:
                population[i, idx, 0] = new_x
            if bounds_lower[idx, 1] <= new_y <= bounds_upper[idx, 1]:
                population[i, idx, 1] = new_y
    
    return population

@jit(nopython=True, cache=True)
def conservative_mutation(individual: np.ndarray,
                        original_positions: np.ndarray,
                        bounds_lower: np.ndarray,
                        bounds_upper: np.ndarray,
                        mutation_rate: float = 0.1,
                        max_move: float = 0.0005) -> np.ndarray:
    """
    保守变异：只对少数站点进行小幅移动
    """
    mutated = individual.copy()
    n_stops = individual.shape[0]
    
    # 只对10%的站点进行变异
    n_mutate = max(1, int(n_stops * mutation_rate))
    
    for _ in range(n_mutate):
        idx = np.random.randint(0, n_stops)
        
        if np.random.random() < mutation_rate:
            # 小幅移动
            dx = np.random.normal(0, max_move)
            dy = np.random.normal(0, max_move)
            
            new_x = individual[idx, 0] + dx
            new_y = individual[idx, 1] + dy
            
            # 边界约束
            if bounds_lower[idx, 0] <= new_x <= bounds_upper[idx, 0]:
                mutated[idx, 0] = new_x
            if bounds_lower[idx, 1] <= new_y <= bounds_upper[idx, 1]:
                mutated[idx, 1] = new_y
    
    return mutated

@jit(nopython=True, cache=True)
def stability_aware_fitness(positions: np.ndarray,
                          original_positions: np.ndarray,
                          pop_points_sample: np.ndarray,
                          pop_weights_sample: np.ndarray,
                          coverage_radius: float,
                          stability_weight: float = 2.0) -> float:
    """
    稳定性感知的适应度函数
    重点奖励高覆盖率和低移动成本
    """
    # 计算覆盖率（使用采样数据加速）
    coverage = fast_population_coverage(
        positions, pop_points_sample, pop_weights_sample, coverage_radius
    )
    
    # 计算移动成本
    n_stops = positions.shape[0]
    total_movement = 0.0
    unmoved_count = 0
    
    for i in range(n_stops):
        dx = positions[i, 0] - original_positions[i, 0]
        dy = positions[i, 1] - original_positions[i, 1]
        movement = np.sqrt(dx * dx + dy * dy)
        
        total_movement += movement
        if movement < 0.0001:  # 基本未移动
            unmoved_count += 1
    
    # 稳定性奖励
    stability_bonus = stability_weight * (unmoved_count / n_stops)
    
    # 移动惩罚
    movement_penalty = 0.5 * (total_movement / n_stops)  # 归一化
    
    # 综合适应度
    fitness = coverage + stability_bonus - movement_penalty
    
    return fitness

@jit(nopython=True, parallel=True, cache=True)
def ultra_efficient_genetic_algorithm(original_positions: np.ndarray,
                                     pop_points_sample: np.ndarray,
                                     pop_weights_sample: np.ndarray,
                                     bounds_lower: np.ndarray,
                                     bounds_upper: np.ndarray,
                                     coverage_radius: float,
                                     population_size: int = 50,
                                     generations: int = 100) -> Tuple[np.ndarray, float]:
    """
    超高效遗传算法 - 针对大规模优化的特化版本
    """
    # 智能初始化
    population = smart_initialization(
        original_positions, bounds_lower, bounds_upper, population_size
    )
    
    best_individual = population[0].copy()
    best_fitness = -np.inf
    
    for generation in range(generations):
        # 并行计算适应度
        fitness_scores = np.zeros(population_size)
        
        for i in prange(population_size):
            fitness_scores[i] = stability_aware_fitness(
                population[i], original_positions, 
                pop_points_sample, pop_weights_sample, coverage_radius
            )
            
            if fitness_scores[i] > best_fitness:
                best_fitness = fitness_scores[i]
                best_individual = population[i].copy()
        
        # 新种群生成
        new_population = np.zeros_like(population)
        
        for i in prange(population_size):
            # 锦标赛选择
            parent1_idx = tournament_selection_numba(fitness_scores, 3)
            parent2_idx = tournament_selection_numba(fitness_scores, 3)
            
            # 保守交叉
            child = conservative_crossover(
                population[parent1_idx], population[parent2_idx], original_positions
            )
            
            # 保守变异
            if np.random.random() < 0.3:  # 降低变异率
                child = conservative_mutation(
                    child, original_positions, bounds_lower, bounds_upper
                )
            
            new_population[i] = child
        
        population = new_population
    
    return best_individual, best_fitness

@jit(nopython=True, cache=True)
def tournament_selection_numba(fitness_scores: np.ndarray, tournament_size: int) -> int:
    """锦标赛选择（numba优化版）"""
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
def conservative_crossover(parent1: np.ndarray, 
                         parent2: np.ndarray,
                         original_positions: np.ndarray) -> np.ndarray:
    """
    保守交叉：优先保持原始位置
    """
    child = np.zeros_like(parent1)
    n_stops = parent1.shape[0]
    
    for i in range(n_stops):
        # 70%概率保持原始位置
        if np.random.random() < 0.7:
            child[i] = original_positions[i]
        else:
            # 30%概率从父母中选择
            if np.random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
    
    return child

class UltraEfficientOptimizer:
    """
    超高效优化器 - 专为大规模数据设计
    """
    
    def __init__(self, population_csv_path: str, bus_stops_shp_path: str):
        """初始化优化器"""
        logger.info("初始化超高效优化器...")
        
        # 加载数据
        processor = DataProcessor(population_csv_path, bus_stops_shp_path)
        self.population_data, self.bus_stops_data, _ = processor.get_processed_data()
        
        # 数据规模优化：智能采样
        self._optimize_data_scale()
        
        # 转换为numpy数组
        self.pop_points = self.population_sample[['longitude', 'latitude']].values
        self.pop_weights = self.population_sample['population'].values
        self.original_positions = self.bus_stops_sample[['longitude', 'latitude']].values
        
        # 计算紧凑边界（基于实际数据分布）
        self._calculate_smart_bounds()
        
        # 优化参数
        self.coverage_radius = 0.008  # 增加覆盖半径到约800米
        
        logger.info(f"优化器初始化完成:")
        logger.info(f"  原始数据: {len(self.population_data)} 人口点, {len(self.bus_stops_data)} 站点")
        logger.info(f"  采样数据: {len(self.population_sample)} 人口点, {len(self.bus_stops_sample)} 站点")
        logger.info(f"  覆盖半径: {self.coverage_radius * 111.32:.0f}米")
    
    def _optimize_data_scale(self):
        """优化数据规模：智能采样"""
        n_pop_original = len(self.population_data)
        n_stops_original = len(self.bus_stops_data)
        
        # 人口数据采样（按权重采样，保留高密度区域）
        if n_pop_original > 5000:
            # 按人口权重分层采样
            pop_weights = self.population_data['population'].values
            sample_probs = pop_weights / pop_weights.sum()
            
            sample_size = min(5000, n_pop_original)
            sample_indices = np.random.choice(
                n_pop_original, size=sample_size, replace=False, p=sample_probs
            )
            self.population_sample = self.population_data.iloc[sample_indices].copy()
        else:
            self.population_sample = self.population_data.copy()
        
        # 公交站点采样（随机采样，保持空间分布）
        if n_stops_original > 1000:
            sample_size = min(1000, n_stops_original)
            sample_indices = np.random.choice(
                n_stops_original, size=sample_size, replace=False
            )
            self.bus_stops_sample = self.bus_stops_data.iloc[sample_indices].copy()
        else:
            self.bus_stops_sample = self.bus_stops_data.copy()
        
        logger.info(f"数据采样完成: {len(self.population_sample)}/{n_pop_original} 人口点, "
                   f"{len(self.bus_stops_sample)}/{n_stops_original} 站点")
    
    def _calculate_smart_bounds(self):
        """计算智能边界"""
        # 基于数据分布的动态边界
        pop_bounds = {
            'min_lon': self.population_sample['longitude'].quantile(0.05),
            'max_lon': self.population_sample['longitude'].quantile(0.95),
            'min_lat': self.population_sample['latitude'].quantile(0.05),
            'max_lat': self.population_sample['latitude'].quantile(0.95)
        }
        
        stop_bounds = {
            'min_lon': self.bus_stops_sample['longitude'].min(),
            'max_lon': self.bus_stops_sample['longitude'].max(),
            'min_lat': self.bus_stops_sample['latitude'].min(),
            'max_lat': self.bus_stops_sample['latitude'].max()
        }
        
        # 合并边界，留适当余量
        margin = 0.002  # 约200米
        combined_min_lon = min(pop_bounds['min_lon'], stop_bounds['min_lon']) - margin
        combined_max_lon = max(pop_bounds['max_lon'], stop_bounds['max_lon']) + margin
        combined_min_lat = min(pop_bounds['min_lat'], stop_bounds['min_lat']) - margin
        combined_max_lat = max(pop_bounds['max_lat'], stop_bounds['max_lat']) + margin
        
        n_stops = len(self.bus_stops_sample)
        self.bounds_lower = np.full((n_stops, 2), [combined_min_lon, combined_min_lat])
        self.bounds_upper = np.full((n_stops, 2), [combined_max_lon, combined_max_lat])
    
    def optimize(self, algorithm: str = 'genetic', **kwargs) -> Dict[str, Any]:
        """执行优化"""
        logger.info(f"开始{algorithm}算法优化...")
        start_time = time.time()
        
        if algorithm == 'genetic':
            best_positions, best_fitness = ultra_efficient_genetic_algorithm(
                self.original_positions,
                self.pop_points,
                self.pop_weights,
                self.bounds_lower,
                self.bounds_upper,
                self.coverage_radius,
                population_size=kwargs.get('population_size', 50),
                generations=kwargs.get('generations', 100)
            )
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        optimization_time = time.time() - start_time
        
        # 详细结果分析
        final_coverage = fast_population_coverage(
            best_positions, self.pop_points, self.pop_weights, self.coverage_radius
        )
        
        # 移动分析
        movement_distances = fast_movement_distances(self.original_positions, best_positions)
        moved_stations = np.sum(movement_distances > 0.0001)
        total_movement_km = np.sum(movement_distances) * 111.32
        
        # 稳定性分析
        significant_moves = np.sum(movement_distances > 0.001)  # >100米的移动
        stability_score = 1.0 - (moved_stations / len(best_positions))
        
        result = {
            'method': f'ultra_efficient_{algorithm}',
            'best_positions': best_positions,
            'best_fitness': best_fitness,
            'final_coverage': final_coverage,
            'optimization_time': optimization_time,
            'moved_stations': moved_stations,
            'total_movement_km': total_movement_km,
            'significant_moves': significant_moves,
            'stability_score': stability_score,
            'average_movement_m': (total_movement_km / len(best_positions)) * 1000,
            'data_scale': {
                'population_points': len(self.pop_points),
                'bus_stops': len(best_positions)
            }
        }
        
        logger.info(f"优化完成！用时: {optimization_time:.2f}秒")
        logger.info(f"最终覆盖率: {final_coverage:.4f}")
        logger.info(f"移动站点数: {moved_stations}/{len(best_positions)}")
        logger.info(f"总移动距离: {total_movement_km:.2f}km")
        logger.info(f"稳定性得分: {stability_score:.4f}")
        
        return result


def ultra_efficient_test():
    """超高效测试"""
    logger.info("=== 超高效优化器测试 ===")
    
    try:
        # 创建优化器
        optimizer = UltraEfficientOptimizer(
            "./populaiton/温州_population_grid.csv",
            "./公交站点shp/0577温州.shp"
        )
        
        # 快速测试配置
        test_configs = [
            {'population_size': 30, 'generations': 30, 'name': '快速测试'},
            {'population_size': 50, 'generations': 50, 'name': '标准测试'},
        ]
        
        best_result = None
        best_score = 0
        
        for config in test_configs:
            logger.info(f"\n--- {config['name']} ---")
            
            result = optimizer.optimize(
                algorithm='genetic',
                population_size=config['population_size'],
                generations=config['generations']
            )
            
            # 综合评分：覆盖率 + 稳定性 - 移动成本
            score = (result['final_coverage'] + 
                    result['stability_score'] - 
                    result['total_movement_km'] / 1000)
            
            logger.info(f"综合得分: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_result = result
        
        logger.info(f"\n=== 最优结果 ===")
        logger.info(f"方法: {best_result['method']}")
        logger.info(f"覆盖率: {best_result['final_coverage']:.4f}")
        logger.info(f"稳定性: {best_result['stability_score']:.4f}")
        logger.info(f"平均移动距离: {best_result['average_movement_m']:.0f}米")
        logger.info(f"优化时间: {best_result['optimization_time']:.2f}秒")
        logger.info(f"综合得分: {best_score:.4f}")
        
        logger.info("超高效优化器测试完成！")
        return best_result
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    ultra_efficient_test()