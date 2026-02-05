"""
终极优化器 - 完美版本
集成所有优化，提供最佳性能和结果
"""

import numpy as np
import pandas as pd
from numba import jit, prange
import logging
import time
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

from data_preprocessing import DataProcessor
from acceleration_utils import fast_population_coverage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@jit(nopython=True, cache=True)
def intelligent_initialization(original_positions: np.ndarray,
                             population_size: int,
                             max_move_ratio: float = 0.15,
                             max_move_distance: float = 0.003) -> np.ndarray:
    """
    智能初始化：精确控制移动策略
    - 大部分个体几乎不移动
    - 少数个体进行战略性调整
    """
    n_stops = original_positions.shape[0]
    population = np.zeros((population_size, n_stops, 2))
    
    for i in range(population_size):
        population[i] = original_positions.copy()
        
        # 每个个体只移动10-20%的站点
        n_move = max(1, int(n_stops * (0.1 + np.random.random() * max_move_ratio)))
        move_indices = np.random.choice(n_stops, n_move, replace=False)
        
        for idx in move_indices:
            # 渐进式移动：距离按正态分布
            move_distance = np.abs(np.random.normal(0, max_move_distance * 0.3))
            move_distance = min(move_distance, max_move_distance)
            
            # 随机方向
            angle = np.random.random() * 2 * np.pi
            dx = move_distance * np.cos(angle)
            dy = move_distance * np.sin(angle)
            
            population[i, idx, 0] += dx
            population[i, idx, 1] += dy
    
    return population

@jit(nopython=True, cache=True)
def enhanced_fitness(positions: np.ndarray,
                   original_positions: np.ndarray,
                   pop_points: np.ndarray,
                   pop_weights: np.ndarray,
                   coverage_radius: float) -> float:
    """
    增强适应度函数：多目标平衡优化
    """
    n_stops = positions.shape[0]
    
    # 1. 人口覆盖率 (权重: 5.0)
    coverage = fast_population_coverage(
        positions, pop_points, pop_weights, coverage_radius
    )
    coverage_score = 5.0 * coverage
    
    # 2. 稳定性奖励 (权重: 3.0)
    unmoved_count = 0
    total_movement = 0.0
    significant_moves = 0
    
    for i in range(n_stops):
        dx = positions[i, 0] - original_positions[i, 0]
        dy = positions[i, 1] - original_positions[i, 1]
        movement = np.sqrt(dx * dx + dy * dy)
        
        total_movement += movement
        
        if movement < 0.0001:  # 未移动 (<10米)
            unmoved_count += 1
        elif movement > 0.002:  # 显著移动 (>200米)
            significant_moves += 1
    
    stability_ratio = unmoved_count / n_stops
    stability_score = 3.0 * stability_ratio
    
    # 3. 移动效率奖励 (权重: 1.0)
    # 奖励少量但有效的移动
    if total_movement > 0:
        efficiency = coverage / (total_movement + 1e-8)
        efficiency_score = 1.0 * min(efficiency * 100, 2.0)
    else:
        efficiency_score = 1.0
    
    # 4. 过度移动惩罚 (权重: -2.0)
    movement_penalty = -2.0 * (significant_moves / n_stops)
    
    # 综合适应度
    total_fitness = coverage_score + stability_score + efficiency_score + movement_penalty
    
    return total_fitness

@jit(nopython=True, cache=True)
def elite_selection(fitness_scores: np.ndarray, elite_ratio: float = 0.2) -> np.ndarray:
    """精英选择：保留最优个体"""
    pop_size = len(fitness_scores)
    n_elite = max(1, int(pop_size * elite_ratio))
    
    # 获取最优个体的索引
    elite_indices = np.zeros(n_elite, dtype=np.int32)
    fitness_copy = fitness_scores.copy()
    
    for i in range(n_elite):
        best_idx = np.argmax(fitness_copy)
        elite_indices[i] = best_idx
        fitness_copy[best_idx] = -np.inf  # 标记已选择
    
    return elite_indices

@jit(nopython=True, cache=True)  
def adaptive_crossover(parent1: np.ndarray, parent2: np.ndarray,
                      original_positions: np.ndarray,
                      generation: int, max_generations: int) -> np.ndarray:
    """
    自适应交叉：根据进化阶段调整策略
    """
    child = np.zeros_like(parent1)
    n_stops = parent1.shape[0]
    
    # 自适应参数：早期保守，后期激进
    progress = generation / max_generations
    conservation_prob = 0.8 - 0.3 * progress  # 0.8 -> 0.5
    
    for i in range(n_stops):
        rand_val = np.random.random()
        
        if rand_val < conservation_prob:
            # 保守策略：保持原始位置
            child[i] = original_positions[i]
        elif rand_val < conservation_prob + 0.15:
            # 选择移动较少的父母
            move1 = np.sqrt((parent1[i, 0] - original_positions[i, 0])**2 + 
                          (parent1[i, 1] - original_positions[i, 1])**2)
            move2 = np.sqrt((parent2[i, 0] - original_positions[i, 0])**2 + 
                          (parent2[i, 1] - original_positions[i, 1])**2)
            
            if move1 <= move2:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        else:
            # 随机选择
            if np.random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
    
    return child

@jit(nopython=True, parallel=True, cache=True)
def ultimate_genetic_algorithm(original_positions: np.ndarray,
                             pop_points: np.ndarray,
                             pop_weights: np.ndarray,
                             coverage_radius: float,
                             population_size: int = 80,
                             generations: int = 150) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    终极遗传算法：完美平衡性能与结果
    """
    # 智能初始化
    population = intelligent_initialization(original_positions, population_size)
    
    # 记录进化历史
    fitness_history = np.zeros(generations)
    best_individual = population[0].copy()
    best_fitness = -np.inf
    
    for generation in range(generations):
        # 并行适应度计算
        fitness_scores = np.zeros(population_size)
        
        for i in prange(population_size):
            fitness_scores[i] = enhanced_fitness(
                population[i], original_positions, pop_points, pop_weights, coverage_radius
            )
            
            if fitness_scores[i] > best_fitness:
                best_fitness = fitness_scores[i]
                best_individual = population[i].copy()
        
        fitness_history[generation] = best_fitness
        
        # 精英保留
        elite_indices = elite_selection(fitness_scores, 0.2)
        
        # 新一代生成
        new_population = np.zeros_like(population)
        
        # 保留精英
        for i in range(len(elite_indices)):
            new_population[i] = population[elite_indices[i]]
        
        # 生成剩余个体
        for i in prange(len(elite_indices), population_size):
            # 锦标赛选择
            parent1_idx = tournament_selection_fast(fitness_scores, 5)
            parent2_idx = tournament_selection_fast(fitness_scores, 5)
            
            # 自适应交叉
            child = adaptive_crossover(
                population[parent1_idx], population[parent2_idx], 
                original_positions, generation, generations
            )
            
            # 自适应变异
            if np.random.random() < 0.1:  # 低变异率
                child = precision_mutation(child, original_positions, generation, generations)
            
            new_population[i] = child
        
        population = new_population
        
        # 早停机制
        if generation > 20:
            recent_improvement = fitness_history[generation] - fitness_history[generation-20]
            if recent_improvement < 0.001:  # 20代内改善小于0.001
                break
    
    return best_individual, best_fitness, fitness_history[:generation+1]

@jit(nopython=True, cache=True)
def tournament_selection_fast(fitness_scores: np.ndarray, tournament_size: int) -> int:
    """快速锦标赛选择"""
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
def precision_mutation(individual: np.ndarray,
                      original_positions: np.ndarray,
                      generation: int,
                      max_generations: int) -> np.ndarray:
    """
    精密变异：高度控制的微调操作
    """
    mutated = individual.copy()
    n_stops = individual.shape[0]
    
    # 自适应变异强度：早期大步，后期微调
    progress = generation / max_generations
    base_strength = 0.001 * (1 - 0.7 * progress)  # 0.001 -> 0.0003
    
    # 只对5%的站点进行变异
    n_mutate = max(1, int(n_stops * 0.05))
    
    for _ in range(n_mutate):
        idx = np.random.randint(0, n_stops)
        
        # 高斯变异
        dx = np.random.normal(0, base_strength)
        dy = np.random.normal(0, base_strength)
        
        mutated[idx, 0] += dx
        mutated[idx, 1] += dy
    
    return mutated

class FinalOptimizer:
    """
    终极优化器 - 完美版本
    """
    
    def __init__(self, population_csv_path: str, bus_stops_shp_path: str):
        """初始化优化器"""
        logger.info("🚀 初始化终极优化器...")
        
        # 加载数据
        processor = DataProcessor(population_csv_path, bus_stops_shp_path)
        self.population_data, self.bus_stops_data, _ = processor.get_processed_data()
        
        # 智能采样策略
        self._smart_sampling()
        
        # 数据转换
        self.pop_points = self.population_sample[['longitude', 'latitude']].values
        self.pop_weights = self.population_sample['population'].values
        self.original_positions = self.bus_stops_sample[['longitude', 'latitude']].values
        
        # 优化参数
        self.coverage_radius = 0.008  # 约800米，符合实际公交服务半径
        
        logger.info(f"✅ 优化器初始化完成:")
        logger.info(f"   📊 数据规模: {len(self.pop_points)} 人口点, {len(self.original_positions)} 站点")  
        logger.info(f"   🎯 覆盖半径: {self.coverage_radius * 111.32:.0f}米")
        logger.info(f"   💡 总人口: {self.pop_weights.sum():.0f}")
    
    def _smart_sampling(self):
        """智能采样：保持数据质量的同时提升效率"""
        # 人口数据：按密度和地理分布采样
        if len(self.population_data) > 3000:
            # 分层采样：高密度区域 + 随机采样
            pop_sorted = self.population_data.sort_values('population', ascending=False)
            
            # 高密度区域（前30%）
            high_density = pop_sorted.head(int(len(pop_sorted) * 0.3))
            
            # 其余区域随机采样
            remaining = pop_sorted.tail(int(len(pop_sorted) * 0.7))
            random_sample = remaining.sample(n=min(2000, len(remaining)), random_state=42)
            
            self.population_sample = pd.concat([high_density, random_sample]).reset_index(drop=True)
        else:
            self.population_sample = self.population_data.copy()
        
        # 公交站点：地理均匀采样
        if len(self.bus_stops_data) > 800:
            # 空间网格采样，保持地理分布
            sample_size = min(800, len(self.bus_stops_data))
            self.bus_stops_sample = self.bus_stops_data.sample(
                n=sample_size, random_state=42
            ).reset_index(drop=True)
        else:
            self.bus_stops_sample = self.bus_stops_data.copy()
    
    def optimize(self, mode: str = 'balanced') -> Dict[str, Any]:
        """
        执行优化
        
        Args:
            mode: 优化模式
                - 'fast': 快速模式 (30个体, 50代)
                - 'balanced': 平衡模式 (50个体, 100代) 
                - 'thorough': 彻底模式 (80个体, 150代)
        """
        mode_configs = {
            'fast': {'population_size': 30, 'generations': 50},
            'balanced': {'population_size': 50, 'generations': 100},
            'thorough': {'population_size': 80, 'generations': 150}
        }
        
        config = mode_configs.get(mode, mode_configs['balanced'])
        
        logger.info(f"🎯 开始{mode}模式优化...")
        logger.info(f"   参数: {config['population_size']}个体 × {config['generations']}代")
        
        start_time = time.time()
        
        # 执行遗传算法
        best_positions, best_fitness, fitness_history = ultimate_genetic_algorithm(
            self.original_positions,
            self.pop_points,
            self.pop_weights,
            self.coverage_radius,
            population_size=config['population_size'],
            generations=config['generations']
        )
        
        optimization_time = time.time() - start_time
        
        # 详细分析结果
        result = self._analyze_results(best_positions, best_fitness, optimization_time, fitness_history)
        result['mode'] = mode
        result['config'] = config
        
        # 显示结果
        self._display_results(result)
        
        return result
    
    def _analyze_results(self, best_positions: np.ndarray, best_fitness: float, 
                        optimization_time: float, fitness_history: np.ndarray) -> Dict[str, Any]:
        """详细分析优化结果"""
        # 覆盖率分析
        final_coverage = fast_population_coverage(
            best_positions, self.pop_points, self.pop_weights, self.coverage_radius
        )
        
        # 移动分析  
        n_stops = len(best_positions)
        movement_distances = np.zeros(n_stops)
        moved_stations = 0
        significant_moves = 0
        total_movement = 0.0
        
        for i in range(n_stops):
            dx = best_positions[i, 0] - self.original_positions[i, 0]  
            dy = best_positions[i, 1] - self.original_positions[i, 1]
            distance = np.sqrt(dx * dx + dy * dy)
            movement_distances[i] = distance
            total_movement += distance
            
            if distance > 0.0001:  # >10米
                moved_stations += 1
            if distance > 0.002:   # >200米  
                significant_moves += 1
        
        # 效率指标
        stability_score = 1.0 - (moved_stations / n_stops)
        average_movement_m = (total_movement * 111.32 * 1000) / n_stops
        efficiency_score = final_coverage / (total_movement + 1e-8) if total_movement > 0 else float('inf')
        
        return {
            'best_positions': best_positions,
            'best_fitness': best_fitness,
            'final_coverage': final_coverage,
            'optimization_time': optimization_time,
            'fitness_history': fitness_history,
            'movement_analysis': {
                'total_stations': n_stops,
                'moved_stations': moved_stations,
                'significant_moves': significant_moves,
                'total_movement_km': total_movement * 111.32,
                'average_movement_m': average_movement_m,
                'max_movement_m': np.max(movement_distances) * 111.32 * 1000,
                'stability_score': stability_score
            },
            'performance_metrics': {
                'coverage_improvement': final_coverage - self._calculate_baseline_coverage(),
                'efficiency_score': efficiency_score,
                'convergence_generation': len(fitness_history)
            }
        }
    
    def _calculate_baseline_coverage(self) -> float:
        """计算基线覆盖率"""
        return fast_population_coverage(
            self.original_positions, self.pop_points, self.pop_weights, self.coverage_radius
        )
    
    def _display_results(self, result: Dict[str, Any]) -> None:
        """展示优化结果"""
        logger.info(f"\n🎉 优化完成!")
        logger.info(f"⏱️  用时: {result['optimization_time']:.2f}秒")
        logger.info(f"🏆 最佳适应度: {result['best_fitness']:.4f}")
        logger.info(f"📈 最终覆盖率: {result['final_coverage']:.4f} ({result['final_coverage']*100:.2f}%)")
        
        movement = result['movement_analysis']
        logger.info(f"🚌 移动站点: {movement['moved_stations']}/{movement['total_stations']} ({movement['moved_stations']/movement['total_stations']*100:.1f}%)")
        logger.info(f"📏 平均移动距离: {movement['average_movement_m']:.0f}米")
        logger.info(f"📍 最大移动距离: {movement['max_movement_m']:.0f}米")
        logger.info(f"⭐ 稳定性得分: {movement['stability_score']:.4f}")
        
        perf = result['performance_metrics']
        logger.info(f"📊 覆盖率改善: {perf['coverage_improvement']:.4f}")
        logger.info(f"⚡ 效率得分: {perf['efficiency_score']:.2f}")
        logger.info(f"🔄 收敛代数: {perf['convergence_generation']}")
    
    def create_evolution_plot(self, fitness_history: np.ndarray, save_path: str = "evolution.png") -> None:
        """创建进化过程图表"""
        plt.figure(figsize=(12, 8))
        
        # 适应度进化曲线
        plt.subplot(2, 2, 1)
        plt.plot(fitness_history, 'b-', linewidth=2)
        plt.title('适应度进化曲线', fontsize=14)
        plt.xlabel('代数')
        plt.ylabel('最佳适应度')
        plt.grid(True, alpha=0.3)
        
        # 收敛性分析
        plt.subplot(2, 2, 2)
        if len(fitness_history) > 10:
            improvements = np.diff(fitness_history)
            plt.plot(improvements, 'r-', linewidth=2)
            plt.title('每代改善幅度', fontsize=14)
            plt.xlabel('代数')
            plt.ylabel('适应度改善')
            plt.grid(True, alpha=0.3)
        
        # 滑动平均
        plt.subplot(2, 2, 3)
        if len(fitness_history) > 5:
            window_size = min(10, len(fitness_history) // 3)
            moving_avg = np.convolve(fitness_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(len(fitness_history)), fitness_history, 'lightblue', alpha=0.6, label='原始')
            plt.plot(range(window_size-1, len(fitness_history)), moving_avg, 'darkblue', linewidth=3, label='滑动平均')
            plt.title(f'{window_size}代滑动平均', fontsize=14)
            plt.xlabel('代数')
            plt.ylabel('适应度')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 统计摘要
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.8, f'最终适应度: {fitness_history[-1]:.4f}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.7, f'最大适应度: {np.max(fitness_history):.4f}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.6, f'收敛代数: {len(fitness_history)}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.5, f'总改善: {fitness_history[-1] - fitness_history[0]:.4f}', fontsize=12, transform=plt.gca().transAxes)
        plt.title('优化摘要', fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 进化图表已保存: {save_path}")


def final_test():
    """终极测试"""
    logger.info("🚀 === 终极优化器测试 === 🚀")
    
    try:
        # 创建优化器
        optimizer = FinalOptimizer(
            "./populaiton/温州_population_grid.csv",
            "./公交站点shp/0577温州.shp"
        )
        
        # 测试不同模式
        modes = ['fast', 'balanced']
        results = {}
        
        for mode in modes:
            logger.info(f"\n📋 测试{mode}模式...")
            result = optimizer.optimize(mode=mode)
            results[mode] = result
            
            # 创建进化图表
            optimizer.create_evolution_plot(
                result['fitness_history'], 
                f"evolution_{mode}.png"
            )
        
        # 对比结果
        logger.info(f"\n📊 === 模式对比 ===")
        for mode, result in results.items():
            logger.info(f"{mode:>8s}: 覆盖率{result['final_coverage']:.4f} | "
                       f"用时{result['optimization_time']:.1f}s | "
                       f"移动{result['movement_analysis']['moved_stations']}站点")
        
        # 推荐最佳模式
        best_mode = max(results.keys(), key=lambda x: results[x]['final_coverage'])
        logger.info(f"\n🏆 推荐模式: {best_mode}")
        logger.info(f"🎯 实现了{results[best_mode]['final_coverage']:.2%}的人口覆盖率")
        logger.info(f"⚡ 仅用{results[best_mode]['optimization_time']:.1f}秒完成优化")
        
        return results[best_mode]
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        raise


if __name__ == "__main__":
    final_test()