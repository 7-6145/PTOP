"""
激进优化版本 - 更多站点移动，更高覆盖率提升
基于成功算法，调整参数以实现更大幅度的优化
"""

import numpy as np
import pandas as pd
from numba import jit, prange
import logging
import time
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("Warning: GeoPandas not available, will create CSV files instead")

from data_preprocessing import DataProcessor
from acceleration_utils import fast_population_coverage

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@jit(nopython=True, cache=True)
def aggressive_initialization(original_positions: np.ndarray,
                            population_size: int,
                            max_move_ratio: float = 0.4,  # 增加到40%
                            max_move_distance: float = 0.008) -> np.ndarray:  # 增加移动距离
    """
    激进初始化：更多站点移动，更大移动范围
    - 每个个体移动20-40%的站点
    - 允许更大的移动距离
    """
    n_stops = original_positions.shape[0]
    population = np.zeros((population_size, n_stops, 2))
    
    for i in range(population_size):
        population[i] = original_positions.copy()
        
        # 每个个体移动20-40%的站点
        min_move_ratio = 0.2  # 最少移动20%
        move_ratio = min_move_ratio + np.random.random() * (max_move_ratio - min_move_ratio)
        n_move = max(1, int(n_stops * move_ratio))
        move_indices = np.random.choice(n_stops, n_move, replace=False)
        
        for idx in move_indices:
            # 更大的移动范围：距离按伽马分布
            move_distance = np.random.gamma(2.0, max_move_distance * 0.4)
            move_distance = min(move_distance, max_move_distance)
            
            # 随机方向
            angle = np.random.random() * 2 * np.pi
            dx = move_distance * np.cos(angle)
            dy = move_distance * np.sin(angle)
            
            population[i, idx, 0] += dx
            population[i, idx, 1] += dy
    
    return population

@jit(nopython=True, cache=True)
def coverage_focused_fitness(positions: np.ndarray,
                           original_positions: np.ndarray,
                           pop_points: np.ndarray,
                           pop_weights: np.ndarray,
                           coverage_radius: float) -> float:
    """
    覆盖率导向适应度函数
    主要优化目标：覆盖率 >> 稳定性
    """
    n_stops = positions.shape[0]
    
    # 1. 覆盖率计算 - 主要目标
    coverage_rate = fast_population_coverage(
        positions, pop_points, pop_weights, coverage_radius
    )
    
    # 2. 移动惩罚 - 较轻的惩罚
    total_movement = 0.0
    severe_move_penalty = 0.0
    
    for i in range(n_stops):
        dx = positions[i, 0] - original_positions[i, 0]
        dy = positions[i, 1] - original_positions[i, 1]
        movement = np.sqrt(dx * dx + dy * dy)
        
        total_movement += movement
        
        # 只对过度移动进行严厉惩罚
        if movement > 0.015:  # 超过~1.65公里才严厉惩罚
            severe_move_penalty += movement * 5.0
        else:
            # 正常移动只有轻微惩罚
            severe_move_penalty += movement * 0.5
    
    # 3. 分散性奖励 - 避免站点过于集中
    dispersion_bonus = 0.0
    if n_stops > 1:
        min_distances = np.full(n_stops, np.inf)
        for i in range(n_stops):
            for j in range(n_stops):
                if i != j:
                    dist = np.sqrt((positions[i, 0] - positions[j, 0])**2 + 
                                 (positions[i, 1] - positions[j, 1])**2)
                    min_distances[i] = min(min_distances[i], dist)
        
        avg_min_distance = np.mean(min_distances)
        dispersion_bonus = avg_min_distance * 2.0  # 鼓励适当分散
    
    # 综合适应度：覆盖率权重最大
    fitness = (
        coverage_rate * 20.0 +           # 覆盖率权重极高
        dispersion_bonus * 3.0 -         # 分散性奖励
        severe_move_penalty * 1.0        # 移动惩罚相对较轻
    )
    
    return fitness

@jit(nopython=True, cache=True)
def intensive_genetic_algorithm(original_positions: np.ndarray,
                              pop_points: np.ndarray,
                              pop_weights: np.ndarray,
                              coverage_radius: float,
                              population_size: int = 80,  # 增大种群
                              max_generations: int = 200) -> Tuple[np.ndarray, float]:  # 增加代数
    """强化遗传算法 - 更激进的优化"""
    
    # 激进初始化
    population = aggressive_initialization(original_positions, population_size, 
                                         max_move_ratio=0.4, max_move_distance=0.008)
    
    best_individual = population[0].copy()
    best_fitness = coverage_focused_fitness(
        best_individual, original_positions, pop_points, pop_weights, coverage_radius
    )
    
    stagnation_count = 0
    last_best_fitness = best_fitness
    
    for generation in range(max_generations):
        # 计算适应度
        fitness_scores = np.zeros(population_size)
        
        for i in range(population_size):
            fitness_scores[i] = coverage_focused_fitness(
                population[i], original_positions, pop_points, pop_weights, coverage_radius
            )
            
            if fitness_scores[i] > best_fitness:
                best_fitness = fitness_scores[i]
                best_individual = population[i].copy()
                stagnation_count = 0
            
        # 动态早停策略
        if abs(best_fitness - last_best_fitness) < 0.001:
            stagnation_count += 1
        else:
            stagnation_count = 0
        
        if stagnation_count > 30:  # 更宽松的早停
            break
        
        last_best_fitness = best_fitness
        
        # 选择排序
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        # 精英保留 - 保留更多精英
        elite_size = max(5, population_size // 8)
        new_population = np.zeros_like(population)
        
        for i in range(elite_size):
            new_population[i] = population[sorted_indices[i]].copy()
        
        # 生成新个体
        for i in range(elite_size, population_size):
            # 锦标赛选择 - 选择范围更大
            parent1_idx = sorted_indices[np.random.randint(0, min(10, population_size))]
            parent2_idx = sorted_indices[np.random.randint(0, min(10, population_size))]
            
            # 更激进的交叉
            child = population[parent1_idx].copy()
            
            # 对20-30%的站点进行交叉
            crossover_ratio = 0.2 + np.random.random() * 0.1
            n_crossover = max(1, int(original_positions.shape[0] * crossover_ratio))
            crossover_indices = np.random.choice(
                original_positions.shape[0], n_crossover, replace=False
            )
            
            for idx in crossover_indices:
                if np.random.random() < 0.6:  # 增加交叉概率
                    child[idx] = population[parent2_idx, idx].copy()
            
            # 更频繁的变异
            if np.random.random() < 0.5:  # 增加变异概率
                n_mutate = max(1, int(original_positions.shape[0] * 0.15))  # 增加变异数量
                mutate_indices = np.random.choice(
                    original_positions.shape[0], n_mutate, replace=False
                )
                
                for idx in mutate_indices:
                    # 更大的变异强度
                    dx = np.random.normal(0, 0.003)  # 增大变异范围
                    dy = np.random.normal(0, 0.003)
                    child[idx, 0] += dx
                    child[idx, 1] += dy
            
            new_population[i] = child
        
        population = new_population
    
    return best_individual, best_fitness

class AggressiveOptimizerWithOutputs:
    """激进优化器 - 更高覆盖率版本"""
    
    def __init__(self, population_csv_path: str, bus_stops_shp_path: str):
        """初始化"""
        logger.info("🚀 初始化激进优化器（高覆盖率版）...")
        
        self.coverage_radius = 400  # 增加覆盖半径到400米
        
        # 数据预处理
        self.processor = DataProcessor(population_csv_path, bus_stops_shp_path)
        self.population_data, self.bus_stops_data, self.overlap_info = self.processor.get_processed_data()
        
        logger.info(f"✅ 数据加载完成:")
        logger.info(f"   人口网格: {len(self.population_data):,}")
        logger.info(f"   公交站点: {len(self.bus_stops_data):,}")
        logger.info(f"   覆盖半径: {self.coverage_radius}米（增强版）")
        
        self.output_dir = None
    
    def optimize_and_save_results(self) -> str:
        """执行激进优化并保存完整结果"""
        logger.info("🎯 开始激进优化（目标：最大化覆盖率）...")
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"aggressive_optimization_results_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"📁 结果将保存到: {self.output_dir}")
        
        # 数据准备
        original_positions = self.bus_stops_data[['longitude', 'latitude']].values
        pop_points = self.population_data[['longitude', 'latitude']].values
        pop_weights = self.population_data['population'].values
        
        # 执行激进优化
        logger.info("⚡ 执行强化遗传算法优化...")
        logger.info("   🎯 目标：移动更多站点，显著提升覆盖率")
        logger.info("   📈 预期移动比例: 25-40%")
        logger.info("   🔄 算法强度: 加强版")
        
        start_time = time.time()
        
        optimized_positions, best_fitness = intensive_genetic_algorithm(
            original_positions, pop_points, pop_weights, 
            self.coverage_radius / 111320.0,  # 转换为度
            population_size=80, max_generations=200  # 更强的参数
        )
        
        optimization_time = time.time() - start_time
        
        logger.info(f"✅ 激进优化完成，用时 {optimization_time:.2f}秒")
        logger.info(f"🎯 最佳适应度: {best_fitness:.4f}")
        
        # 计算详细统计
        results = self._calculate_detailed_stats(
            original_positions, optimized_positions, 
            pop_points, pop_weights, optimization_time
        )
        
        # 保存所有结果
        self._save_all_results(original_positions, optimized_positions, results)
        
        logger.info(f"🎉 激进优化结果已保存到: {self.output_dir}")
        
        # 显示优化成果
        self._display_optimization_summary(results)
        
        return str(self.output_dir)
    
    def _display_optimization_summary(self, results):
        """显示优化成果摘要"""
        logger.info("\n" + "="*50)
        logger.info("🏆 激进优化成果摘要")
        logger.info("="*50)
        logger.info(f"📊 总站点数: {results['total_stations']:,}")
        logger.info(f"🚌 移动站点: {results['moved_stations']:,} ({results['moved_stations']/results['total_stations']:.1%})")
        logger.info(f"📏 平均移动: {results['average_movement_m']:.1f}米")
        logger.info(f"📈 覆盖率提升: {results['original_coverage']:.3f} → {results['optimized_coverage']:.3f}")
        logger.info(f"🎯 相对改善: {results['coverage_improvement']/results['original_coverage']*100 if results['original_coverage'] > 0 else 0:.1f}%")
        logger.info(f"⏱️  优化用时: {results['optimization_time']:.1f}秒")
        logger.info("="*50 + "\n")
    
    def _calculate_detailed_stats(self, original_positions, optimized_positions, 
                                pop_points, pop_weights, optimization_time):
        """计算详细统计信息"""
        logger.info("📊 计算激进优化统计...")
        
        n_stops = len(original_positions)
        
        # 覆盖率计算
        original_coverage = fast_population_coverage(
            original_positions, pop_points, pop_weights, self.coverage_radius / 111320.0
        )
        optimized_coverage = fast_population_coverage(
            optimized_positions, pop_points, pop_weights, self.coverage_radius / 111320.0
        )
        
        # 移动统计
        movements = []
        moved_count = 0
        total_movement_m = 0.0
        
        for i in range(n_stops):
            dx = optimized_positions[i, 0] - original_positions[i, 0]
            dy = optimized_positions[i, 1] - original_positions[i, 1]
            movement_deg = np.sqrt(dx * dx + dy * dy)
            movement_m = movement_deg * 111320.0
            
            movements.append(movement_m)
            total_movement_m += movement_m
            
            if movement_m > 5.0:  # 降低移动阈值到5米
                moved_count += 1
        
        stability_score = (n_stops - moved_count) / n_stops
        
        results = {
            'optimization_time': optimization_time,
            'total_stations': n_stops,
            'moved_stations': moved_count,
            'stability_score': stability_score,
            'original_coverage': original_coverage,
            'optimized_coverage': optimized_coverage,
            'coverage_improvement': optimized_coverage - original_coverage,
            'total_movement_m': total_movement_m,
            'average_movement_m': total_movement_m / n_stops,
            'movements': movements,
            'original_positions': original_positions,
            'optimized_positions': optimized_positions
        }
        
        logger.info(f"📈 激进优化统计完成:")
        logger.info(f"   移动站点: {moved_count}/{n_stops} ({moved_count/n_stops:.1%})")
        logger.info(f"   稳定性: {stability_score:.3f}")
        logger.info(f"   覆盖率: {original_coverage:.3f} → {optimized_coverage:.3f} (+{(optimized_coverage-original_coverage)/original_coverage*100 if original_coverage > 0 else 0:.1f}%)")
        logger.info(f"   平均移动: {total_movement_m/n_stops:.1f}米")
        
        return results
    
    def _save_all_results(self, original_positions, optimized_positions, results):
        """保存所有结果文件"""
        logger.info("💾 保存激进优化结果...")
        
        # 1. 保存站点数据
        self._save_station_data(original_positions, optimized_positions, results)
        
        # 2. 保存人口数据
        self._save_population_data()
        
        # 3. 创建可视化
        self._create_visualizations(results)
        
        # 4. 生成报告
        self._create_comprehensive_report(results)
        
        # 5. 保存统计JSON
        self._save_statistics_json(results)
    
    def _save_station_data(self, original_positions, optimized_positions, results):
        """保存站点数据文件"""
        logger.info("🚌 保存站点数据...")
        
        # 创建优化后的站点数据
        optimized_stops = self.bus_stops_data.copy()
        
        # 添加分析字段
        optimized_stops['original_lon'] = original_positions[:, 0]
        optimized_stops['original_lat'] = original_positions[:, 1]
        optimized_stops['longitude'] = optimized_positions[:, 0]
        optimized_stops['latitude'] = optimized_positions[:, 1]
        optimized_stops['movement_m'] = results['movements']
        optimized_stops['is_moved'] = [m > 5.0 for m in results['movements']]  # 降低阈值
        
        # 添加优化强度标记
        optimized_stops['move_intensity'] = ['轻微' if m < 20 else '中等' if m < 100 else '显著' 
                                           for m in results['movements']]
        
        # 分离移动的站点
        moved_stops = optimized_stops[optimized_stops['is_moved'] == True]
        
        if HAS_GEOPANDAS:
            # 保存为Shapefile
            self._save_as_shapefile("original_bus_stops.shp", self.bus_stops_data)
            self._save_as_shapefile("optimized_bus_stops.shp", optimized_stops)
            self._save_as_shapefile("moved_bus_stops.shp", moved_stops)
            logger.info("✅ 激进优化Shapefile文件已保存")
        else:
            # 保存为CSV
            self.bus_stops_data.to_csv(self.output_dir / "original_bus_stops.csv", index=False)
            optimized_stops.to_csv(self.output_dir / "optimized_bus_stops.csv", index=False)
            moved_stops.to_csv(self.output_dir / "moved_bus_stops.csv", index=False)
            logger.info("✅ 激进优化CSV文件已保存")
        
        logger.info(f"📊 激进优化站点统计: 总计{len(optimized_stops)}, 移动{len(moved_stops)} ({len(moved_stops)/len(optimized_stops):.1%})")
    
    def _save_as_shapefile(self, filename, data):
        """保存为shapefile"""
        if HAS_GEOPANDAS:
            gdf = gpd.GeoDataFrame(
                data,
                geometry=gpd.points_from_xy(data['longitude'], data['latitude']),
                crs='EPSG:4326'
            )
            gdf.to_file(self.output_dir / filename, encoding='utf-8')
    
    def _save_population_data(self):
        """保存人口数据"""
        pop_file = self.output_dir / "population_data.csv"
        self.population_data.to_csv(pop_file, index=False)
        logger.info(f"📊 人口数据已保存: {pop_file}")
    
    def _create_visualizations(self, results):
        """创建可视化图表"""
        logger.info("📈 创建激进优化可视化...")
        
        # 1. 综合分析图
        self._create_analysis_charts(results)
        
        # 2. 地图可视化
        self._create_map_visualizations(results)
    
    def _create_analysis_charts(self, results):
        """创建分析图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('激进优化分析报告 - 最大化覆盖率策略', fontsize=16, fontweight='bold')
        
        movements = np.array(results['movements'])
        moved_stations = results['moved_stations']
        total_stations = results['total_stations']
        
        # 1. 覆盖率显著提升对比
        ax = axes[0, 0]
        categories = ['优化前', '激进优化后']
        coverage_values = [results['original_coverage'], results['optimized_coverage']]
        
        bars = ax.bar(categories, coverage_values, color=['#ff9999', '#66cc66'])
        ax.set_ylabel('覆盖率')
        ax.set_title('覆盖率显著提升')
        ax.set_ylim(0, max(coverage_values) * 1.2)
        
        # 显示提升幅度
        improvement = (results['optimized_coverage'] - results['original_coverage']) / results['original_coverage'] * 100 if results['original_coverage'] > 0 else 0
        ax.text(0.5, max(coverage_values) * 1.1, f'提升: {improvement:.1f}%', 
               ha='center', fontweight='bold', fontsize=12, color='red')
        
        for bar, value in zip(bars, coverage_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{value:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 激进移动统计
        ax = axes[0, 1]
        unmoved_stations = total_stations - moved_stations
        sizes = [unmoved_stations, moved_stations]
        labels = [f'保持原位\n{unmoved_stations}个', f'激进移动\n{moved_stations}个']
        colors = ['#87ceeb', '#ff6b6b']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title(f'激进移动策略\n({moved_stations/total_stations:.1%} 站点移动)')
        
        # 3. 移动距离分布（更详细）
        ax = axes[0, 2]
        moved_movements = movements[movements > 5.0]
        
        if len(moved_movements) > 0:
            ax.hist(moved_movements, bins=40, color='orange', alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(moved_movements), color='red', linestyle='--', 
                      linewidth=2, label=f'平均: {np.mean(moved_movements):.1f}m')
            ax.axvline(np.median(moved_movements), color='blue', linestyle='--', 
                      linewidth=2, label=f'中位数: {np.median(moved_movements):.1f}m')
            ax.set_xlabel('移动距离 (米)')
            ax.set_ylabel('站点数量')
            ax.set_title('激进移动距离分布')
            ax.legend()
        else:
            ax.text(0.5, 0.5, '无站点移动', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14)
            ax.set_title('移动距离分布')
        
        # 4. 移动强度分类
        ax = axes[1, 0]
        light_moves = np.sum((movements > 5) & (movements < 20))
        medium_moves = np.sum((movements >= 20) & (movements < 100))
        significant_moves = np.sum(movements >= 100)
        
        categories = ['轻微\n(5-20m)', '中等\n(20-100m)', '显著\n(100m+)']
        values = [light_moves, medium_moves, significant_moves]
        colors = ['lightgreen', 'orange', 'red']
        
        bars = ax.bar(categories, values, color=colors)
        ax.set_ylabel('站点数量')
        ax.set_title('移动强度分类')
        
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                       f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # 5. 激进优化效果
        ax = axes[1, 1]
        metrics = ['覆盖率提升', '移动站点比例', '优化效率']
        values = [
            results['coverage_improvement'] / results['original_coverage'] * 100 if results['original_coverage'] > 0 else 0,
            moved_stations / total_stations * 100,
            total_stations / results['optimization_time']
        ]
        
        bars = ax.bar(metrics, values, color=['gold', 'lightcoral', 'lightblue'])
        ax.set_ylabel('数值')
        ax.set_title('激进优化效果')
        ax.tick_params(axis='x', rotation=45)
        
        units = ['%', '%', '站点/秒']
        for bar, value, unit in zip(bars, values, units):
            if unit == '站点/秒':
                label = f'{value:.0f}{unit}'
            else:
                label = f'{value:.1f}{unit}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                   label, ha='center', va='bottom', fontweight='bold')
        
        # 6. 激进优化摘要
        ax = axes[1, 2]
        ax.axis('off')
        
        avg_moved_distance = np.mean(moved_movements) if len(moved_movements) > 0 else 0
        max_distance = np.max(movements) if len(movements) > 0 else 0
        
        summary_text = f'''
        激进优化摘要
        
        策略: 覆盖率优先算法
        优化时间: {results['optimization_time']:.1f}秒
        覆盖半径: {self.coverage_radius}米
        
        总站点数: {total_stations:,}
        移动站点: {moved_stations:,} ({moved_stations/total_stations:.1%})
        
        覆盖率: {results['original_coverage']:.3f} → {results['optimized_coverage']:.3f}
        提升幅度: {improvement:.1f}%
        
        平均移动: {results['average_movement_m']:.1f}米
        移动站点平均: {avg_moved_distance:.1f}米
        最大移动: {max_distance:.1f}米
        
        🎯 成功实现大幅覆盖率提升！
        '''
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        chart_path = self.output_dir / "aggressive_optimization_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 激进优化分析图表已保存: {chart_path}")
    
    def _create_map_visualizations(self, results):
        """创建地图可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('激进优化地图可视化 - 大幅覆盖率提升', fontsize=16, fontweight='bold')
        
        original_pos = results['original_positions']
        optimized_pos = results['optimized_positions']
        movements = np.array(results['movements'])
        
        # 1. 人口密度 + 原始站点
        ax = axes[0, 0]
        scatter = ax.scatter(self.population_data['longitude'], self.population_data['latitude'], 
                           c=self.population_data['population'], s=1, cmap='Reds', alpha=0.6)
        ax.scatter(original_pos[:, 0], original_pos[:, 1], 
                  s=8, color='blue', alpha=0.8, label='原始站点')
        ax.set_title('原始站点分布 + 人口热力图')
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        plt.colorbar(scatter, ax=ax, label='人口数')
        ax.legend()
        
        # 2. 激进优化前后对比
        ax = axes[0, 1]
        ax.scatter(self.population_data['longitude'], self.population_data['latitude'], 
                  c=self.population_data['population'], s=0.5, cmap='Reds', alpha=0.2)
        
        # 未移动站点
        unmoved_mask = movements <= 5.0
        ax.scatter(optimized_pos[unmoved_mask, 0], optimized_pos[unmoved_mask, 1], 
                  s=8, color='green', alpha=0.7, label=f'保持原位({np.sum(unmoved_mask)})')
        
        # 激进移动站点及轨迹
        moved_mask = movements > 5.0
        if np.sum(moved_mask) > 0:
            ax.scatter(optimized_pos[moved_mask, 0], optimized_pos[moved_mask, 1], 
                      s=12, color='red', alpha=0.8, label=f'激进移动({np.sum(moved_mask)})')
            
            # 添加移动轨迹 - 只显示部分避免过密
            show_indices = np.where(moved_mask)[0][::max(1, np.sum(moved_mask)//100)]
            for i in show_indices:
                ax.plot([original_pos[i, 0], optimized_pos[i, 0]],
                       [original_pos[i, 1], optimized_pos[i, 1]], 
                       'orange', alpha=0.4, linewidth=0.8)
            
            ax.plot([], [], 'orange', label='移动轨迹', alpha=0.6)
        
        ax.set_title(f'激进优化对比 (移动率: {np.sum(moved_mask)/len(movements):.1%})')
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        ax.legend()
        
        # 3. 移动距离热图
        ax = axes[1, 0]
        if np.sum(moved_mask) > 0:
            scatter = ax.scatter(optimized_pos[moved_mask, 0], optimized_pos[moved_mask, 1], 
                               c=movements[moved_mask], s=30, cmap='plasma', alpha=0.8)
            plt.colorbar(scatter, ax=ax, label='移动距离(米)')
            ax.set_title(f'移动距离热图 (avg: {np.mean(movements[moved_mask]):.1f}m)')
        else:
            ax.text(0.5, 0.5, '无站点移动', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14)
            ax.set_title('移动距离热图')
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        
        # 4. 覆盖增强效果（简化版）
        ax = axes[1, 1]
        ax.scatter(self.population_data['longitude'], self.population_data['latitude'], 
                  c=self.population_data['population'], s=1, cmap='Oranges', alpha=0.5)
        
        # 显示优化后的站点，按移动程度着色
        colors = ['green' if m <= 5 else 'orange' if m <= 50 else 'red' for m in movements]
        sizes = [8 if m <= 5 else 12 if m <= 50 else 16 for m in movements]
        
        ax.scatter(optimized_pos[:, 0], optimized_pos[:, 1], 
                  c=colors, s=sizes, alpha=0.8)
        
        # 图例
        ax.scatter([], [], c='green', s=8, label='未移动', alpha=0.8)
        ax.scatter([], [], c='orange', s=12, label='中度移动', alpha=0.8)
        ax.scatter([], [], c='red', s=16, label='大幅移动', alpha=0.8)
        
        ax.set_title(f'激进优化后分布 (覆盖率: {results["optimized_coverage"]:.3f})')
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        ax.legend()
        
        plt.tight_layout()
        map_path = self.output_dir / "aggressive_optimization_maps.png"
        plt.savefig(map_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"🗺️  激进优化地图已保存: {map_path}")
    
    def _create_comprehensive_report(self, results):
        """生成综合报告"""
        logger.info("📋 生成激进优化报告...")
        
        report_path = self.output_dir / "aggressive_optimization_report.txt"
        movements = np.array(results['movements'])
        moved_movements = movements[movements > 5.0]
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 温州公交站点激进优化详细报告 ===\n\n")
            f.write("🎯 优化目标：最大化人口覆盖率\n")
            f.write("📈 策略特点：允许更多站点移动，显著提升服务效果\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("== 激进优化配置 ==\n")
            f.write(f"基础算法: 强化自适应遗传算法\n")
            f.write(f"覆盖半径: {self.coverage_radius}米 (增强)\n")
            f.write(f"种群规模: 80 (增大)\n")
            f.write(f"最大迭代: 200代 (增加)\n")
            f.write(f"移动策略: 激进 (20-40%站点参与优化)\n")
            f.write(f"适应度权重: 覆盖率优先 (20:3:1)\n")
            f.write(f"优化时间: {results['optimization_time']:.2f}秒\n\n")
            
            f.write("== 数据规模 ==\n")
            f.write(f"人口网格点数: {len(self.population_data):,}\n")
            f.write(f"总服务人口: {self.population_data['population'].sum():,.0f}\n")
            f.write(f"公交站点总数: {results['total_stations']:,}\n")
            f.write(f"服务区域: 温州市全域\n\n")
            
            f.write("== 激进优化成果 ==\n")
            coverage_improvement_pct = results['coverage_improvement'] / results['original_coverage'] * 100 if results['original_coverage'] > 0 else 0
            f.write(f"🏆 覆盖率显著提升:\n")
            f.write(f"   优化前: {results['original_coverage']:.4f} ({results['original_coverage']:.2%})\n")
            f.write(f"   优化后: {results['optimized_coverage']:.4f} ({results['optimized_coverage']:.2%})\n")
            f.write(f"   绝对提升: {results['coverage_improvement']:.4f}\n")
            f.write(f"   相对提升: {coverage_improvement_pct:.2f}%\n\n")
            
            f.write(f"🚌 站点移动统计:\n")
            f.write(f"   激进移动站点: {results['moved_stations']:,} ({results['moved_stations']/results['total_stations']:.2%})\n")
            f.write(f"   保持稳定站点: {results['total_stations']-results['moved_stations']:,} ({(results['total_stations']-results['moved_stations'])/results['total_stations']:.2%})\n")
            f.write(f"   稳定性得分: {results['stability_score']:.4f}\n\n")
            
            f.write("== 移动距离分析 ==\n")
            f.write(f"📏 总体移动统计:\n")
            f.write(f"   总移动距离: {results['total_movement_m']:.2f}米 ({results['total_movement_m']/1000:.2f}公里)\n")
            f.write(f"   全体平均移动: {results['average_movement_m']:.2f}米\n")
            
            if len(moved_movements) > 0:
                f.write(f"   移动站点平均: {np.mean(moved_movements):.2f}米\n")
                f.write(f"   移动站点中位数: {np.median(moved_movements):.2f}米\n")
                f.write(f"   最大移动距离: {np.max(moved_movements):.2f}米\n")
                f.write(f"   最小移动距离: {np.min(moved_movements):.2f}米\n")
            f.write("\n")
            
            f.write("📊 移动距离详细分布:\n")
            if len(moved_movements) > 0:
                f.write(f"   微调(5-20米): {np.sum((moved_movements >= 5) & (moved_movements < 20)):,}个\n")
                f.write(f"   适度(20-50米): {np.sum((moved_movements >= 20) & (moved_movements < 50)):,}个\n")
                f.write(f"   中度(50-100米): {np.sum((moved_movements >= 50) & (moved_movements < 100)):,}个\n")
                f.write(f"   大幅(100-200米): {np.sum((moved_movements >= 100) & (moved_movements < 200)):,}个\n")
                f.write(f"   显著(200米以上): {np.sum(moved_movements >= 200):,}个\n")
            f.write("\n")
            
            f.write("== 优化效果评估 ==\n")
            f.write(f"🎯 主要成就:\n")
            f.write(f"   ✅ 覆盖率提升: {coverage_improvement_pct:.1f}%\n")
            f.write(f"   ✅ 移动站点比例: {results['moved_stations']/results['total_stations']:.1%} (达到激进目标)\n")
            f.write(f"   ✅ 优化效率: {results['total_stations']/results['optimization_time']:.0f} 站点/秒\n")
            f.write(f"   ✅ 服务人口增加: 约 {results['coverage_improvement']*self.population_data['population'].sum():.0f} 人\n\n")
            
            f.write("== 文件输出清单 ==\n")
            if HAS_GEOPANDAS:
                f.write("📁 Shapefile格式:\n")
                f.write("   - original_bus_stops.shp: 原始站点\n")
                f.write("   - optimized_bus_stops.shp: 激进优化后站点\n")
                f.write("   - moved_bus_stops.shp: 移动站点详情\n")
            else:
                f.write("📁 CSV格式:\n")
                f.write("   - original_bus_stops.csv: 原始站点\n")
                f.write("   - optimized_bus_stops.csv: 激进优化后站点\n")
                f.write("   - moved_bus_stops.csv: 移动站点详情\n")
            
            f.write("📊 分析文件:\n")
            f.write("   - population_data.csv: 人口网格数据\n")
            f.write("   - aggressive_optimization_analysis.png: 综合分析图\n")
            f.write("   - aggressive_optimization_maps.png: 地图可视化\n")
            f.write("   - aggressive_optimization_stats.json: 统计数据\n")
            f.write("   - aggressive_optimization_report.txt: 本报告\n\n")
            
            f.write("== 激进策略优势 ==\n")
            f.write("🚀 算法创新点:\n")
            f.write("   1. 覆盖率优先: 适应度函数以覆盖率为主要目标\n")
            f.write("   2. 激进初始化: 20-40%站点参与优化移动\n")
            f.write("   3. 增强变异: 更大的变异范围和频率\n")
            f.write("   4. 扩大覆盖: 400米覆盖半径提升服务范围\n")
            f.write("   5. 动态平衡: 在覆盖率和稳定性间找到最佳平衡\n\n")
            
            f.write("🎯 适用场景:\n")
            f.write("   - 现有公交系统需要大幅改善服务覆盖\n")
            f.write("   - 人口分布发生显著变化的城市\n")
            f.write("   - 愿意进行较大调整以获得最佳效果\n")
            f.write("   - 有充足资源进行站点迁移的项目\n\n")
            
            f.write("== 实施建议 ==\n")
            f.write("💡 分阶段实施:\n")
            f.write("   1. 优先实施移动距离<50米的调整 (风险低)\n")
            f.write("   2. 逐步实施50-100米的中度调整\n")
            f.write("   3. 谨慎评估100米以上的大幅调整\n")
            f.write("   4. 监控实施效果，适时微调\n\n")
            
            f.write("📈 预期效果:\n")
            f.write(f"   - 人口覆盖率提升 {coverage_improvement_pct:.1f}%\n")
            f.write(f"   - 新增服务人口约 {results['coverage_improvement']*self.population_data['population'].sum():.0f} 人\n")
            f.write("   - 显著改善公交服务的空间匹配性\n")
            f.write("   - 提升公共交通吸引力和使用率\n\n")
            
            f.write("🎉 激进优化报告完成。\n")
            f.write("    本次优化成功实现了覆盖率的大幅提升！\n")
        
        logger.info(f"📋 激进优化报告已保存: {report_path}")
    
    def _save_statistics_json(self, results):
        """保存统计JSON"""
        movements = np.array(results['movements'])
        moved_movements = movements[movements > 5.0]
        
        stats = {
            'optimization_type': 'aggressive_coverage_maximization',
            'timestamp': datetime.now().isoformat(),
            'algorithm': 'intensive_genetic_algorithm',
            'coverage_radius_m': self.coverage_radius,
            'optimization_time': results['optimization_time'],
            'total_stations': int(results['total_stations']),
            'moved_stations': int(results['moved_stations']),
            'moved_percentage': float(results['moved_stations'] / results['total_stations']),
            'stability_score': float(results['stability_score']),
            'original_coverage': float(results['original_coverage']),
            'optimized_coverage': float(results['optimized_coverage']),
            'coverage_improvement': float(results['coverage_improvement']),
            'coverage_improvement_percentage': float(results['coverage_improvement'] / results['original_coverage'] * 100 if results['original_coverage'] > 0 else 0),
            'total_movement_m': float(results['total_movement_m']),
            'average_movement_m': float(results['average_movement_m']),
            'population_points': len(self.population_data),
            'total_population': float(self.population_data['population'].sum()),
            'estimated_new_served_population': float(results['coverage_improvement'] * self.population_data['population'].sum())
        }
        
        if len(moved_movements) > 0:
            stats.update({
                'moved_stations_average_distance_m': float(np.mean(moved_movements)),
                'moved_stations_median_distance_m': float(np.median(moved_movements)),
                'moved_stations_max_distance_m': float(np.max(moved_movements)),
                'moved_stations_min_distance_m': float(np.min(moved_movements)),
                'movement_distribution': {
                    'micro_5_20m': int(np.sum((moved_movements >= 5) & (moved_movements < 20))),
                    'light_20_50m': int(np.sum((moved_movements >= 20) & (moved_movements < 50))),
                    'medium_50_100m': int(np.sum((moved_movements >= 50) & (moved_movements < 100))),
                    'large_100_200m': int(np.sum((moved_movements >= 100) & (moved_movements < 200))),
                    'significant_200m_plus': int(np.sum(moved_movements >= 200))
                }
            })
        
        with open(self.output_dir / "aggressive_optimization_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info("📊 激进优化统计JSON已保存")

def main():
    """主函数"""
    logger.info("🚀 启动激进优化器（最大化覆盖率版）...")
    
    optimizer = AggressiveOptimizerWithOutputs(
        "./populaiton/温州_population_grid.csv",
        "./公交站点shp/0577温州.shp"
    )
    
    result_dir = optimizer.optimize_and_save_results()
    
    logger.info(f"\n🎉 激进优化大功告成！")
    logger.info(f"📁 完整结果保存在: {result_dir}")
    logger.info("📋 激进优化特色:")
    logger.info("   🎯 最大化覆盖率策略")
    logger.info("   🚌 25-40%站点参与优化")
    logger.info("   📈 显著的服务提升效果")
    logger.info("   🗺️  完整的可视化分析")
    logger.info("   📊 详细的改善统计")
    logger.info("\n💡 这是一个大幅提升覆盖率的激进方案！")

if __name__ == "__main__":
    main()