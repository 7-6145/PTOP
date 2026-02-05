"""
多城市批量优化系统 - UltraThink版本
基于final_optimizer_with_outputs.py的算法，支持批量处理十几个城市的数据
包含智能城市识别、数据预处理、批量优化和结果汇总功能
"""

import numpy as np
import pandas as pd
from numba import jit, prange
import logging
import time
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import re
import os
import glob
from concurrent.futures import ProcessPoolExecutor
import traceback

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("Warning: GeoPandas not available, will create CSV files instead")

from data_preprocessing import DataProcessor
from acceleration_utils import fast_population_coverage

# 设置中文字体和日志
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 复用final_optimizer_with_outputs.py的核心算法函数
@jit(nopython=True, cache=True)
def intelligent_initialization(original_positions: np.ndarray,
                             population_size: int,
                             max_move_ratio: float = 0.15,
                             max_move_distance: float = 0.003) -> np.ndarray:
    """智能初始化：精确控制移动策略"""
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
def stability_aware_fitness(positions: np.ndarray,
                          original_positions: np.ndarray,
                          pop_points: np.ndarray,
                          pop_weights: np.ndarray,
                          coverage_radius: float) -> float:
    """稳定性感知适应度函数"""
    n_stops = positions.shape[0]
    
    # 1. 覆盖率计算
    coverage_rate = fast_population_coverage(
        positions, pop_points, pop_weights, coverage_radius
    )
    
    # 2. 稳定性计算
    unmoved_count = 0
    total_movement = 0.0
    movement_penalty = 0.0
    
    for i in range(n_stops):
        dx = positions[i, 0] - original_positions[i, 0]
        dy = positions[i, 1] - original_positions[i, 1]
        movement = np.sqrt(dx * dx + dy * dy)
        
        total_movement += movement
        
        # 稳定性：移动小于阈值认为未移动
        if movement < 0.0001:  # ~11米
            unmoved_count += 1
        else:
            # 移动惩罚：超出合理范围大幅惩罚
            if movement > 0.005:  # ~550米
                movement_penalty += movement * 10
            else:
                movement_penalty += movement
    
    stability_score = unmoved_count / n_stops
    avg_movement = total_movement / n_stops
    
    # 综合适应度：优先覆盖率，兼顾稳定性
    fitness = (
        coverage_rate * 10.0 +           # 覆盖率权重最高
        stability_score * 5.0 -          # 稳定性奖励
        movement_penalty * 2.0           # 移动惩罚
    )
    
    return fitness

@jit(nopython=True, cache=True)
def adaptive_genetic_algorithm(original_positions: np.ndarray,
                             pop_points: np.ndarray,
                             pop_weights: np.ndarray,
                             coverage_radius: float,
                             population_size: int = 50,
                             max_generations: int = 100) -> Tuple[np.ndarray, float]:
    """自适应遗传算法"""
    
    # 智能初始化
    population = intelligent_initialization(original_positions, population_size)
    
    best_individual = population[0].copy()
    best_fitness = stability_aware_fitness(
        best_individual, original_positions, pop_points, pop_weights, coverage_radius
    )
    
    stagnation_count = 0
    
    for generation in range(max_generations):
        # 计算适应度
        fitness_scores = np.zeros(population_size)
        
        for i in range(population_size):
            fitness_scores[i] = stability_aware_fitness(
                population[i], original_positions, pop_points, pop_weights, coverage_radius
            )
            
            if fitness_scores[i] > best_fitness:
                best_fitness = fitness_scores[i]
                best_individual = population[i].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
        
        # 早停策略
        if stagnation_count > 20:
            break
        
        # 选择排序
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        # 精英保留
        elite_size = max(2, population_size // 10)
        new_population = np.zeros_like(population)
        
        for i in range(elite_size):
            new_population[i] = population[sorted_indices[i]].copy()
        
        # 生成新个体
        for i in range(elite_size, population_size):
            # 锦标赛选择
            parent1_idx = sorted_indices[np.random.randint(0, min(5, population_size))]
            parent2_idx = sorted_indices[np.random.randint(0, min(5, population_size))]
            
            # 保守交叉
            child = population[parent1_idx].copy()
            
            # 只对10%的站点进行交叉
            n_crossover = max(1, int(original_positions.shape[0] * 0.1))
            crossover_indices = np.random.choice(
                original_positions.shape[0], n_crossover, replace=False
            )
            
            for idx in crossover_indices:
                if np.random.random() < 0.5:
                    child[idx] = population[parent2_idx, idx].copy()
            
            # 保守变异
            if np.random.random() < 0.3:
                n_mutate = max(1, int(original_positions.shape[0] * 0.05))
                mutate_indices = np.random.choice(
                    original_positions.shape[0], n_mutate, replace=False
                )
                
                for idx in mutate_indices:
                    dx = np.random.normal(0, 0.001)
                    dy = np.random.normal(0, 0.001)
                    child[idx, 0] += dx
                    child[idx, 1] += dy
            
            new_population[i] = child
        
        population = new_population
    
    return best_individual, best_fitness

class CityDataMatcher:
    """智能城市数据匹配器"""
    
    def __init__(self, bus_stops_dir: str, population_dir: str):
        """初始化"""
        self.bus_stops_dir = Path(bus_stops_dir)
        self.population_dir = Path(population_dir)
        
    def extract_city_name(self, filename: str) -> str:
        """从文件名中提取城市名称"""
        # 移除文件扩展名
        base_name = Path(filename).stem
        
        # 常见的城市名提取模式
        patterns = [
            r'(\d{4})([^._]+)',  # 匹配如 "0577温州"
            r'([^._\d]+)(?:_.*)?',  # 匹配如 "温州_population_grid"
            r'([^._]+)',  # 通用模式
        ]
        
        for pattern in patterns:
            match = re.search(pattern, base_name)
            if match:
                city_name = match.group(1) if len(match.groups()) == 1 else match.group(2)
                # 清理城市名
                city_name = re.sub(r'[^\u4e00-\u9fff\w]', '', city_name)  # 保留中文和字母数字
                if len(city_name) >= 2:  # 城市名至少2个字符
                    return city_name
        
        return base_name
    
    def discover_cities(self) -> List[Dict[str, Any]]:
        """自动发现并匹配城市数据"""
        logger.info("🔍 开始自动发现城市数据...")
        
        # 1. 发现公交站点文件
        bus_files = []
        for ext in ['*.shp', '*.csv']:
            bus_files.extend(list(self.bus_stops_dir.glob(ext)))
        
        # 2. 发现人口数据文件  
        pop_files = []
        for ext in ['*.csv', '*.shp']:
            pop_files.extend(list(self.population_dir.glob(ext)))
        
        logger.info(f"📁 发现 {len(bus_files)} 个公交文件，{len(pop_files)} 个人口文件")
        
        # 3. 提取城市名并匹配
        bus_cities = {}
        for file in bus_files:
            city = self.extract_city_name(file.name)
            if city:
                bus_cities[city] = file
        
        pop_cities = {}
        for file in pop_files:
            city = self.extract_city_name(file.name)
            if city:
                pop_cities[city] = file
        
        # 4. 智能匹配
        matched_cities = []
        
        for bus_city, bus_file in bus_cities.items():
            # 寻找最佳匹配的人口文件
            best_match = None
            best_score = 0
            
            for pop_city, pop_file in pop_cities.items():
                # 计算匹配分数
                score = self._calculate_match_score(bus_city, pop_city)
                if score > best_score and score > 0.3:  # 最低匹配阈值
                    best_score = score
                    best_match = (pop_city, pop_file)
            
            if best_match:
                matched_cities.append({
                    'city_name': bus_city,
                    'bus_file': str(bus_file),
                    'population_file': str(best_match[1]),
                    'match_score': best_score
                })
                logger.info(f"✅ 匹配成功: {bus_city} (匹配度: {best_score:.2f})")
                logger.info(f"   公交: {bus_file.name}")
                logger.info(f"   人口: {best_match[1].name}")
            else:
                logger.warning(f"❌ 未找到匹配: {bus_city} - {bus_file.name}")
        
        logger.info(f"🎯 成功匹配 {len(matched_cities)} 个城市")
        return matched_cities
    
    def _calculate_match_score(self, name1: str, name2: str) -> float:
        """计算两个城市名的匹配分数"""
        # 完全匹配
        if name1 == name2:
            return 1.0
        
        # 包含关系
        if name1 in name2 or name2 in name1:
            return 0.8
        
        # 编辑距离匹配
        distance = self._edit_distance(name1, name2)
        max_len = max(len(name1), len(name2))
        if max_len > 0:
            similarity = 1.0 - distance / max_len
            return max(0.0, similarity)
        
        return 0.0
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

class MultiCityOptimizer:
    """多城市批量优化器"""
    
    def __init__(self, bus_stops_dir: str, population_dir: str):
        """初始化"""
        self.bus_stops_dir = bus_stops_dir
        self.population_dir = population_dir
        self.coverage_radius = 300  # 300米覆盖半径
        
        # 创建批量结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_output_dir = Path(f"multi_city_optimization_batch_{timestamp}")
        self.batch_output_dir.mkdir(exist_ok=True)
        
        logger.info(f"🚀 初始化多城市批量优化器...")
        logger.info(f"📁 批量结果目录: {self.batch_output_dir}")
        
        # 城市数据匹配器
        self.matcher = CityDataMatcher(bus_stops_dir, population_dir)
        
        # 批量结果统计
        self.batch_results = {
            'total_cities': 0,
            'successful_cities': 0,
            'failed_cities': 0,
            'city_results': [],
            'batch_start_time': datetime.now(),
            'batch_statistics': {}
        }
    
    def run_batch_optimization(self) -> str:
        """运行批量优化"""
        logger.info("🎯 开始多城市批量优化...")
        
        # 1. 发现并匹配城市数据
        cities = self.matcher.discover_cities()
        if not cities:
            logger.error("❌ 未发现任何可匹配的城市数据！")
            return str(self.batch_output_dir)
        
        self.batch_results['total_cities'] = len(cities)
        
        # 2. 批量优化处理
        logger.info(f"🏗️  开始处理 {len(cities)} 个城市...")
        
        for i, city_info in enumerate(cities, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🏙️  处理城市 {i}/{len(cities)}: {city_info['city_name']}")
            logger.info(f"{'='*60}")
            
            try:
                city_result = self._optimize_single_city(city_info, i, len(cities))
                self.batch_results['city_results'].append(city_result)
                
                if city_result['status'] == 'success':
                    self.batch_results['successful_cities'] += 1
                    logger.info(f"✅ 城市 {city_info['city_name']} 优化成功！")
                else:
                    self.batch_results['failed_cities'] += 1
                    logger.error(f"❌ 城市 {city_info['city_name']} 优化失败: {city_result.get('error', 'Unknown')}")
                    
            except Exception as e:
                logger.error(f"❌ 城市 {city_info['city_name']} 处理异常: {e}")
                logger.error(f"详细错误: {traceback.format_exc()}")
                
                self.batch_results['city_results'].append({
                    'city_name': city_info['city_name'],
                    'status': 'error',
                    'error': str(e)
                })
                self.batch_results['failed_cities'] += 1
        
        # 3. 生成批量结果报告
        self._create_batch_summary()
        
        # 4. 显示批量完成信息
        self._display_batch_completion()
        
        return str(self.batch_output_dir)
    
    def _optimize_single_city(self, city_info: Dict, city_index: int, total_cities: int) -> Dict:
        """优化单个城市"""
        city_name = city_info['city_name']
        bus_file = city_info['bus_file']
        pop_file = city_info['population_file']
        
        try:
            # 1. 数据预处理
            logger.info(f"📊 {city_name}: 开始数据预处理...")
            start_time = time.time()
            
            processor = DataProcessor(pop_file, bus_file)
            population_data, bus_stops_data, overlap_info = processor.get_processed_data()
            
            preprocessing_time = time.time() - start_time
            
            logger.info(f"✅ {city_name}: 数据预处理完成 ({preprocessing_time:.1f}s)")
            logger.info(f"   人口网格: {len(population_data):,}")
            logger.info(f"   公交站点: {len(bus_stops_data):,}")
            logger.info(f"   总人口: {population_data['population'].sum():,.0f}")
            
            # 2. 数据有效性检查
            if len(bus_stops_data) < 10:
                return {
                    'city_name': city_name,
                    'status': 'skipped',
                    'error': f'站点数量过少 ({len(bus_stops_data)})',
                    'preprocessing_time': preprocessing_time
                }
            
            if len(population_data) < 50:
                return {
                    'city_name': city_name,
                    'status': 'skipped',
                    'error': f'人口网格过少 ({len(population_data)})',
                    'preprocessing_time': preprocessing_time
                }
            
            # 3. 执行遗传算法优化
            logger.info(f"⚡ {city_name}: 执行遗传算法优化...")
            
            original_positions = bus_stops_data[['longitude', 'latitude']].values
            pop_points = population_data[['longitude', 'latitude']].values
            pop_weights = population_data['population'].values
            
            optimization_start = time.time()
            
            # 根据城市规模调整算法参数
            population_size = min(60, max(30, len(bus_stops_data) // 100))
            max_generations = min(120, max(50, len(bus_stops_data) // 50))
            
            optimized_positions, best_fitness = adaptive_genetic_algorithm(
                original_positions, pop_points, pop_weights, 
                self.coverage_radius / 111320.0,  # 转换为度
                population_size=population_size, 
                max_generations=max_generations
            )
            
            optimization_time = time.time() - optimization_start
            
            # 4. 计算详细统计
            results = self._calculate_city_stats(
                city_name, original_positions, optimized_positions, 
                pop_points, pop_weights, preprocessing_time, 
                optimization_time, best_fitness
            )
            
            # 5. 保存城市结果
            city_output_dir = self.batch_output_dir / f"{city_name}_results"
            city_output_dir.mkdir(exist_ok=True)
            
            self._save_city_results(
                city_output_dir, city_name, population_data, bus_stops_data,
                original_positions, optimized_positions, results
            )
            
            logger.info(f"🎉 {city_name}: 优化完成！")
            logger.info(f"   移动站点: {results['moved_stations']}/{results['total_stations']} ({results['moved_stations']/results['total_stations']:.1%})")
            logger.info(f"   覆盖率: {results['original_coverage']:.3f} → {results['optimized_coverage']:.3f}")
            logger.info(f"   用时: 预处理{preprocessing_time:.1f}s + 优化{optimization_time:.1f}s")
            
            results.update({
                'status': 'success',
                'city_name': city_name,
                'output_dir': str(city_output_dir)
            })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ {city_name}: 优化失败 - {e}")
            return {
                'city_name': city_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def _calculate_city_stats(self, city_name, original_positions, optimized_positions, 
                            pop_points, pop_weights, preprocessing_time, 
                            optimization_time, best_fitness):
        """计算城市优化统计"""
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
            
            if movement_m > 10.0:  # 移动超过10米
                moved_count += 1
        
        stability_score = (n_stops - moved_count) / n_stops
        total_population = pop_weights.sum()
        
        return {
            'city_name': city_name,
            'preprocessing_time': preprocessing_time,
            'optimization_time': optimization_time,
            'total_time': preprocessing_time + optimization_time,
            'best_fitness': best_fitness,
            'total_stations': n_stops,
            'total_population': float(total_population),
            'population_grids': len(pop_points),
            'moved_stations': moved_count,
            'stability_score': stability_score,
            'original_coverage': original_coverage,
            'optimized_coverage': optimized_coverage,
            'coverage_improvement': optimized_coverage - original_coverage,
            'coverage_improvement_pct': (optimized_coverage - original_coverage) / original_coverage * 100 if original_coverage > 0 else 0,
            'total_movement_m': total_movement_m,
            'average_movement_m': total_movement_m / n_stops,
            'movements': movements,
            'original_positions': original_positions,
            'optimized_positions': optimized_positions
        }
    
    def _save_city_results(self, output_dir, city_name, population_data, bus_stops_data,
                          original_positions, optimized_positions, results):
        """保存单个城市的结果"""
        
        # 1. 保存站点数据
        optimized_stops = bus_stops_data.copy()
        optimized_stops['original_lon'] = original_positions[:, 0]
        optimized_stops['original_lat'] = original_positions[:, 1]
        optimized_stops['longitude'] = optimized_positions[:, 0]
        optimized_stops['latitude'] = optimized_positions[:, 1]
        optimized_stops['movement_m'] = results['movements']
        optimized_stops['is_moved'] = [m > 10.0 for m in results['movements']]
        
        moved_stops = optimized_stops[optimized_stops['is_moved'] == True]
        
        if HAS_GEOPANDAS:
            # 保存为Shapefile
            self._save_city_shapefile(output_dir / "original_bus_stops.shp", bus_stops_data)
            self._save_city_shapefile(output_dir / "optimized_bus_stops.shp", optimized_stops)
            self._save_city_shapefile(output_dir / "moved_bus_stops.shp", moved_stops)
        else:
            # 保存为CSV
            bus_stops_data.to_csv(output_dir / "original_bus_stops.csv", index=False)
            optimized_stops.to_csv(output_dir / "optimized_bus_stops.csv", index=False)
            moved_stops.to_csv(output_dir / "moved_bus_stops.csv", index=False)
        
        # 2. 保存人口数据
        population_data.to_csv(output_dir / "population_data.csv", index=False)
        
        # 3. 保存统计数据
        city_stats = {k: v for k, v in results.items() 
                     if k not in ['movements', 'original_positions', 'optimized_positions']}
        
        with open(output_dir / f"{city_name}_optimization_stats.json", 'w', encoding='utf-8') as f:
            json.dump(city_stats, f, indent=2, ensure_ascii=False)
        
        # 4. 生成城市报告
        self._create_city_report(output_dir, city_name, results)
    
    def _save_city_shapefile(self, filename, data):
        """保存城市shapefile"""
        if HAS_GEOPANDAS:
            gdf = gpd.GeoDataFrame(
                data,
                geometry=gpd.points_from_xy(data['longitude'], data['latitude']),
                crs='EPSG:4326'
            )
            gdf.to_file(filename, encoding='utf-8')
    
    def _create_city_report(self, output_dir, city_name, results):
        """创建城市优化报告"""
        report_path = output_dir / f"{city_name}_optimization_report.txt"
        movements = np.array(results['movements'])
        moved_movements = movements[movements > 10.0]
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"=== {city_name} 公交站点优化报告 ===\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"算法版本: final_optimizer_with_outputs.py (多城市批量版)\n\n")
            
            f.write("== 优化参数 ==\n")
            f.write(f"覆盖半径: {self.coverage_radius}米\n")
            f.write(f"数据预处理时间: {results['preprocessing_time']:.2f}秒\n")
            f.write(f"算法优化时间: {results['optimization_time']:.2f}秒\n")
            f.write(f"总处理时间: {results['total_time']:.2f}秒\n\n")
            
            f.write("== 数据规模 ==\n")
            f.write(f"人口网格数: {results['population_grids']:,}\n")
            f.write(f"总服务人口: {results['total_population']:,.0f}\n")
            f.write(f"公交站点数: {results['total_stations']:,}\n\n")
            
            f.write("== 优化结果 ==\n")
            f.write(f"移动站点数: {results['moved_stations']:,} ({results['moved_stations']/results['total_stations']:.2%})\n")
            f.write(f"稳定性得分: {results['stability_score']:.4f}\n")
            f.write(f"原始覆盖率: {results['original_coverage']:.4f}\n")
            f.write(f"优化后覆盖率: {results['optimized_coverage']:.4f}\n")
            f.write(f"覆盖率提升: {results['coverage_improvement']:.4f} ({results['coverage_improvement_pct']:.1f}%)\n")
            f.write(f"平均移动距离: {results['average_movement_m']:.2f}米\n")
            
            if len(moved_movements) > 0:
                f.write(f"移动站点平均距离: {np.mean(moved_movements):.2f}米\n")
                f.write(f"最大移动距离: {np.max(moved_movements):.2f}米\n")
            
            f.write(f"总移动距离: {results['total_movement_m']:.2f}米\n\n")
            
            f.write("报告完成。\n")
    
    def _create_batch_summary(self):
        """创建批量优化汇总报告"""
        logger.info("📋 生成批量优化汇总报告...")
        
        # 1. 计算批量统计
        successful_results = [r for r in self.batch_results['city_results'] if r.get('status') == 'success']
        
        if successful_results:
            # 汇总统计
            total_stations = sum(r.get('total_stations', 0) for r in successful_results)
            total_moved = sum(r.get('moved_stations', 0) for r in successful_results)
            total_population = sum(r.get('total_population', 0) for r in successful_results)
            avg_coverage_improvement = np.mean([r.get('coverage_improvement_pct', 0) for r in successful_results])
            total_optimization_time = sum(r.get('optimization_time', 0) for r in successful_results)
            
            self.batch_results['batch_statistics'] = {
                'total_stations_processed': total_stations,
                'total_stations_moved': total_moved,
                'overall_move_percentage': total_moved / total_stations if total_stations > 0 else 0,
                'total_population_served': total_population,
                'average_coverage_improvement_pct': avg_coverage_improvement,
                'total_optimization_time': total_optimization_time,
                'average_time_per_city': total_optimization_time / len(successful_results) if successful_results else 0
            }
        
        # 2. 保存汇总统计JSON
        batch_stats_file = self.batch_output_dir / "batch_optimization_summary.json"
        with open(batch_stats_file, 'w', encoding='utf-8') as f:
            # 转换numpy类型为Python原生类型
            serializable_results = []
            for result in self.batch_results['city_results']:
                clean_result = {}
                for k, v in result.items():
                    if k in ['movements', 'original_positions', 'optimized_positions']:
                        continue  # 跳过大数组
                    elif isinstance(v, (np.integer, np.floating)):
                        clean_result[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        clean_result[k] = v.tolist()
                    else:
                        clean_result[k] = v
                serializable_results.append(clean_result)
            
            summary_data = {
                'batch_info': {
                    'start_time': self.batch_results['batch_start_time'].isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_cities': self.batch_results['total_cities'],
                    'successful_cities': self.batch_results['successful_cities'],
                    'failed_cities': self.batch_results['failed_cities'],
                    'success_rate': self.batch_results['successful_cities'] / self.batch_results['total_cities'] if self.batch_results['total_cities'] > 0 else 0
                },
                'batch_statistics': self.batch_results.get('batch_statistics', {}),
                'city_results': serializable_results
            }
            
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # 3. 生成文本汇总报告
        self._create_batch_text_report()
        
        logger.info(f"📊 批量汇总报告已保存: {batch_stats_file}")
    
    def _create_batch_text_report(self):
        """生成批量优化文本报告"""
        report_path = self.batch_output_dir / "batch_optimization_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 多城市公交站点批量优化报告 ===\n\n")
            f.write(f"批量处理时间: {self.batch_results['batch_start_time'].strftime('%Y-%m-%d %H:%M:%S')} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"算法版本: final_optimizer_with_outputs.py (UltraThink 多城市批量版)\n\n")
            
            f.write("== 批量处理概况 ==\n")
            f.write(f"总处理城市数: {self.batch_results['total_cities']}\n")
            f.write(f"成功优化城市: {self.batch_results['successful_cities']}\n")
            f.write(f"失败城市数: {self.batch_results['failed_cities']}\n")
            f.write(f"成功率: {self.batch_results['successful_cities']/self.batch_results['total_cities']:.1%}\n\n")
            
            if 'batch_statistics' in self.batch_results:
                stats = self.batch_results['batch_statistics']
                f.write("== 批量优化统计 ==\n")
                f.write(f"总处理站点数: {stats.get('total_stations_processed', 0):,}\n")
                f.write(f"总移动站点数: {stats.get('total_stations_moved', 0):,}\n")
                f.write(f"整体移动比例: {stats.get('overall_move_percentage', 0):.2%}\n")
                f.write(f"总服务人口: {stats.get('total_population_served', 0):,.0f}\n")
                f.write(f"平均覆盖率提升: {stats.get('average_coverage_improvement_pct', 0):.1f}%\n")
                f.write(f"总优化时间: {stats.get('total_optimization_time', 0):.1f}秒\n")
                f.write(f"平均每城市用时: {stats.get('average_time_per_city', 0):.1f}秒\n\n")
            
            f.write("== 各城市优化详情 ==\n")
            for i, result in enumerate(self.batch_results['city_results'], 1):
                city_name = result.get('city_name', f'City_{i}')
                status = result.get('status', 'unknown')
                
                f.write(f"{i}. {city_name}: {status.upper()}\n")
                
                if status == 'success':
                    f.write(f"   站点数: {result.get('total_stations', 0):,} (移动: {result.get('moved_stations', 0):,})\n")
                    f.write(f"   覆盖率提升: {result.get('coverage_improvement_pct', 0):.1f}%\n")
                    f.write(f"   用时: {result.get('total_time', 0):.1f}秒\n")
                elif status in ['failed', 'error']:
                    f.write(f"   错误: {result.get('error', 'Unknown error')}\n")
                elif status == 'skipped':
                    f.write(f"   跳过原因: {result.get('error', 'Unknown reason')}\n")
                
                f.write("\n")
            
            f.write("== 算法特点 ==\n")
            f.write("1. 基于final_optimizer_with_outputs.py的成熟算法\n")
            f.write("2. 智能城市数据匹配和预处理\n")
            f.write("3. 自适应参数调整适应不同规模城市\n")
            f.write("4. 稳定性优先的保守优化策略\n")
            f.write("5. 完整的结果文件输出和统计报告\n\n")
            
            f.write("批量优化报告完成。\n")
    
    def _display_batch_completion(self):
        """显示批量完成信息"""
        logger.info("\n" + "="*80)
        logger.info("🎉 多城市批量优化完成！")
        logger.info("="*80)
        logger.info(f"📊 处理结果:")
        logger.info(f"   总城市数: {self.batch_results['total_cities']}")
        logger.info(f"   成功优化: {self.batch_results['successful_cities']}")
        logger.info(f"   失败数量: {self.batch_results['failed_cities']}")
        logger.info(f"   成功率: {self.batch_results['successful_cities']/self.batch_results['total_cities']:.1%}")
        
        if 'batch_statistics' in self.batch_results:
            stats = self.batch_results['batch_statistics']
            logger.info(f"🚌 优化统计:")
            logger.info(f"   总站点: {stats.get('total_stations_processed', 0):,}")
            logger.info(f"   移动站点: {stats.get('total_stations_moved', 0):,}")
            logger.info(f"   平均覆盖提升: {stats.get('average_coverage_improvement_pct', 0):.1f}%")
        
        logger.info(f"📁 结果保存位置: {self.batch_output_dir}")
        logger.info("="*80 + "\n")

def main():
    """主函数"""
    logger.info("🚀 启动多城市批量优化系统 (UltraThink版)...")
    logger.info("🎯 基于final_optimizer_with_outputs.py的成熟算法")
    
    # 配置路径
    bus_stops_dir = "./公交站点shp"
    population_dir = "./populaiton"
    
    # 检查目录是否存在
    if not Path(bus_stops_dir).exists():
        logger.error(f"❌ 公交站点目录不存在: {bus_stops_dir}")
        return
    
    if not Path(population_dir).exists():
        logger.error(f"❌ 人口数据目录不存在: {population_dir}")
        return
    
    # 创建并运行批量优化器
    optimizer = MultiCityOptimizer(bus_stops_dir, population_dir)
    result_dir = optimizer.run_batch_optimization()
    
    logger.info(f"🎊 多城市批量优化大功告成！")
    logger.info(f"📁 完整结果查看: {result_dir}")
    logger.info("💡 每个城市都有独立的结果文件夹和完整分析报告")

if __name__ == "__main__":
    main()