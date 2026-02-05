"""
全规模优化器 - 修复版本
解决Windows多进程兼容性问题
"""

import numpy as np
import pandas as pd
from numba import jit, prange
import logging
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
import gc
import psutil
import folium
from folium import plugins
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json
import os

from data_preprocessing import DataProcessor
from acceleration_utils import fast_population_coverage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局函数，避免pickle问题
def simple_worker_func(x):
    """全局工作函数"""
    return x * x

def optimize_block_worker(block_info):
    """优化单个区域的工作函数"""
    try:
        block_data, coverage_radius = block_info
        return _optimize_single_block(block_data, coverage_radius)
    except Exception as e:
        logger.error(f"区域优化失败: {e}")
        return {'status': 'error', 'error': str(e)}

def _optimize_single_block(block_data: Dict, coverage_radius: float) -> Dict[str, Any]:
    """优化单个区域"""
    block_id = block_data['bounds']['block_id']
    
    logger.info(f"🔄 优化区域 {block_id}...")
    start_time = time.time()
    
    # 提取数据
    pop_data = block_data['population_data']
    stop_data = block_data['bus_stops_data']
    
    if len(stop_data) == 0:
        return {'block_id': block_id, 'status': 'empty', 'result': None}
    
    # 数据预处理
    pop_points = pop_data[['longitude', 'latitude']].values
    pop_weights = pop_data['population'].values
    original_positions = stop_data[['longitude', 'latitude']].values
    stop_indices = stop_data['stop_idx'].values
    
    # 根据块大小调整参数
    n_stops = len(original_positions)
    
    if n_stops < 50:
        population_size, generations = 20, 20
    elif n_stops < 200:
        population_size, generations = 30, 30
    elif n_stops < 500:
        population_size, generations = 40, 40
    else:
        population_size, generations = 50, 50
    
    # 执行简化的优化算法
    optimized_positions, best_fitness = _simplified_genetic_algorithm(
        original_positions, pop_points, pop_weights, coverage_radius,
        population_size, generations
    )
    
    optimization_time = time.time() - start_time
    
    # 计算结果统计
    coverage = fast_population_coverage(
        optimized_positions, pop_points, pop_weights, coverage_radius
    )
    
    moved_count = 0
    total_movement = 0.0
    
    for i in range(len(original_positions)):
        dx = optimized_positions[i, 0] - original_positions[i, 0]
        dy = optimized_positions[i, 1] - original_positions[i, 1]
        movement = np.sqrt(dx * dx + dy * dy) * 111.32 * 1000  # 转换为米
        
        if movement > 10:  # 移动超过10米
            moved_count += 1
        total_movement += movement
    
    result = {
        'block_id': block_id,
        'status': 'success',
        'optimization_time': optimization_time,
        'n_stops': n_stops,
        'n_population': len(pop_points),
        'coverage': coverage,
        'moved_stations': moved_count,
        'total_movement_m': total_movement,
        'average_movement_m': total_movement / n_stops,
        'best_fitness': best_fitness,
        'original_positions': original_positions,
        'optimized_positions': optimized_positions,
        'stop_indices': stop_indices
    }
    
    logger.info(f"✅ 区域 {block_id} 完成: "
               f"覆盖率{coverage:.3f}, 移动{moved_count}站点, "
               f"用时{optimization_time:.1f}s")
    
    return result

@jit(nopython=True, cache=True)
def _simplified_genetic_algorithm(original_positions: np.ndarray,
                                pop_points: np.ndarray,
                                pop_weights: np.ndarray,
                                coverage_radius: float,
                                population_size: int,
                                generations: int) -> Tuple[np.ndarray, float]:
    """简化的遗传算法（numba优化）"""
    n_stops = len(original_positions)
    
    # 初始化种群
    population = np.zeros((population_size, n_stops, 2))
    for i in range(population_size):
        population[i] = original_positions.copy()
        
        # 只移动少数站点
        n_move = max(1, int(n_stops * 0.15))
        for _ in range(n_move):
            idx = np.random.randint(0, n_stops)
            dx = np.random.normal(0, 0.001)
            dy = np.random.normal(0, 0.001)
            population[i, idx, 0] += dx
            population[i, idx, 1] += dy
    
    best_individual = population[0].copy()
    best_fitness = -np.inf
    
    for generation in range(generations):
        # 计算适应度
        fitness_scores = np.zeros(population_size)
        
        for i in range(population_size):
            # 计算覆盖率
            coverage = fast_population_coverage(
                population[i], pop_points, pop_weights, coverage_radius
            )
            
            # 计算稳定性
            total_movement = 0.0
            unmoved_count = 0
            
            for j in range(n_stops):
                dx = population[i, j, 0] - original_positions[j, 0]
                dy = population[i, j, 1] - original_positions[j, 1]
                movement = np.sqrt(dx * dx + dy * dy)
                
                total_movement += movement
                if movement < 0.0001:
                    unmoved_count += 1
            
            # 复合适应度
            stability = unmoved_count / n_stops
            fitness_scores[i] = 3.0 * coverage + 2.0 * stability - 0.5 * total_movement
            
            if fitness_scores[i] > best_fitness:
                best_fitness = fitness_scores[i]
                best_individual = population[i].copy()
        
        # 新种群生成（简化版本）
        new_population = np.zeros_like(population)
        
        # 保留最佳个体
        best_idx = np.argmax(fitness_scores)
        new_population[0] = population[best_idx]
        
        # 生成新个体
        for i in range(1, population_size):
            # 选择父母
            parent_idx = _tournament_select_simple(fitness_scores)
            child = population[parent_idx].copy()
            
            # 变异
            if np.random.random() < 0.2:
                for _ in range(max(1, int(n_stops * 0.05))):
                    idx = np.random.randint(0, n_stops)
                    child[idx, 0] += np.random.normal(0, 0.0005)
                    child[idx, 1] += np.random.normal(0, 0.0005)
            
            new_population[i] = child
        
        population = new_population
    
    return best_individual, best_fitness

@jit(nopython=True, cache=True)
def _tournament_select_simple(fitness_scores: np.ndarray) -> int:
    """简单锦标赛选择"""
    pop_size = len(fitness_scores)
    idx1 = np.random.randint(0, pop_size)
    idx2 = np.random.randint(0, pop_size)
    idx3 = np.random.randint(0, pop_size)
    
    if fitness_scores[idx1] >= fitness_scores[idx2] and fitness_scores[idx1] >= fitness_scores[idx3]:
        return idx1
    elif fitness_scores[idx2] >= fitness_scores[idx3]:
        return idx2
    else:
        return idx3

class QuickFullScaleOptimizer:
    """快速全规模优化器"""
    
    def __init__(self, population_csv_path: str, bus_stops_shp_path: str):
        """初始化优化器"""
        logger.info("🚀 初始化快速全规模优化器...")
        
        # 检查系统资源
        memory = psutil.virtual_memory()
        cpu_count = mp.cpu_count()
        
        logger.info(f"💻 系统资源:")
        logger.info(f"   CPU: {cpu_count} 核心")
        logger.info(f"   内存: {memory.available / (1024**3):.1f}GB 可用")
        
        # 加载数据
        processor = DataProcessor(population_csv_path, bus_stops_shp_path)
        self.population_data, self.bus_stops_data, _ = processor.get_processed_data()
        
        # 添加索引
        self.population_data = self.population_data.reset_index(drop=True)
        self.bus_stops_data = self.bus_stops_data.reset_index(drop=True)
        self.bus_stops_data['stop_idx'] = range(len(self.bus_stops_data))
        
        # 创建空间分块
        self._create_spatial_blocks()
        
        # 优化参数
        self.coverage_radius = 0.008
        self.n_processes = min(cpu_count, 8)  # 限制最大进程数
        
        logger.info(f"✅ 初始化完成:")
        logger.info(f"   数据: {len(self.population_data)} 人口点, {len(self.bus_stops_data)} 站点")
        logger.info(f"   分块: {len(self.spatial_blocks)} 个区域")
        logger.info(f"   进程: {self.n_processes} 个")
    
    def _create_spatial_blocks(self):
        """创建空间分块"""
        logger.info("🗺️  创建空间分块...")
        
        # 计算边界
        all_lons = np.concatenate([
            self.population_data['longitude'].values,
            self.bus_stops_data['longitude'].values
        ])
        all_lats = np.concatenate([
            self.population_data['latitude'].values,
            self.bus_stops_data['latitude'].values
        ])
        
        min_lon, max_lon = all_lons.min(), all_lons.max()
        min_lat, max_lat = all_lats.min(), all_lats.max()
        
        # 动态网格大小
        target_stations_per_block = 800
        total_stations = len(self.bus_stops_data)
        n_blocks = max(4, total_stations // target_stations_per_block)
        
        # 网格维度
        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat
        aspect_ratio = lon_range / lat_range
        
        grid_cols = max(2, int(np.sqrt(n_blocks * aspect_ratio)))
        grid_rows = max(2, int(n_blocks / grid_cols))
        
        # 创建网格
        lon_step = lon_range / grid_cols
        lat_step = lat_range / grid_rows
        
        self.spatial_blocks = []
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                # 块边界
                block_min_lon = min_lon + col * lon_step
                block_max_lon = min_lon + (col + 1) * lon_step
                block_min_lat = min_lat + row * lat_step
                block_max_lat = min_lat + (row + 1) * lat_step
                
                # 添加边界缓冲
                margin = 0.005
                if col > 0:
                    block_min_lon -= margin
                if col < grid_cols - 1:
                    block_max_lon += margin
                if row > 0:
                    block_min_lat -= margin
                if row < grid_rows - 1:
                    block_max_lat += margin
                
                # 筛选数据
                pop_mask = (
                    (self.population_data['longitude'] >= block_min_lon) &
                    (self.population_data['longitude'] <= block_max_lon) &
                    (self.population_data['latitude'] >= block_min_lat) &
                    (self.population_data['latitude'] <= block_max_lat)
                )
                
                stop_mask = (
                    (self.bus_stops_data['longitude'] >= block_min_lon) &
                    (self.bus_stops_data['longitude'] <= block_max_lon) &
                    (self.bus_stops_data['latitude'] >= block_min_lat) &
                    (self.bus_stops_data['latitude'] <= block_max_lat)
                )
                
                block_pop = self.population_data[pop_mask].copy()
                block_stops = self.bus_stops_data[stop_mask].copy()
                
                if len(block_stops) > 0:
                    block = {
                        'bounds': {
                            'block_id': f"block_{row}_{col}",
                            'min_lon': block_min_lon,
                            'max_lon': block_max_lon,
                            'min_lat': block_min_lat,
                            'max_lat': block_max_lat
                        },
                        'population_data': block_pop,
                        'bus_stops_data': block_stops,
                        'n_population': len(block_pop),
                        'n_stops': len(block_stops)
                    }
                    self.spatial_blocks.append(block)
        
        logger.info(f"✅ 分块完成: {len(self.spatial_blocks)} 个区域")
        for i, block in enumerate(self.spatial_blocks):
            logger.info(f"   区域{i+1}: {block['n_stops']}站点, {block['n_population']}人口")
    
    def optimize_full_scale(self) -> Dict[str, Any]:
        """执行全规模优化"""
        logger.info(f"🚀 开始全规模优化...")
        logger.info(f"   处理 {len(self.spatial_blocks)} 个区域")
        logger.info(f"   使用 {self.n_processes} 个进程")
        
        start_time = time.time()
        
        # 准备任务数据
        block_tasks = [(block, self.coverage_radius) for block in self.spatial_blocks]
        
        # 串行处理各区域（避免多进程问题）
        all_results = []
        
        logger.info("🔄 采用串行处理模式以确保稳定性...")
        
        for i, (block, coverage_radius) in enumerate(block_tasks, 1):
            try:
                logger.info(f"🔄 处理区域 {i}/{len(block_tasks)}: {block['bounds']['block_id']}...")
                
                # 直接调用优化函数
                result = _optimize_single_block(block, coverage_radius)
                
                if result.get('status') == 'success':
                    all_results.append(result)
                    # 安全访问metrics字段
                    metrics = result.get('metrics', {})
                    coverage_rate = metrics.get('coverage_rate', 0)
                    moved_stations = metrics.get('moved_stations', 0)
                    optimization_time = metrics.get('optimization_time', 0)
                    logger.info(f"✅ 区域 {result['block_id']} 完成: 覆盖率{coverage_rate:.3f}, 移动{moved_stations}站点, 用时{optimization_time:.1f}s")
                else:
                    logger.error(f"❌ 区域 {i} 优化失败: {result.get('error', '未知错误')}")
                    
                # 强制垃圾回收
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ 区域 {i} 处理失败: {e}")
                
        # 如果串行处理也有问题，尝试更保守的多进程
        if len(all_results) < len(block_tasks) // 2:
            logger.warning("🔄 串行处理效果不佳，尝试保守多进程模式...")
            all_results = []
            
            # 极保守的多进程设置
            effective_processes = min(2, self.n_processes)
            with ProcessPoolExecutor(max_workers=effective_processes) as executor:
                for i, task in enumerate(block_tasks, 1):
                    try:
                        future = executor.submit(optimize_block_worker, task)
                        result = future.result(timeout=600)  # 10分钟超时
                        
                        if result.get('status') == 'success':
                            all_results.append(result)
                            logger.info(f"📦 完成区域 {i}/{len(block_tasks)}: {result.get('block_id', '未知')}")
                        else:
                            logger.error(f"❌ 区域 {i} 优化失败: {result.get('error', '未知错误')}")
                    except Exception as e:
                        logger.error(f"❌ 区域 {i} 处理失败: {e}")
        
        total_time = time.time() - start_time
        
        # 合并结果
        final_result = self._merge_results(all_results, total_time)
        
        # 显示结果
        self._display_results(final_result)
        
        # 保存结果
        self._save_results(final_result)
        
        return final_result
    
    def _merge_results(self, block_results: List[Dict], total_time: float) -> Dict[str, Any]:
        """合并块结果"""
        logger.info("🔗 合并优化结果...")
        
        successful_results = [r for r in block_results if r.get('status') == 'success']
        
        if not successful_results:
            raise RuntimeError("所有区域优化都失败了！")
        
        # 统计全局指标
        total_stations = sum(r['n_stops'] for r in successful_results)
        total_population = sum(r['n_population'] for r in successful_results)
        total_moved = sum(r['moved_stations'] for r in successful_results)
        total_movement = sum(r['total_movement_m'] for r in successful_results)
        
        # 计算加权覆盖率
        weighted_coverage = sum(r['coverage'] * r['n_population'] for r in successful_results)
        global_coverage = weighted_coverage / total_population if total_population > 0 else 0
        
        # 构建结果
        result = {
            'optimization_time': total_time,
            'successful_blocks': len(successful_results),
            'failed_blocks': len(block_results) - len(successful_results),
            'global_metrics': {
                'total_stations': total_stations,
                'total_population': total_population,
                'moved_stations': total_moved,
                'total_movement_m': total_movement,
                'average_movement_m': total_movement / total_stations if total_stations > 0 else 0,
                'global_coverage': global_coverage,
                'stability_score': 1.0 - (total_moved / total_stations) if total_stations > 0 else 1.0
            },
            'block_results': successful_results
        }
        
        return result
    
    def _display_results(self, result: Dict[str, Any]):
        """显示结果"""
        logger.info(f"\n🎉 === 全规模优化完成 === 🎉")
        logger.info(f"⏱️  总优化时间: {result['optimization_time']:.1f}秒")
        logger.info(f"✅ 成功处理: {result['successful_blocks']} 个区域")
        logger.info(f"❌ 失败处理: {result['failed_blocks']} 个区域")
        
        metrics = result['global_metrics']
        logger.info(f"\n📊 全局优化指标:")
        logger.info(f"   🏢 总站点数: {metrics['total_stations']:,}")
        logger.info(f"   👥 总人口数: {metrics['total_population']:,}")
        logger.info(f"   🎯 全局覆盖率: {metrics['global_coverage']:.4f} ({metrics['global_coverage']*100:.2f}%)")
        logger.info(f"   🚌 移动站点: {metrics['moved_stations']:,} ({metrics['moved_stations']/metrics['total_stations']*100:.1f}%)")
        logger.info(f"   📏 平均移动: {metrics['average_movement_m']:.0f}米")
        logger.info(f"   ⭐ 稳定性: {metrics['stability_score']:.4f}")
        
        # 计算服务人口
        covered_population = metrics['global_coverage'] * metrics['total_population']
        logger.info(f"   👥 服务人口: {covered_population:,.0f} 人")
    
    def _save_results(self, result: Dict[str, Any]):
        """保存结果"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"full_scale_results_{timestamp}")
        output_dir.mkdir(exist_ok=True)
        
        # 保存摘要
        summary = {
            'optimization_time': result['optimization_time'],
            'successful_blocks': result['successful_blocks'],
            'global_metrics': result['global_metrics'],
            'timestamp': timestamp
        }
        
        with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 结果已保存到: {output_dir}")


def quick_test():
    """快速测试"""
    logger.info("🚀 === 快速全规模优化测试 === 🚀")
    
    try:
        # 快速多进程测试
        logger.info("🔄 测试多进程...")
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(simple_worker_func, i) for i in range(5)]
            results = [f.result() for f in futures]
        logger.info(f"✅ 多进程测试成功: {results}")
        
        # 创建优化器
        optimizer = QuickFullScaleOptimizer(
            "./populaiton/温州_population_grid.csv", 
            "./公交站点shp/0577温州.shp"
        )
        
        # 运行优化
        result = optimizer.optimize_full_scale()
        
        logger.info(f"\n🏆 优化成功完成！")
        return result
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        raise


if __name__ == "__main__":
    if os.name == 'nt':  # Windows
        mp.set_start_method('spawn', force=True)
    
    quick_test()