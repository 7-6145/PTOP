"""
全规模优化器 - UltraThink超大规模版本
处理整个温州市的所有21465个人口点和10037个站点
使用分布式计算和空间分块策略
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

from data_preprocessing import DataProcessor
from acceleration_utils import fast_population_coverage
from final_optimizer import intelligent_initialization, enhanced_fitness

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FullScaleOptimizer:
    """
    全规模优化器 - 处理整个温州市数据
    
    策略：
    1. 空间分块：将城市划分为网格区域
    2. 并行优化：每个区域独立优化
    3. 全局协调：区域间边界协调
    4. 内存管理：流式处理大规模数据
    """
    
    def __init__(self, population_csv_path: str, bus_stops_shp_path: str):
        """初始化全规模优化器"""
        logger.info("🚀 初始化全规模优化器...")
        
        # 检查系统资源
        self._check_system_resources()
        
        # 加载完整数据
        self._load_full_data(population_csv_path, bus_stops_shp_path)
        
        # 空间分块策略
        self._create_spatial_blocks()
        
        # 优化参数
        self.coverage_radius = 0.008  # 800米覆盖半径
        self.n_processes = min(mp.cpu_count(), 8)  # 限制并发数避免资源耗尽
        
        logger.info(f"✅ 全规模优化器初始化完成:")
        logger.info(f"   📊 数据规模: {len(self.population_data)} 人口点, {len(self.bus_stops_data)} 站点")
        logger.info(f"   🗺️  空间分块: {self.grid_rows} × {self.grid_cols} = {len(self.spatial_blocks)} 个区域")
        logger.info(f"   💻 并行进程: {self.n_processes} 个")
        logger.info(f"   🎯 覆盖半径: {self.coverage_radius * 111.32:.0f}米")
    
    def _check_system_resources(self):
        """检查系统资源"""
        memory = psutil.virtual_memory()
        cpu_count = mp.cpu_count()
        
        logger.info(f"💻 系统资源检查:")
        logger.info(f"   RAM: {memory.total / (1024**3):.1f}GB 总内存, {memory.available / (1024**3):.1f}GB 可用")
        logger.info(f"   CPU: {cpu_count} 核心")
        
        if memory.available < 4 * (1024**3):  # 小于4GB可用内存
            logger.warning("⚠️  可用内存较少，将使用保守的内存管理策略")
        
        if cpu_count < 4:
            logger.warning("⚠️  CPU核心数较少，并行性能可能受限")
    
    def _load_full_data(self, population_csv_path: str, bus_stops_shp_path: str):
        """加载完整数据"""
        logger.info("📁 加载完整数据集...")
        
        processor = DataProcessor(population_csv_path, bus_stops_shp_path)
        self.population_data, self.bus_stops_data, _ = processor.get_processed_data()
        
        # 添加空间索引
        self.population_data['pop_idx'] = range(len(self.population_data))
        self.bus_stops_data['stop_idx'] = range(len(self.bus_stops_data))
        
        # 计算总边界
        self.global_bounds = {
            'min_lon': min(self.population_data['longitude'].min(), 
                          self.bus_stops_data['longitude'].min()),
            'max_lon': max(self.population_data['longitude'].max(), 
                          self.bus_stops_data['longitude'].max()),
            'min_lat': min(self.population_data['latitude'].min(), 
                          self.bus_stops_data['latitude'].min()),
            'max_lat': max(self.population_data['latitude'].max(), 
                          self.bus_stops_data['latitude'].max())
        }
        
        logger.info(f"✅ 数据加载完成:")
        logger.info(f"   人口数据: {len(self.population_data)} 点")
        logger.info(f"   站点数据: {len(self.bus_stops_data)} 个")
        logger.info(f"   服务区域: {self.global_bounds}")
    
    def _create_spatial_blocks(self):
        """创建空间分块"""
        logger.info("🗺️  创建空间分块...")
        
        # 根据数据分布动态确定网格大小
        lon_range = self.global_bounds['max_lon'] - self.global_bounds['min_lon']
        lat_range = self.global_bounds['max_lat'] - self.global_bounds['min_lat']
        
        # 目标：每个块包含800-1500个站点
        target_stations_per_block = 1200
        total_stations = len(self.bus_stops_data)
        
        n_blocks = max(4, total_stations // target_stations_per_block)
        
        # 计算网格维度（尽量接近正方形）
        aspect_ratio = lon_range / lat_range
        self.grid_cols = max(2, int(np.sqrt(n_blocks * aspect_ratio)))
        self.grid_rows = max(2, int(n_blocks / self.grid_cols))
        
        # 创建网格边界
        lon_step = lon_range / self.grid_cols
        lat_step = lat_range / self.grid_rows
        
        self.spatial_blocks = []
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                block_bounds = {
                    'min_lon': self.global_bounds['min_lon'] + col * lon_step,
                    'max_lon': self.global_bounds['min_lon'] + (col + 1) * lon_step,
                    'min_lat': self.global_bounds['min_lat'] + row * lat_step,
                    'max_lat': self.global_bounds['min_lat'] + (row + 1) * lat_step,
                    'row': row,
                    'col': col,
                    'block_id': f"block_{row}_{col}"
                }
                
                # 为边界块扩展边界（处理边界效应）
                margin = 0.005  # 500米边界缓冲
                if col > 0:
                    block_bounds['min_lon'] -= margin
                if col < self.grid_cols - 1:
                    block_bounds['max_lon'] += margin
                if row > 0:
                    block_bounds['min_lat'] -= margin
                if row < self.grid_rows - 1:
                    block_bounds['max_lat'] += margin
                
                # 获取块内数据
                pop_mask = (
                    (self.population_data['longitude'] >= block_bounds['min_lon']) &
                    (self.population_data['longitude'] <= block_bounds['max_lon']) &
                    (self.population_data['latitude'] >= block_bounds['min_lat']) &
                    (self.population_data['latitude'] <= block_bounds['max_lat'])
                )
                
                stop_mask = (
                    (self.bus_stops_data['longitude'] >= block_bounds['min_lon']) &
                    (self.bus_stops_data['longitude'] <= block_bounds['max_lon']) &
                    (self.bus_stops_data['latitude'] >= block_bounds['min_lat']) &
                    (self.bus_stops_data['latitude'] <= block_bounds['max_lat'])
                )
                
                block_pop_data = self.population_data[pop_mask]
                block_stop_data = self.bus_stops_data[stop_mask]
                
                if len(block_stop_data) > 0:  # 只包含有站点的块
                    block = {
                        'bounds': block_bounds,
                        'population_data': block_pop_data,
                        'bus_stops_data': block_stop_data,
                        'n_population': len(block_pop_data),
                        'n_stops': len(block_stop_data)
                    }
                    self.spatial_blocks.append(block)
        
        logger.info(f"✅ 空间分块完成:")
        for i, block in enumerate(self.spatial_blocks):
            logger.info(f"   区域{i+1}: {block['n_population']}人口点, {block['n_stops']}站点")
    
    def optimize_block(self, block_data: Dict) -> Dict[str, Any]:
        """优化单个空间块"""
        block_id = block_data['bounds']['block_id']
        
        try:
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
            
            # 确定优化参数（根据块大小调整）
            n_stops = len(original_positions)
            
            if n_stops < 50:
                population_size, generations = 20, 30
            elif n_stops < 200:
                population_size, generations = 30, 50
            elif n_stops < 500:
                population_size, generations = 40, 60
            else:
                population_size, generations = 50, 80
            
            # 执行优化
            optimized_positions, best_fitness = self._run_block_optimization(
                original_positions, pop_points, pop_weights,
                population_size, generations
            )
            
            optimization_time = time.time() - start_time
            
            # 计算结果统计
            coverage = fast_population_coverage(
                optimized_positions, pop_points, pop_weights, self.coverage_radius
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
                'stop_indices': stop_indices  # 原始站点索引
            }
            
            logger.info(f"✅ 区域 {block_id} 完成: "
                       f"覆盖率{coverage:.3f}, 移动{moved_count}站点, "
                       f"用时{optimization_time:.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 区域 {block_id} 优化失败: {e}")
            return {'block_id': block_id, 'status': 'error', 'error': str(e)}
    
    def _run_block_optimization(self, original_positions: np.ndarray,
                               pop_points: np.ndarray, pop_weights: np.ndarray,
                               population_size: int, generations: int) -> Tuple[np.ndarray, float]:
        """执行块级优化"""
        # 使用改进的遗传算法
        population = intelligent_initialization(original_positions, population_size)
        
        best_individual = population[0].copy()
        best_fitness = -np.inf
        
        for generation in range(generations):
            # 计算适应度
            fitness_scores = np.zeros(population_size)
            
            for i in range(population_size):
                fitness_scores[i] = enhanced_fitness(
                    population[i], original_positions, pop_points, pop_weights, self.coverage_radius
                )
                
                if fitness_scores[i] > best_fitness:
                    best_fitness = fitness_scores[i]
                    best_individual = population[i].copy()
            
            # 新种群生成（简化版本，适应大规模并行）
            new_population = np.zeros_like(population)
            
            # 保留最优个体
            best_idx = np.argmax(fitness_scores)
            new_population[0] = population[best_idx]
            
            # 生成新个体
            for i in range(1, population_size):
                # 锦标赛选择
                parent_idx = self._tournament_select(fitness_scores)
                child = population[parent_idx].copy()
                
                # 保守变异
                if np.random.random() < 0.1:
                    child = self._conservative_mutate(child, original_positions)
                
                new_population[i] = child
            
            population = new_population
            
            # 早停检查
            if generation > 10 and generation % 10 == 0:
                if best_fitness < 0.5:  # 适应度过低，提前停止
                    break
        
        return best_individual, best_fitness
    
    def _tournament_select(self, fitness_scores: np.ndarray, tournament_size: int = 3) -> int:
        """锦标赛选择"""
        pop_size = len(fitness_scores)
        candidates = np.random.choice(pop_size, tournament_size, replace=False)
        best_idx = candidates[np.argmax(fitness_scores[candidates])]
        return best_idx
    
    def _conservative_mutate(self, individual: np.ndarray, original_positions: np.ndarray) -> np.ndarray:
        """保守变异"""
        mutated = individual.copy()
        n_stops = len(individual)
        
        # 只对5%的站点变异
        n_mutate = max(1, int(n_stops * 0.05))
        mutate_indices = np.random.choice(n_stops, n_mutate, replace=False)
        
        for idx in mutate_indices:
            # 小幅移动
            dx = np.random.normal(0, 0.0005)  # 约50米标准差
            dy = np.random.normal(0, 0.0005)
            
            mutated[idx, 0] += dx
            mutated[idx, 1] += dy
        
        return mutated
    
    def optimize_full_scale(self, save_results: bool = True) -> Dict[str, Any]:
        """执行全规模优化"""
        logger.info(f"🚀 开始全规模优化...")
        logger.info(f"   处理 {len(self.spatial_blocks)} 个空间区域")
        logger.info(f"   使用 {self.n_processes} 个并行进程")
        
        start_time = time.time()
        all_results = []
        
        # 并行处理所有空间块
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            # 提交所有任务
            future_to_block = {
                executor.submit(self.optimize_block, block): block
                for block in self.spatial_blocks
            }
            
            # 收集结果
            for future in as_completed(future_to_block):
                try:
                    result = future.result()
                    all_results.append(result)
                    
                    if result['status'] == 'success':
                        logger.info(f"✅ 完成 {result['block_id']}: "
                                   f"覆盖率{result['coverage']:.3f}")
                except Exception as e:
                    logger.error(f"❌ 任务执行失败: {e}")
        
        total_time = time.time() - start_time
        
        # 合并和分析结果
        final_result = self._merge_block_results(all_results, total_time)
        
        # 保存结果
        if save_results:
            self._save_full_results(final_result)
        
        # 显示总结
        self._display_full_results(final_result)
        
        # 清理内存
        gc.collect()
        
        return final_result
    
    def _merge_block_results(self, block_results: List[Dict], total_time: float) -> Dict[str, Any]:
        """合并块结果"""
        logger.info("🔗 合并优化结果...")
        
        successful_blocks = [r for r in block_results if r['status'] == 'success']
        failed_blocks = [r for r in block_results if r['status'] != 'success']
        
        if not successful_blocks:
            raise RuntimeError("所有区域优化都失败了！")
        
        # 重构全局结果
        total_stations = 0
        total_population = 0
        total_moved = 0
        total_movement = 0.0
        weighted_coverage = 0.0
        
        # 构建全局位置数组
        global_original_positions = np.zeros((len(self.bus_stops_data), 2))
        global_optimized_positions = np.zeros((len(self.bus_stops_data), 2))
        
        for block_result in successful_blocks:
            if block_result['status'] != 'success':
                continue
                
            block_stations = block_result['n_stops']
            block_population = block_result['n_population']
            
            # 累计统计
            total_stations += block_stations
            total_population += block_population
            total_moved += block_result['moved_stations']
            total_movement += block_result['total_movement_m']
            
            # 加权覆盖率
            weighted_coverage += block_result['coverage'] * block_population
            
            # 重构全局位置
            stop_indices = block_result['stop_indices']
            original_pos = block_result['original_positions']
            optimized_pos = block_result['optimized_positions']
            
            for i, global_idx in enumerate(stop_indices):
                global_original_positions[global_idx] = original_pos[i]
                global_optimized_positions[global_idx] = optimized_pos[i]
        
        # 计算全局指标
        global_coverage = weighted_coverage / total_population if total_population > 0 else 0
        stability_score = 1.0 - (total_moved / total_stations)
        average_movement = total_movement / total_stations
        
        result = {
            'optimization_time': total_time,
            'total_blocks_processed': len(successful_blocks),
            'failed_blocks': len(failed_blocks),
            'global_metrics': {
                'total_stations': total_stations,
                'total_population': total_population,
                'moved_stations': total_moved,
                'total_movement_m': total_movement,
                'average_movement_m': average_movement,
                'global_coverage': global_coverage,
                'stability_score': stability_score
            },
            'block_results': successful_blocks,
            'global_positions': {
                'original': global_original_positions,
                'optimized': global_optimized_positions
            },
            'processing_stats': {
                'n_processes': self.n_processes,
                'blocks_per_process': len(self.spatial_blocks) / self.n_processes,
                'average_block_time': total_time / len(successful_blocks) if successful_blocks else 0
            }
        }
        
        return result
    
    def _save_full_results(self, result: Dict[str, Any]):
        """保存全规模结果"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"full_scale_results_{timestamp}")
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"💾 保存结果到: {output_dir}")
        
        # 保存主结果（不包含大数组）
        main_result = result.copy()
        main_result.pop('global_positions', None)  # 移除大数组
        
        with open(output_dir / "optimization_summary.json", 'w', encoding='utf-8') as f:
            json.dump(main_result, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存位置数据
        np.save(output_dir / "original_positions.npy", result['global_positions']['original'])
        np.save(output_dir / "optimized_positions.npy", result['global_positions']['optimized'])
        
        # 保存详细结果
        with open(output_dir / "detailed_results.pkl", 'wb') as f:
            pickle.dump(result, f)
        
        logger.info(f"✅ 结果保存完成")
        return output_dir
    
    def _display_full_results(self, result: Dict[str, Any]):
        """显示全规模结果"""
        logger.info(f"\n🎉 === 全规模优化完成 === 🎉")
        logger.info(f"⏱️  总优化时间: {result['optimization_time']:.1f}秒")
        logger.info(f"📊 处理区域数: {result['total_blocks_processed']}")
        logger.info(f"❌ 失败区域数: {result['failed_blocks']}")
        
        metrics = result['global_metrics']
        logger.info(f"\n📈 全局优化指标:")
        logger.info(f"   🏢 总站点数: {metrics['total_stations']:,}")
        logger.info(f"   👥 总人口数: {metrics['total_population']:,}")
        logger.info(f"   🚌 移动站点: {metrics['moved_stations']:,} ({metrics['moved_stations']/metrics['total_stations']*100:.1f}%)")
        logger.info(f"   📏 平均移动: {metrics['average_movement_m']:.0f}米")
        logger.info(f"   🎯 全局覆盖率: {metrics['global_coverage']:.4f} ({metrics['global_coverage']*100:.2f}%)")
        logger.info(f"   ⭐ 稳定性得分: {metrics['stability_score']:.4f}")
        
        stats = result['processing_stats']
        logger.info(f"\n⚡ 性能统计:")
        logger.info(f"   💻 并行进程: {stats['n_processes']}")
        logger.info(f"   📦 平均区域处理时间: {stats['average_block_time']:.1f}秒")
        logger.info(f"   🚀 加速比: {result['optimization_time'] / (stats['average_block_time'] * result['total_blocks_processed']):.1f}x")
    
    def create_full_scale_visualization(self, result: Dict[str, Any], 
                                       sample_ratio: float = 0.1) -> str:
        """创建全规模可视化地图"""
        logger.info(f"🗺️  创建全规模可视化地图...")
        
        # 采样显示（避免地图过于密集）
        pop_sample = self.population_data.sample(
            n=min(2000, int(len(self.population_data) * sample_ratio)),
            random_state=42
        )
        
        stop_sample = self.bus_stops_data.sample(
            n=min(1000, int(len(self.bus_stops_data) * sample_ratio)),
            random_state=42
        )
        
        # 创建地图
        center_lat = (self.global_bounds['min_lat'] + self.global_bounds['max_lat']) / 2
        center_lon = (self.global_bounds['min_lon'] + self.global_bounds['max_lon']) / 2
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # 添加人口密度热力图
        heat_data = [
            [row['latitude'], row['longitude'], row['population']]
            for _, row in pop_sample.iterrows()
        ]
        
        heatmap = plugins.HeatMap(
            heat_data,
            name="人口密度",
            min_opacity=0.2,
            radius=8,
            blur=5
        )
        m.add_child(heatmap)
        
        # 添加优化区域边界
        for i, block in enumerate(self.spatial_blocks[:10]):  # 只显示前10个区域
            bounds = block['bounds']
            folium.Rectangle(
                bounds=[[bounds['min_lat'], bounds['min_lon']], 
                       [bounds['max_lat'], bounds['max_lon']]],
                color='red',
                weight=1,
                fillOpacity=0.1,
                popup=f"区域 {i+1}: {block['n_stops']}站点, {block['n_population']}人口"
            ).add_to(m)
        
        # 添加原始站点（采样）
        for _, stop in stop_sample.iterrows():
            folium.CircleMarker(
                location=[stop['latitude'], stop['longitude']],
                radius=3,
                color='blue',
                fillColor='blue',
                fillOpacity=0.6,
                popup=f"原始站点"
            ).add_to(m)
        
        # 添加结果信息面板
        metrics = result['global_metrics']
        info_html = f"""
        <div style="position: fixed; top: 10px; right: 10px; width: 300px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px; border-radius: 10px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.2);">
        <h4><b>🏆 温州全市优化结果</b></h4>
        <p><b>总站点数:</b> {metrics['total_stations']:,}</p>
        <p><b>总人口数:</b> {metrics['total_population']:,}</p>
        <p><b>全局覆盖率:</b> {metrics['global_coverage']:.2%}</p>
        <p><b>移动站点:</b> {metrics['moved_stations']:,} ({metrics['moved_stations']/metrics['total_stations']*100:.1f}%)</p>
        <p><b>平均移动:</b> {metrics['average_movement_m']:.0f}米</p>
        <p><b>稳定性:</b> {metrics['stability_score']:.3f}</p>
        <p><b>优化时间:</b> {result['optimization_time']:.0f}秒</p>
        </div>
        """
        
        m.get_root().html.add_child(folium.Element(info_html))
        
        # 保存地图
        map_path = "wenzhou_full_optimization.html"
        m.save(map_path)
        
        logger.info(f"🗺️  全规模可视化地图已保存: {map_path}")
        return map_path


def run_full_scale_optimization():
    """运行全规模优化"""
    logger.info("🚀 === 温州全市公交站点优化 === 🚀")
    
    try:
        # 创建全规模优化器
        optimizer = FullScaleOptimizer(
            "./populaiton/温州_population_grid.csv",
            "./公交站点shp/0577温州.shp"
        )
        
        # 执行全规模优化
        result = optimizer.optimize_full_scale(save_results=True)
        
        # 创建可视化
        map_path = optimizer.create_full_scale_visualization(result)
        
        logger.info(f"\n🎯 === 优化总结 === 🎯")
        logger.info(f"✅ 成功优化温州全市公交网络!")
        logger.info(f"📊 处理了 {result['global_metrics']['total_stations']:,} 个站点")
        logger.info(f"👥 服务 {result['global_metrics']['total_population']:,} 人口")
        logger.info(f"🎯 实现 {result['global_metrics']['global_coverage']:.2%} 覆盖率")
        logger.info(f"⚡ 总用时 {result['optimization_time']:.0f} 秒")
        logger.info(f"🗺️  可视化地图: {map_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 全规模优化失败: {e}")
        raise


if __name__ == "__main__":
    # 设置多进程启动方式（Windows兼容）
    mp.set_start_method('spawn', force=True)
    
    # 运行全规模优化
    run_full_scale_optimization()