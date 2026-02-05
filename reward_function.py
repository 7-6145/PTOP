"""
奖励函数模块
实现基于人口覆盖率和移动距离的复合奖励函数
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from numba import jit
import logging

from spatial_index import PopulationCoverageCalculator, BusStopSpatialIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardFunction:
    """
    复合奖励函数类
    优化目标：最大化人口覆盖率，同时最小化站点移动距离
    """
    
    def __init__(self, 
                 population_data: pd.DataFrame,
                 bus_stops_data: pd.DataFrame,
                 coverage_radius: float = 0.005,
                 coverage_weight: float = 1.0,
                 movement_weight: float = 0.3,
                 stability_weight: float = 0.1,
                 min_distance_penalty_weight: float = 0.5):
        """
        初始化奖励函数
        
        Args:
            population_data: 人口数据
            bus_stops_data: 公交站点数据
            coverage_radius: 站点覆盖半径
            coverage_weight: 人口覆盖率权重
            movement_weight: 移动距离惩罚权重
            stability_weight: 稳定性奖励权重
            min_distance_penalty_weight: 站点最小距离惩罚权重
        """
        self.coverage_radius = coverage_radius
        self.coverage_weight = coverage_weight
        self.movement_weight = movement_weight
        self.stability_weight = stability_weight
        self.min_distance_penalty_weight = min_distance_penalty_weight
        
        # 初始化覆盖率计算器
        self.coverage_calculator = PopulationCoverageCalculator(
            population_data, coverage_radius
        )
        
        # 初始化站点空间索引
        self.stop_spatial_index = BusStopSpatialIndex(bus_stops_data)
        
        # 计算基准覆盖率（原始站点配置）
        original_positions = self.stop_spatial_index.original_positions
        self.baseline_coverage = self.coverage_calculator.calculate_coverage_ratio_fast(
            original_positions
        )
        
        # 参数设置
        self.min_distance_threshold = 0.001  # 最小站点间距（约100米）
        self.max_movement_distance = 0.01    # 最大允许移动距离（约1公里）
        
        logger.info(f"奖励函数初始化完成")
        logger.info(f"基准覆盖率: {self.baseline_coverage:.4f}")
        logger.info(f"权重设置 - 覆盖率: {coverage_weight}, 移动: {movement_weight}, 稳定性: {stability_weight}")
    
    def calculate_reward(self, new_positions: np.ndarray) -> Dict[str, float]:
        """
        计算复合奖励函数值
        
        Args:
            new_positions: 新的站点位置 shape (n_stops, 2)
            
        Returns:
            包含各组件奖励的字典
        """
        # 1. 人口覆盖率奖励
        coverage_ratio = self.coverage_calculator.calculate_coverage_ratio_fast(new_positions)
        coverage_improvement = coverage_ratio - self.baseline_coverage
        coverage_reward = self.coverage_weight * coverage_improvement
        
        # 2. 移动距离惩罚
        movement_distances = self._calculate_movement_distances(new_positions)
        total_movement = movement_distances.sum()
        normalized_movement = min(total_movement / self.max_movement_distance, 1.0)
        movement_penalty = -self.movement_weight * normalized_movement
        
        # 3. 稳定性奖励（奖励少量移动的解决方案）
        unmoved_count = np.sum(movement_distances < 0.0001)  # 几乎没移动的站点
        stability_bonus = self.stability_weight * (unmoved_count / len(new_positions))
        
        # 4. 站点间最小距离惩罚（避免站点过于集中）
        min_distance_penalty = self._calculate_min_distance_penalty(new_positions)
        
        # 5. 边界惩罚（避免站点移出服务区域）
        boundary_penalty = self._calculate_boundary_penalty(new_positions)
        
        # 总奖励
        total_reward = (coverage_reward + 
                       movement_penalty + 
                       stability_bonus - 
                       min_distance_penalty - 
                       boundary_penalty)
        
        return {
            'total_reward': total_reward,
            'coverage_reward': coverage_reward,
            'movement_penalty': movement_penalty,
            'stability_bonus': stability_bonus,
            'min_distance_penalty': min_distance_penalty,
            'boundary_penalty': boundary_penalty,
            'coverage_ratio': coverage_ratio,
            'coverage_improvement': coverage_improvement,
            'total_movement': total_movement,
            'unmoved_count': unmoved_count
        }
    
    def _calculate_movement_distances(self, new_positions: np.ndarray) -> np.ndarray:
        """计算每个站点的移动距离"""
        original_positions = self.stop_spatial_index.original_positions
        
        # 使用优化的距离计算
        distances = self._haversine_distance_matrix(original_positions, new_positions)
        
        # 对角线元素是对应站点的移动距离
        movement_distances = np.diag(distances)
        
        return movement_distances
    
    def _calculate_min_distance_penalty(self, positions: np.ndarray) -> float:
        """
        计算站点间最小距离惩罚
        惩罚过于接近的站点
        """
        if len(positions) <= 1:
            return 0.0
        
        # 计算所有站点间的距离
        distances = self._haversine_distance_matrix(positions, positions)
        
        # 排除对角线（自己与自己的距离）
        np.fill_diagonal(distances, np.inf)
        
        # 找到最小距离
        min_distances = np.min(distances, axis=1)
        
        # 计算惩罚：距离过近的站点数量
        too_close_count = np.sum(min_distances < self.min_distance_threshold)
        penalty = self.min_distance_penalty_weight * (too_close_count / len(positions))
        
        return penalty
    
    def _calculate_boundary_penalty(self, positions: np.ndarray) -> float:
        """
        计算边界惩罚
        惩罚移出合理服务区域的站点
        """
        # 基于原始人口数据定义合理边界
        pop_data = self.coverage_calculator.population_data
        
        min_lon = pop_data['longitude'].min() - 0.005
        max_lon = pop_data['longitude'].max() + 0.005
        min_lat = pop_data['latitude'].min() - 0.005
        max_lat = pop_data['latitude'].max() + 0.005
        
        penalty = 0.0
        for pos in positions:
            lon, lat = pos
            if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                penalty += 0.5  # 每个出界站点惩罚0.5
        
        return penalty
    
    @staticmethod
    @jit(nopython=True)
    def _haversine_distance_matrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """
        优化的Haversine距离矩阵计算
        """
        n1, n2 = points1.shape[0], points2.shape[0]
        distances = np.zeros((n1, n2))
        
        R = 6371  # 地球半径 (km)
        
        for i in range(n1):
            for j in range(n2):
                lon1, lat1 = np.radians(points1[i, 0]), np.radians(points1[i, 1])
                lon2, lat2 = np.radians(points2[j, 0]), np.radians(points2[j, 1])
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                
                distances[i, j] = R * c / 111.32
        
        return distances
    
    def evaluate_solution_quality(self, new_positions: np.ndarray) -> Dict[str, Any]:
        """
        评估解决方案质量的全面分析
        
        Args:
            new_positions: 新的站点位置
            
        Returns:
            详细的质量评估报告
        """
        reward_dict = self.calculate_reward(new_positions)
        
        # 额外的质量指标
        movement_distances = self._calculate_movement_distances(new_positions)
        
        quality_report = {
            **reward_dict,
            'quality_metrics': {
                'coverage_improvement_percent': reward_dict['coverage_improvement'] * 100,
                'average_movement_km': movement_distances.mean() * 111.32,
                'max_movement_km': movement_distances.max() * 111.32,
                'moved_stations_count': np.sum(movement_distances > 0.0001),
                'significantly_moved_count': np.sum(movement_distances > 0.002),  # >200米
            }
        }
        
        return quality_report
    
    def get_optimization_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取优化边界
        基于原始站点位置和合理移动范围
        """
        original_positions = self.stop_spatial_index.original_positions
        
        # 允许每个站点在原位置周围一定范围内移动
        move_range = 0.01  # 约1公里
        
        lower_bounds = original_positions - move_range
        upper_bounds = original_positions + move_range
        
        # 同时考虑人口分布边界
        pop_data = self.coverage_calculator.population_data
        global_min_lon = pop_data['longitude'].min() - 0.002
        global_max_lon = pop_data['longitude'].max() + 0.002
        global_min_lat = pop_data['latitude'].min() - 0.002
        global_max_lat = pop_data['latitude'].max() + 0.002
        
        # 应用全局边界限制
        lower_bounds[:, 0] = np.maximum(lower_bounds[:, 0], global_min_lon)
        lower_bounds[:, 1] = np.maximum(lower_bounds[:, 1], global_min_lat)
        upper_bounds[:, 0] = np.minimum(upper_bounds[:, 0], global_max_lon)
        upper_bounds[:, 1] = np.minimum(upper_bounds[:, 1], global_max_lat)
        
        return lower_bounds, upper_bounds


def test_reward_function():
    """测试奖励函数"""
    logger.info("开始测试奖励函数...")
    
    try:
        # 加载预处理数据
        pop_data = pd.read_csv("processed_population_data.csv")
        bus_data = pd.read_csv("processed_population_data.csv")  # 临时使用人口数据
        
        # 创建一些假的公交站点数据进行测试
        n_stops = 10
        bus_data = pd.DataFrame({
            'longitude': np.linspace(pop_data['longitude'].min(), pop_data['longitude'].max(), n_stops),
            'latitude': np.linspace(pop_data['latitude'].min(), pop_data['latitude'].max(), n_stops)
        })
        
        # 初始化奖励函数
        reward_func = RewardFunction(pop_data, bus_data)
        
        # 测试原始位置
        original_positions = bus_data[['longitude', 'latitude']].values
        original_reward = reward_func.calculate_reward(original_positions)
        
        logger.info("原始配置奖励:")
        for key, value in original_reward.items():
            logger.info(f"  {key}: {value:.6f}")
        
        # 测试轻微移动
        moved_positions = original_positions.copy()
        moved_positions[:3] += np.random.normal(0, 0.002, (3, 2))  # 移动前3个站点
        
        moved_reward = reward_func.calculate_reward(moved_positions)
        logger.info("\n移动后配置奖励:")
        for key, value in moved_reward.items():
            logger.info(f"  {key}: {value:.6f}")
        
        # 质量评估
        quality = reward_func.evaluate_solution_quality(moved_positions)
        logger.info("\n质量评估:")
        for key, value in quality['quality_metrics'].items():
            logger.info(f"  {key}: {value:.4f}")
        
        logger.info("奖励函数测试完成!")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    test_reward_function()