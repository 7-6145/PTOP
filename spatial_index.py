"""
空间索引模块
使用rtree构建高效的空间搜索结构，加速邻近查询和覆盖分析
"""

import numpy as np
import pandas as pd
from rtree import index
from scipy.spatial.distance import cdist
from numba import jit, prange
from typing import List, Tuple, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpatialIndex:
    """空间索引类，用于快速邻近搜索"""
    
    def __init__(self, points: np.ndarray, properties: Optional[Dict] = None):
        """
        初始化空间索引
        
        Args:
            points: shape (n, 2) 的坐标数组 [longitude, latitude]
            properties: 可选的属性字典，如人口数据
        """
        self.points = np.array(points)
        self.properties = properties if properties is not None else {}
        self.n_points = len(self.points)
        
        # 构建rtree索引
        self.idx = index.Index()
        self._build_index()
        
        logger.info(f"构建空间索引完成，包含 {self.n_points} 个点")
    
    def _build_index(self):
        """构建rtree索引"""
        for i, (x, y) in enumerate(self.points):
            # rtree需要边界框 (minx, miny, maxx, maxy)
            self.idx.insert(i, (x, y, x, y))
    
    def find_neighbors_within_radius(self, center: Tuple[float, float], 
                                   radius: float) -> List[int]:
        """
        查找半径内的所有邻居点
        
        Args:
            center: 中心点坐标 (lon, lat)
            radius: 搜索半径（度）
            
        Returns:
            邻居点的索引列表
        """
        lon, lat = center
        # 创建搜索边界框
        bbox = (lon - radius, lat - radius, lon + radius, lat + radius)
        
        # 使用rtree粗筛
        candidates = list(self.idx.intersection(bbox))
        
        # 精确距离筛选
        if not candidates:
            return []
        
        candidate_points = self.points[candidates]
        distances = self._haversine_distance_vectorized(
            np.array([[lon, lat]]), candidate_points
        )[0]
        
        # 返回真正在半径内的点
        valid_indices = np.where(distances <= radius)[0]
        return [candidates[i] for i in valid_indices]
    
    def find_k_nearest_neighbors(self, center: Tuple[float, float], 
                               k: int = 5) -> List[Tuple[int, float]]:
        """
        查找k个最近邻
        
        Args:
            center: 中心点坐标
            k: 邻居数量
            
        Returns:
            (索引, 距离) 的列表，按距离排序
        """
        # 计算所有点到中心的距离
        center_array = np.array([center])
        distances = self._haversine_distance_vectorized(center_array, self.points)[0]
        
        # 找到k个最近的点
        k_indices = np.argpartition(distances, k)[:k]
        k_distances = distances[k_indices]
        
        # 按距离排序
        sort_order = np.argsort(k_distances)
        
        return [(k_indices[i], k_distances[i]) for i in sort_order]
    
    @staticmethod
    @jit(nopython=True)
    def _haversine_distance_vectorized(points1: np.ndarray, 
                                     points2: np.ndarray) -> np.ndarray:
        """
        使用Haversine公式计算地理距离（优化版本）
        
        Args:
            points1: shape (n1, 2) 的坐标数组
            points2: shape (n2, 2) 的坐标数组
            
        Returns:
            shape (n1, n2) 的距离矩阵（单位：度）
        """
        n1, n2 = points1.shape[0], points2.shape[0]
        distances = np.zeros((n1, n2))
        
        R = 6371  # 地球半径 (km)
        
        for i in prange(n1):
            for j in prange(n2):
                lon1, lat1 = np.radians(points1[i, 0]), np.radians(points1[i, 1])
                lon2, lat2 = np.radians(points2[j, 0]), np.radians(points2[j, 1])
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                
                # 转换回度数进行比较
                distances[i, j] = R * c / 111.32  # 大约的度数转换
        
        return distances


class PopulationCoverageCalculator:
    """人口覆盖率计算器"""
    
    def __init__(self, population_data: pd.DataFrame, coverage_radius: float = 0.005):
        """
        初始化人口覆盖率计算器
        
        Args:
            population_data: 人口数据，包含longitude, latitude, population列
            coverage_radius: 站点覆盖半径（度，约500米）
        """
        self.population_data = population_data
        self.coverage_radius = coverage_radius
        
        # 构建人口点的空间索引
        pop_points = population_data[['longitude', 'latitude']].values
        self.pop_index = SpatialIndex(pop_points)
        
        self.population_weights = population_data['population'].values
        self.total_population = self.population_weights.sum()
        
        logger.info(f"人口覆盖率计算器初始化完成")
        logger.info(f"总人口: {self.total_population:.2f}")
        logger.info(f"覆盖半径: {coverage_radius} 度")
    
    def calculate_coverage_ratio(self, bus_stop_positions: np.ndarray) -> float:
        """
        计算公交站点的人口覆盖率
        
        Args:
            bus_stop_positions: shape (n_stops, 2) 的站点坐标数组
            
        Returns:
            人口覆盖率 (0-1)
        """
        covered_population = 0.0
        covered_points = set()
        
        for stop_pos in bus_stop_positions:
            # 找到该站点覆盖范围内的人口点
            neighbor_indices = self.pop_index.find_neighbors_within_radius(
                tuple(stop_pos), self.coverage_radius
            )
            
            # 累加未被其他站点覆盖的人口
            for idx in neighbor_indices:
                if idx not in covered_points:
                    covered_population += self.population_weights[idx]
                    covered_points.add(idx)
        
        return covered_population / self.total_population if self.total_population > 0 else 0.0
    
    def calculate_coverage_ratio_fast(self, bus_stop_positions: np.ndarray) -> float:
        """
        快速计算人口覆盖率（使用numba优化）
        
        Args:
            bus_stop_positions: shape (n_stops, 2) 的站点坐标数组
            
        Returns:
            人口覆盖率 (0-1)
        """
        return self._fast_coverage_calculation(
            bus_stop_positions, 
            self.pop_index.points, 
            self.population_weights,
            self.coverage_radius,
            self.total_population
        )
    
    @staticmethod
    @jit(nopython=True)
    def _fast_coverage_calculation(stops: np.ndarray, 
                                 pop_points: np.ndarray,
                                 pop_weights: np.ndarray,
                                 radius: float,
                                 total_pop: float) -> float:
        """
        使用numba优化的快速覆盖率计算
        """
        n_pop = len(pop_points)
        n_stops = len(stops)
        covered = np.zeros(n_pop, dtype=np.bool_)
        
        # 对每个人口点，检查是否被任何站点覆盖
        for i in prange(n_pop):
            pop_point = pop_points[i]
            
            for j in range(n_stops):
                stop_point = stops[j]
                
                # 简化的距离计算（欧几里得距离近似）
                dx = pop_point[0] - stop_point[0]
                dy = pop_point[1] - stop_point[1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance <= radius:
                    covered[i] = True
                    break
        
        # 计算被覆盖的总人口
        covered_population = 0.0
        for i in range(n_pop):
            if covered[i]:
                covered_population += pop_weights[i]
        
        return covered_population / total_pop if total_pop > 0 else 0.0


class BusStopSpatialIndex:
    """公交站点空间索引"""
    
    def __init__(self, bus_stops_data: pd.DataFrame):
        """
        初始化公交站点空间索引
        
        Args:
            bus_stops_data: 公交站点数据
        """
        self.bus_stops_data = bus_stops_data
        
        # 提取坐标
        stop_points = bus_stops_data[['longitude', 'latitude']].values
        self.stop_index = SpatialIndex(stop_points)
        
        self.original_positions = stop_points.copy()
        self.n_stops = len(stop_points)
        
        logger.info(f"公交站点空间索引构建完成，{self.n_stops} 个站点")
    
    def calculate_movement_cost(self, new_positions: np.ndarray) -> float:
        """
        计算站点移动成本
        
        Args:
            new_positions: 新的站点位置
            
        Returns:
            总移动距离
        """
        if len(new_positions) != self.n_stops:
            raise ValueError("新位置数量必须与原站点数量相同")
        
        # 计算每个站点的移动距离
        distances = SpatialIndex._haversine_distance_vectorized(
            self.original_positions, new_positions
        )
        
        # 对角线元素是对应站点的移动距离
        movement_distances = np.diag(distances)
        
        return movement_distances.sum()
    
    def get_neighbor_stations(self, station_idx: int, radius: float) -> List[int]:
        """获取指定站点附近的其他站点"""
        center = tuple(self.original_positions[station_idx])
        neighbors = self.stop_index.find_neighbors_within_radius(center, radius)
        
        # 移除自己
        return [idx for idx in neighbors if idx != station_idx]


def test_spatial_index():
    """测试空间索引功能"""
    logger.info("开始测试空间索引...")
    
    try:
        # 加载预处理数据
        pop_data = pd.read_csv("processed_population_data.csv")
        
        # 测试人口覆盖率计算器
        calculator = PopulationCoverageCalculator(pop_data)
        
        # 创建一些测试站点
        test_stops = np.array([
            [119.65, 27.18],
            [119.67, 27.19],
            [119.69, 27.17]
        ])
        
        # 计算覆盖率
        coverage_ratio = calculator.calculate_coverage_ratio_fast(test_stops)
        logger.info(f"测试覆盖率: {coverage_ratio:.4f}")
        
        logger.info("空间索引测试完成!")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    test_spatial_index()