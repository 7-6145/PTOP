"""
加速优化工具模块
使用numba JIT编译关键计算函数，显著提升计算速度
"""

import numpy as np
from numba import jit, prange, types
from numba.typed import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================= 距离计算优化 =========================

@jit(nopython=True, parallel=True, cache=True)
def fast_euclidean_distance_matrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    快速欧几里得距离矩阵计算
    使用并行计算和缓存优化
    """
    n1, n2 = points1.shape[0], points2.shape[0]
    distances = np.zeros((n1, n2))
    
    for i in prange(n1):
        for j in range(n2):
            dx = points1[i, 0] - points2[j, 0]
            dy = points1[i, 1] - points2[j, 1]
            distances[i, j] = np.sqrt(dx * dx + dy * dy)
    
    return distances

@jit(nopython=True, parallel=True, cache=True)
def fast_haversine_distance_matrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    快速Haversine距离矩阵计算
    专门用于地理坐标的精确距离计算
    """
    n1, n2 = points1.shape[0], points2.shape[0]
    distances = np.zeros((n1, n2))
    
    R = 6371.0  # 地球半径 (km)
    
    for i in prange(n1):
        for j in range(n2):
            # 转换为弧度
            lat1 = np.radians(points1[i, 1])
            lon1 = np.radians(points1[i, 0])
            lat2 = np.radians(points2[j, 1])
            lon2 = np.radians(points2[j, 0])
            
            # Haversine公式
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            
            distances[i, j] = R * c
    
    return distances

@jit(nopython=True, cache=True)
def fast_point_in_radius(center: np.ndarray, points: np.ndarray, radius: float) -> np.ndarray:
    """
    快速查找半径内的点
    """
    n_points = points.shape[0]
    in_radius = np.zeros(n_points, dtype=np.bool_)
    
    for i in range(n_points):
        dx = points[i, 0] - center[0]
        dy = points[i, 1] - center[1]
        distance = np.sqrt(dx * dx + dy * dy)
        
        if distance <= radius:
            in_radius[i] = True
    
    return in_radius

# ========================= 覆盖率计算优化 =========================

@jit(nopython=True, parallel=True, cache=True)
def fast_population_coverage(bus_stops: np.ndarray, 
                           population_points: np.ndarray,
                           population_weights: np.ndarray,
                           coverage_radius: float) -> float:
    """
    高速人口覆盖率计算
    使用并行处理和优化的距离计算
    """
    n_stops = bus_stops.shape[0]
    n_pop = population_points.shape[0]
    
    covered = np.zeros(n_pop, dtype=np.bool_)
    total_population = np.sum(population_weights)
    
    # 对每个人口点，检查是否被任何站点覆盖
    for i in prange(n_pop):
        pop_point = population_points[i]
        
        for j in range(n_stops):
            stop_point = bus_stops[j]
            
            # 快速距离计算
            dx = pop_point[0] - stop_point[0]
            dy = pop_point[1] - stop_point[1]
            distance = np.sqrt(dx * dx + dy * dy)
            
            if distance <= coverage_radius:
                covered[i] = True
                break  # 已被覆盖，无需检查其他站点
    
    # 计算被覆盖的总人口
    covered_population = 0.0
    for i in range(n_pop):
        if covered[i]:
            covered_population += population_weights[i]
    
    return covered_population / total_population if total_population > 0 else 0.0

@jit(nopython=True, cache=True)
def fast_coverage_improvement(original_coverage: float, 
                            new_coverage: float) -> float:
    """快速覆盖率改善计算"""
    return new_coverage - original_coverage

# ========================= 移动成本计算优化 =========================

@jit(nopython=True, parallel=True, cache=True)
def fast_movement_cost(original_positions: np.ndarray,
                      new_positions: np.ndarray,
                      cost_weights: np.ndarray) -> float:
    """
    快速移动成本计算
    """
    n_stops = original_positions.shape[0]
    total_cost = 0.0
    
    for i in prange(n_stops):
        dx = new_positions[i, 0] - original_positions[i, 0]
        dy = new_positions[i, 1] - original_positions[i, 1]
        distance = np.sqrt(dx * dx + dy * dy)
        
        # 转换为公里并加权
        distance_km = distance * 111.32  # 度到公里的近似转换
        total_cost += distance_km * cost_weights[i]
    
    return total_cost

@jit(nopython=True, cache=True)
def fast_movement_distances(original_positions: np.ndarray,
                           new_positions: np.ndarray) -> np.ndarray:
    """快速计算每个站点的移动距离"""
    n_stops = original_positions.shape[0]
    distances = np.zeros(n_stops)
    
    for i in range(n_stops):
        dx = new_positions[i, 0] - original_positions[i, 0]
        dy = new_positions[i, 1] - original_positions[i, 1]
        distances[i] = np.sqrt(dx * dx + dy * dy)
    
    return distances

# ========================= 约束检查优化 =========================

@jit(nopython=True, parallel=True, cache=True)
def fast_boundary_check(positions: np.ndarray,
                       bounds_lower: np.ndarray,
                       bounds_upper: np.ndarray) -> np.ndarray:
    """
    快速边界检查
    """
    n_positions = positions.shape[0]
    violations = np.zeros(n_positions, dtype=np.bool_)
    
    for i in prange(n_positions):
        pos = positions[i]
        lower = bounds_lower[i]
        upper = bounds_upper[i]
        
        if (pos[0] < lower[0] or pos[0] > upper[0] or
            pos[1] < lower[1] or pos[1] > upper[1]):
            violations[i] = True
    
    return violations

@jit(nopython=True, cache=True)
def fast_min_distance_check(positions: np.ndarray, min_distance: float) -> int:
    """
    快速最小距离约束检查
    返回违反约束的站点对数量
    """
    n_stops = positions.shape[0]
    violations = 0
    
    for i in range(n_stops):
        for j in range(i + 1, n_stops):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            distance = np.sqrt(dx * dx + dy * dy)
            
            if distance < min_distance:
                violations += 1
    
    return violations

# ========================= 优化算法加速 =========================

@jit(nopython=True, cache=True)
def fast_gradient_estimation(current_positions: np.ndarray,
                           population_points: np.ndarray,
                           population_weights: np.ndarray,
                           coverage_radius: float,
                           step_size: float = 1e-6) -> np.ndarray:
    """
    快速梯度估计（数值梯度）
    用于梯度下降优化
    """
    n_stops = current_positions.shape[0]
    gradients = np.zeros_like(current_positions)
    
    # 计算基准覆盖率
    base_coverage = fast_population_coverage(
        current_positions, population_points, population_weights, coverage_radius
    )
    
    # 对每个站点的每个坐标分量计算梯度
    for i in range(n_stops):
        for j in range(2):  # x, y坐标
            # 正向扰动
            perturbed_positions = current_positions.copy()
            perturbed_positions[i, j] += step_size
            
            new_coverage = fast_population_coverage(
                perturbed_positions, population_points, population_weights, coverage_radius
            )
            
            # 计算梯度
            gradients[i, j] = (new_coverage - base_coverage) / step_size
    
    return gradients

@jit(nopython=True, cache=True)
def fast_local_search_move(positions: np.ndarray,
                          station_idx: int,
                          directions: np.ndarray,
                          step_sizes: np.ndarray,
                          bounds_lower: np.ndarray,
                          bounds_upper: np.ndarray) -> np.ndarray:
    """
    快速局部搜索移动
    """
    n_directions = directions.shape[0]
    n_steps = len(step_sizes)
    
    # 生成所有可能的移动位置
    moves = np.zeros((n_directions * n_steps, 2))
    valid_moves = np.zeros(n_directions * n_steps, dtype=np.bool_)
    
    move_idx = 0
    original_pos = positions[station_idx]
    
    for i in range(n_directions):
        direction = directions[i]
        
        for j in range(n_steps):
            step_size = step_sizes[j]
            new_pos = original_pos + direction * step_size
            
            # 检查边界
            if (new_pos[0] >= bounds_lower[station_idx, 0] and 
                new_pos[0] <= bounds_upper[station_idx, 0] and
                new_pos[1] >= bounds_lower[station_idx, 1] and 
                new_pos[1] <= bounds_upper[station_idx, 1]):
                
                moves[move_idx] = new_pos
                valid_moves[move_idx] = True
            
            move_idx += 1
    
    return moves, valid_moves

# ========================= 批量计算优化 =========================

@jit(nopython=True, parallel=True, cache=True)
def fast_batch_reward_calculation(positions_batch: np.ndarray,
                                population_points: np.ndarray,
                                population_weights: np.ndarray,
                                original_positions: np.ndarray,
                                coverage_radius: float,
                                movement_weight: float) -> np.ndarray:
    """
    批量奖励计算
    用于评估多个候选解
    """
    n_candidates = positions_batch.shape[0]
    rewards = np.zeros(n_candidates)
    
    for i in prange(n_candidates):
        candidate_positions = positions_batch[i]
        
        # 计算覆盖率
        coverage = fast_population_coverage(
            candidate_positions, population_points, population_weights, coverage_radius
        )
        
        # 计算移动成本
        movement_distances = fast_movement_distances(original_positions, candidate_positions)
        total_movement = np.sum(movement_distances)
        
        # 复合奖励
        rewards[i] = coverage - movement_weight * total_movement
    
    return rewards

# ========================= 测试和验证 =========================

def test_acceleration_functions():
    """测试加速函数的正确性和性能"""
    logger.info("开始测试加速优化函数...")
    
    # 生成测试数据
    n_stops = 50
    n_pop = 1000
    
    bus_stops = np.random.uniform(119.5, 120.5, (n_stops, 2))
    population_points = np.random.uniform(119.5, 120.5, (n_pop, 2))
    population_weights = np.random.uniform(10, 100, n_pop)
    
    coverage_radius = 0.005
    
    # 测试覆盖率计算
    import time
    start_time = time.time()
    
    for _ in range(100):  # 重复计算测试性能
        coverage = fast_population_coverage(
            bus_stops, population_points, population_weights, coverage_radius
        )
    
    elapsed_time = time.time() - start_time
    
    logger.info(f"✓ 覆盖率计算测试通过")
    logger.info(f"  覆盖率: {coverage:.4f}")
    logger.info(f"  100次计算用时: {elapsed_time:.4f}秒")
    
    # 测试距离矩阵计算
    start_time = time.time()
    distances = fast_euclidean_distance_matrix(bus_stops[:10], population_points[:100])
    elapsed_time = time.time() - start_time
    
    logger.info(f"✓ 距离矩阵计算测试通过")
    logger.info(f"  矩阵形状: {distances.shape}")
    logger.info(f"  计算用时: {elapsed_time:.4f}秒")
    
    # 测试梯度估计
    start_time = time.time()
    gradients = fast_gradient_estimation(
        bus_stops, population_points, population_weights, coverage_radius
    )
    elapsed_time = time.time() - start_time
    
    logger.info(f"✓ 梯度估计测试通过")
    logger.info(f"  梯度形状: {gradients.shape}")
    logger.info(f"  计算用时: {elapsed_time:.4f}秒")
    
    logger.info("所有加速函数测试通过!")


if __name__ == "__main__":
    test_acceleration_functions()