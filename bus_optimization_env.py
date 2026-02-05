"""
公交站点优化强化学习环境
基于Gymnasium框架构建的自定义环境
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging

from data_preprocessing import DataProcessor
from reward_function import RewardFunction
from spatial_index import PopulationCoverageCalculator, BusStopSpatialIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BusStopOptimizationEnv(gym.Env):
    """
    公交站点优化环境
    
    状态空间：当前所有站点的坐标 [n_stops * 2]
    动作空间：每个站点的位置调整 [n_stops * 2]，范围[-max_move, max_move]
    奖励：基于人口覆盖率改善和移动成本的复合奖励
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, 
                 population_csv_path: str,
                 bus_stops_shp_path: str,
                 max_move_per_step: float = 0.001,
                 max_episode_steps: int = 100,
                 coverage_radius: float = 0.005,
                 reward_weights: Optional[Dict[str, float]] = None):
        """
        初始化环境
        
        Args:
            population_csv_path: 人口数据路径
            bus_stops_shp_path: 公交站点数据路径
            max_move_per_step: 每步最大移动距离
            max_episode_steps: 最大回合步数
            coverage_radius: 站点覆盖半径
            reward_weights: 奖励权重配置
        """
        super().__init__()
        
        self.max_move_per_step = max_move_per_step
        self.max_episode_steps = max_episode_steps
        self.coverage_radius = coverage_radius
        
        # 设置默认奖励权重
        self.reward_weights = reward_weights or {
            'coverage_weight': 10.0,
            'movement_weight': 0.5,
            'stability_weight': 0.2,
            'min_distance_penalty_weight': 1.0
        }
        
        # 加载和预处理数据
        self._load_data(population_csv_path, bus_stops_shp_path)
        
        # 初始化奖励函数
        self.reward_function = RewardFunction(
            self.population_data, 
            self.bus_stops_data,
            coverage_radius=coverage_radius,
            **self.reward_weights
        )
        
        # 定义动作和观察空间
        self._setup_spaces()
        
        # 环境状态
        self.current_positions = None
        self.initial_positions = None
        self.step_count = 0
        self.episode_rewards = []
        self.best_positions = None
        self.best_reward = -np.inf
        
        logger.info(f"公交站点优化环境初始化完成")
        logger.info(f"站点数量: {self.n_stops}")
        logger.info(f"最大移动步长: {max_move_per_step}")
        logger.info(f"最大回合步数: {max_episode_steps}")
    
    def _load_data(self, population_csv_path: str, bus_stops_shp_path: str):
        """加载数据"""
        processor = DataProcessor(population_csv_path, bus_stops_shp_path)
        self.population_data, self.bus_stops_data, self.analysis = processor.get_processed_data()
        
        # 提取站点位置
        self.n_stops = len(self.bus_stops_data)
        self.original_positions = self.bus_stops_data[['longitude', 'latitude']].values
        
        # 计算有效边界
        self.bounds_lower, self.bounds_upper = self._calculate_bounds()
        
        logger.info(f"数据加载完成: {len(self.population_data)} 人口点, {self.n_stops} 公交站点")
    
    def _calculate_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """计算站点移动的有效边界"""
        # 基于人口分布确定边界
        pop_margin = 0.005
        min_lon = self.population_data['longitude'].min() - pop_margin
        max_lon = self.population_data['longitude'].max() + pop_margin
        min_lat = self.population_data['latitude'].min() - pop_margin  
        max_lat = self.population_data['latitude'].max() + pop_margin
        
        lower_bounds = np.full((self.n_stops, 2), [min_lon, min_lat])
        upper_bounds = np.full((self.n_stops, 2), [max_lon, max_lat])
        
        return lower_bounds, upper_bounds
    
    def _setup_spaces(self):
        """设置动作和观察空间"""
        # 观察空间：所有站点的坐标
        obs_low = self.bounds_lower.flatten()
        obs_high = self.bounds_upper.flatten()
        self.observation_space = spaces.Box(
            low=obs_low, 
            high=obs_high, 
            shape=(self.n_stops * 2,), 
            dtype=np.float32
        )
        
        # 动作空间：每个站点的位置调整
        action_low = np.full(self.n_stops * 2, -self.max_move_per_step)
        action_high = np.full(self.n_stops * 2, self.max_move_per_step)
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            shape=(self.n_stops * 2,),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置到原始位置（可选择添加小量噪声）
        if options and options.get('add_noise', False):
            noise_scale = options.get('noise_scale', 0.0005)
            noise = self.np_random.normal(0, noise_scale, self.original_positions.shape)
            self.current_positions = self.original_positions + noise
        else:
            self.current_positions = self.original_positions.copy()
        
        self.initial_positions = self.current_positions.copy()
        self.step_count = 0
        self.episode_rewards = []
        
        # 确保位置在边界内
        self._clip_positions()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步动作"""
        # 将动作reshape为位置调整
        action = action.reshape(self.n_stops, 2)
        
        # 应用动作（位置调整）
        new_positions = self.current_positions + action
        
        # 确保新位置在边界内
        new_positions = np.clip(new_positions, self.bounds_lower, self.bounds_upper)
        
        # 更新当前位置
        self.current_positions = new_positions
        
        # 计算奖励
        reward_dict = self.reward_function.calculate_reward(new_positions)
        reward = reward_dict['total_reward']
        
        # 记录奖励
        self.episode_rewards.append(reward)
        
        # 更新最佳解
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_positions = new_positions.copy()
        
        # 检查回合结束条件
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_episode_steps
        
        # 额外的终止条件（可选）
        if reward_dict['coverage_improvement'] > 0.1:  # 覆盖率提升超过10%
            terminated = True
        
        observation = self._get_observation()
        info = self._get_info(reward_dict)
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观察"""
        return self.current_positions.flatten().astype(np.float32)
    
    def _get_info(self, reward_dict: Optional[Dict] = None) -> Dict:
        """获取环境信息"""
        info = {
            'step': self.step_count,
            'n_stops': self.n_stops,
            'episode_length': len(self.episode_rewards)
        }
        
        if reward_dict:
            info.update({
                'coverage_ratio': reward_dict['coverage_ratio'],
                'coverage_improvement': reward_dict['coverage_improvement'],
                'total_movement': reward_dict['total_movement'],
                'unmoved_count': reward_dict['unmoved_count']
            })
        
        # 计算与原始位置的总移动距离
        total_displacement = np.linalg.norm(
            self.current_positions - self.original_positions
        )
        info['total_displacement'] = total_displacement
        
        return info
    
    def _clip_positions(self):
        """确保位置在有效边界内"""
        self.current_positions = np.clip(
            self.current_positions, 
            self.bounds_lower, 
            self.bounds_upper
        )
    
    def render(self, mode: str = "human"):
        """渲染环境（简单的文本输出）"""
        if mode == "human":
            print(f"\n=== Step {self.step_count} ===")
            if self.episode_rewards:
                print(f"Last reward: {self.episode_rewards[-1]:.4f}")
            print(f"Average reward: {np.mean(self.episode_rewards):.4f}")
            print(f"Best reward: {self.best_reward:.4f}")
            
            # 显示前5个站点的位置
            print("前5个站点位置:")
            for i in range(min(5, self.n_stops)):
                orig = self.original_positions[i]
                curr = self.current_positions[i]
                movement = np.linalg.norm(curr - orig) * 111.32  # 转换为公里
                print(f"  站点{i}: ({curr[0]:.6f}, {curr[1]:.6f}) 移动{movement:.3f}km")
    
    def get_best_solution(self) -> Dict[str, Any]:
        """获取当前最佳解决方案"""
        if self.best_positions is None:
            return None
        
        quality_report = self.reward_function.evaluate_solution_quality(self.best_positions)
        
        return {
            'positions': self.best_positions,
            'reward': self.best_reward,
            'quality_report': quality_report,
            'improvement_over_original': quality_report['coverage_improvement']
        }
    
    def close(self):
        """关闭环境"""
        pass


def test_environment():
    """测试环境功能"""
    logger.info("开始测试强化学习环境...")
    
    try:
        # 创建环境
        env = BusStopOptimizationEnv(
            population_csv_path="./populaiton/温州_population_grid.csv",
            bus_stops_shp_path="./公交站点shp/0577温州.shp",
            max_move_per_step=0.0005,
            max_episode_steps=20
        )
        
        # 测试重置
        obs, info = env.reset()
        logger.info(f"初始观察形状: {obs.shape}")
        logger.info(f"初始信息: {info}")
        
        # 测试几个随机动作
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            logger.info(f"Step {step + 1}:")
            logger.info(f"  奖励: {reward:.6f}")
            logger.info(f"  覆盖率: {info.get('coverage_ratio', 0):.4f}")
            logger.info(f"  改善: {info.get('coverage_improvement', 0):.4f}")
            logger.info(f"  移动距离: {info.get('total_displacement', 0):.6f}")
            
            env.render()
            
            if terminated or truncated:
                logger.info("回合结束")
                break
        
        # 获取最佳解决方案
        best_solution = env.get_best_solution()
        if best_solution:
            logger.info(f"\n最佳解决方案奖励: {best_solution['reward']:.6f}")
            quality = best_solution['quality_report']['quality_metrics']
            logger.info(f"覆盖率改善: {quality['coverage_improvement_percent']:.2f}%")
            logger.info(f"平均移动距离: {quality['average_movement_km']:.3f}km")
        
        env.close()
        logger.info("环境测试完成!")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    test_environment()