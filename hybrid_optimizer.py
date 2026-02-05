"""
混合优化策略模块
结合PPO强化学习全局探索 + Gurobi精确求解局部优化
实现最高效的公交站点优化算法
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
import time
import pickle
from pathlib import Path

# 强化学习相关
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# 优化求解器
import gurobipy as gp
from gurobipy import GRB
import pulp

# 本地模块
from bus_optimization_env import BusStopOptimizationEnv
from reward_function import RewardFunction
from spatial_index import PopulationCoverageCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridOptimizer:
    """
    混合优化器：PPO探索 + Gurobi精确优化
    
    策略：
    1. 使用PPO进行全局探索，找到有潜力的区域
    2. 在PPO找到的候选解周围使用Gurobi进行局部精确优化
    3. 迭代优化，逐步提升解的质量
    """
    
    def __init__(self,
                 population_csv_path: str,
                 bus_stops_shp_path: str,
                 ppo_config: Optional[Dict] = None,
                 gurobi_config: Optional[Dict] = None,
                 hybrid_config: Optional[Dict] = None):
        """
        初始化混合优化器
        
        Args:
            population_csv_path: 人口数据路径
            bus_stops_shp_path: 公交站点数据路径
            ppo_config: PPO配置参数
            gurobi_config: Gurobi配置参数
            hybrid_config: 混合策略配置
        """
        self.population_csv_path = population_csv_path
        self.bus_stops_shp_path = bus_stops_shp_path
        
        # 配置参数
        self.ppo_config = ppo_config or self._get_default_ppo_config()
        self.gurobi_config = gurobi_config or self._get_default_gurobi_config()
        self.hybrid_config = hybrid_config or self._get_default_hybrid_config()
        
        # 初始化环境
        self.env = None
        self.ppo_model = None
        self.reward_function = None
        
        # 优化历史
        self.optimization_history = []
        self.best_solution = None
        self.best_reward = -np.inf
        
        logger.info("混合优化器初始化完成")
    
    def _get_default_ppo_config(self) -> Dict:
        """PPO默认配置"""
        return {
            'policy': 'MlpPolicy',
            'total_timesteps': 50000,
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.0,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    def _get_default_gurobi_config(self) -> Dict:
        """Gurobi默认配置"""
        return {
            'time_limit': 300,  # 5分钟时间限制
            'mip_gap': 0.01,    # 1% MIP gap
            'threads': -1,      # 使用所有可用线程
            'method': 2,        # 使用barrier方法
            'presolve': 2       # 强预处理
        }
    
    def _get_default_hybrid_config(self) -> Dict:
        """混合策略默认配置"""
        return {
            'max_iterations': 5,
            'ppo_episodes_per_iteration': 10000,
            'gurobi_optimization_radius': 0.002,  # Gurobi优化半径
            'convergence_tolerance': 1e-6,
            'use_warm_start': True,
            'save_intermediate_results': True
        }
    
    def initialize_environment(self):
        """初始化优化环境"""
        self.env = BusStopOptimizationEnv(
            self.population_csv_path,
            self.bus_stops_shp_path,
            max_move_per_step=0.0005,
            max_episode_steps=100
        )
        
        # 包装为向量环境
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        # 获取奖励函数
        self.reward_function = self.env.reward_function
        
        logger.info("优化环境初始化完成")
    
    def train_ppo_phase(self, timesteps: int) -> Tuple[np.ndarray, float]:
        """
        PPO训练阶段
        
        Args:
            timesteps: 训练时间步数
            
        Returns:
            最佳位置和对应奖励
        """
        logger.info(f"开始PPO训练阶段，时间步数: {timesteps}")
        
        # 创建或更新PPO模型
        if self.ppo_model is None:
            self.ppo_model = PPO(
                policy=self.ppo_config['policy'],
                env=self.vec_env,
                learning_rate=self.ppo_config['learning_rate'],
                n_steps=self.ppo_config['n_steps'],
                batch_size=self.ppo_config['batch_size'],
                n_epochs=self.ppo_config['n_epochs'],
                gamma=self.ppo_config['gamma'],
                gae_lambda=self.ppo_config['gae_lambda'],
                clip_range=self.ppo_config['clip_range'],
                ent_coef=self.ppo_config['ent_coef'],
                device=self.ppo_config['device'],
                verbose=1
            )
        
        # 训练PPO
        start_time = time.time()
        self.ppo_model.learn(total_timesteps=timesteps)
        training_time = time.time() - start_time
        
        # 评估训练后的模型
        best_positions, best_reward = self._evaluate_ppo_model()
        
        logger.info(f"PPO训练完成，用时: {training_time:.2f}秒")
        logger.info(f"PPO最佳奖励: {best_reward:.6f}")
        
        return best_positions, best_reward
    
    def _evaluate_ppo_model(self, n_episodes: int = 10) -> Tuple[np.ndarray, float]:
        """评估PPO模型性能"""
        best_reward = -np.inf
        best_positions = None
        
        for episode in range(n_episodes):
            obs = self.vec_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.ppo_model.predict(obs, deterministic=True)
                obs, reward, done, info = self.vec_env.step(action)
                episode_reward += reward[0]
            
            # 获取当前位置
            current_positions = self.env.current_positions
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_positions = current_positions.copy()
        
        return best_positions, best_reward
    
    def gurobi_local_optimization(self, center_positions: np.ndarray, 
                                 optimization_radius: float) -> Tuple[np.ndarray, float]:
        """
        Gurobi局部精确优化
        
        Args:
            center_positions: 优化中心位置
            optimization_radius: 优化半径
            
        Returns:
            优化后的位置和奖励
        """
        logger.info("开始Gurobi局部优化...")
        
        try:
            # 创建Gurobi模型
            model = gp.Model("BusStopOptimization")
            model.setParam('OutputFlag', 0)  # 静默模式
            
            # 应用Gurobi配置
            for param, value in self.gurobi_config.items():
                if param == 'time_limit':
                    model.setParam('TimeLimit', value)
                elif param == 'mip_gap':
                    model.setParam('MIPGap', value)
                elif param == 'threads':
                    model.setParam('Threads', value)
            
            n_stops = len(center_positions)
            
            # 决策变量：每个站点的新位置
            x_vars = {}
            y_vars = {}
            
            for i in range(n_stops):
                center_x, center_y = center_positions[i]
                
                # 位置变量，在中心周围的优化半径内
                x_vars[i] = model.addVar(
                    lb=center_x - optimization_radius,
                    ub=center_x + optimization_radius,
                    vtype=GRB.CONTINUOUS,
                    name=f"x_{i}"
                )
                
                y_vars[i] = model.addVar(
                    lb=center_y - optimization_radius,
                    ub=center_y + optimization_radius,
                    vtype=GRB.CONTINUOUS,
                    name=f"y_{i}"
                )
            
            # 添加边界约束
            bounds_lower, bounds_upper = self.reward_function.get_optimization_bounds()
            
            for i in range(n_stops):
                model.addConstr(x_vars[i] >= bounds_lower[i, 0])
                model.addConstr(x_vars[i] <= bounds_upper[i, 0])
                model.addConstr(y_vars[i] >= bounds_lower[i, 1])
                model.addConstr(y_vars[i] <= bounds_upper[i, 1])
            
            # 站点间最小距离约束
            min_distance = 0.001
            for i in range(n_stops):
                for j in range(i + 1, n_stops):
                    # 使用线性化的距离约束（近似）
                    model.addConstr(
                        (x_vars[i] - x_vars[j]) * (x_vars[i] - x_vars[j]) + 
                        (y_vars[i] - y_vars[j]) * (y_vars[i] - y_vars[j]) >= 
                        min_distance * min_distance
                    )
            
            # 目标函数：使用线性化的人口覆盖率近似
            # 这里简化为最小化与人口中心的加权距离
            pop_data = self.reward_function.coverage_calculator.population_data
            pop_centers = pop_data[['longitude', 'latitude']].values
            pop_weights = pop_data['population'].values
            
            objective_expr = 0
            for i in range(n_stops):
                for j, (px, py) in enumerate(pop_centers[:100]):  # 限制人口点数量以加速
                    weight = pop_weights[j]
                    
                    # 添加距离近似项
                    dx = x_vars[i] - px
                    dy = y_vars[i] - py
                    
                    # 使用L1距离作为近似
                    objective_expr += weight * (dx * dx + dy * dy)
            
            model.setObjective(objective_expr, GRB.MINIMIZE)
            
            # 求解
            start_time = time.time()
            model.optimize()
            solve_time = time.time() - start_time
            
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                # 提取解
                optimized_positions = np.zeros((n_stops, 2))
                for i in range(n_stops):
                    optimized_positions[i, 0] = x_vars[i].X
                    optimized_positions[i, 1] = y_vars[i].X
                
                # 计算实际奖励
                reward_dict = self.reward_function.calculate_reward(optimized_positions)
                optimized_reward = reward_dict['total_reward']
                
                logger.info(f"Gurobi优化完成，用时: {solve_time:.2f}秒")
                logger.info(f"优化状态: {model.status}")
                logger.info(f"Gurobi优化奖励: {optimized_reward:.6f}")
                
                return optimized_positions, optimized_reward
            
            else:
                logger.warning(f"Gurobi求解失败，状态: {model.status}")
                return center_positions, self.reward_function.calculate_reward(center_positions)['total_reward']
        
        except Exception as e:
            logger.error(f"Gurobi优化过程中出错: {e}")
            # 回退到PuLP求解器
            return self._pulp_fallback_optimization(center_positions, optimization_radius)
    
    def _pulp_fallback_optimization(self, center_positions: np.ndarray, 
                                   optimization_radius: float) -> Tuple[np.ndarray, float]:
        """PuLP回退优化（当Gurobi不可用时）"""
        logger.info("使用PuLP进行回退优化...")
        
        try:
            # 创建PuLP问题
            prob = pulp.LpProblem("BusStopOptimization", pulp.LpMinimize)
            
            n_stops = len(center_positions)
            
            # 决策变量
            x_vars = {}
            y_vars = {}
            
            for i in range(n_stops):
                center_x, center_y = center_positions[i]
                
                x_vars[i] = pulp.LpVariable(
                    f"x_{i}",
                    lowBound=center_x - optimization_radius,
                    upBound=center_x + optimization_radius
                )
                
                y_vars[i] = pulp.LpVariable(
                    f"y_{i}",
                    lowBound=center_y - optimization_radius,
                    upBound=center_y + optimization_radius
                )
            
            # 简化的目标函数
            pop_data = self.reward_function.coverage_calculator.population_data
            pop_centers = pop_data[['longitude', 'latitude']].values[:50]  # 限制数量
            pop_weights = pop_data['population'].values[:50]
            
            objective_expr = 0
            for i in range(n_stops):
                for j, (px, py) in enumerate(pop_centers):
                    weight = pop_weights[j]
                    
                    # 使用L1距离
                    objective_expr += weight * (
                        pulp.lpSum([x_vars[i] - px, px - x_vars[i]]) +
                        pulp.lpSum([y_vars[i] - py, py - y_vars[i]])
                    ) / 1000  # 缩放避免数值问题
            
            prob += objective_expr
            
            # 求解
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status == pulp.LpStatusOptimal:
                # 提取解
                optimized_positions = np.zeros((n_stops, 2))
                for i in range(n_stops):
                    optimized_positions[i, 0] = x_vars[i].value()
                    optimized_positions[i, 1] = y_vars[i].value()
                
                reward_dict = self.reward_function.calculate_reward(optimized_positions)
                optimized_reward = reward_dict['total_reward']
                
                logger.info(f"PuLP优化完成，奖励: {optimized_reward:.6f}")
                return optimized_positions, optimized_reward
            
            else:
                logger.warning("PuLP求解失败")
                return center_positions, self.reward_function.calculate_reward(center_positions)['total_reward']
        
        except Exception as e:
            logger.error(f"PuLP优化失败: {e}")
            return center_positions, self.reward_function.calculate_reward(center_positions)['total_reward']
    
    def hybrid_optimize(self) -> Dict[str, Any]:
        """
        执行混合优化主流程
        
        Returns:
            优化结果字典
        """
        logger.info("开始混合优化流程")
        start_time = time.time()
        
        # 初始化环境
        if self.env is None:
            self.initialize_environment()
        
        best_positions = self.env.original_positions.copy()
        best_reward = self.reward_function.calculate_reward(best_positions)['total_reward']
        
        # 迭代优化
        for iteration in range(self.hybrid_config['max_iterations']):
            logger.info(f"\n=== 混合优化迭代 {iteration + 1}/{self.hybrid_config['max_iterations']} ===")
            
            # Phase 1: PPO探索
            ppo_positions, ppo_reward = self.train_ppo_phase(
                self.hybrid_config['ppo_episodes_per_iteration']
            )
            
            # Phase 2: Gurobi局部优化
            if ppo_reward > best_reward - self.hybrid_config['convergence_tolerance']:
                gurobi_positions, gurobi_reward = self.gurobi_local_optimization(
                    ppo_positions,
                    self.hybrid_config['gurobi_optimization_radius']
                )
                
                # 更新最佳解
                if gurobi_reward > best_reward:
                    best_positions = gurobi_positions.copy()
                    best_reward = gurobi_reward
                    logger.info(f"发现更优解，奖励: {best_reward:.6f}")
            
            # 记录历史
            iteration_result = {
                'iteration': iteration + 1,
                'ppo_reward': ppo_reward,
                'gurobi_reward': gurobi_reward if 'gurobi_reward' in locals() else ppo_reward,
                'best_reward': best_reward,
                'positions': best_positions.copy()
            }
            self.optimization_history.append(iteration_result)
            
            # 检查收敛
            if iteration > 0:
                improvement = best_reward - self.optimization_history[-2]['best_reward']
                if abs(improvement) < self.hybrid_config['convergence_tolerance']:
                    logger.info(f"算法收敛，改进幅度: {improvement:.8f}")
                    break
        
        total_time = time.time() - start_time
        
        # 最终质量评估
        final_quality = self.reward_function.evaluate_solution_quality(best_positions)
        
        # 构建结果
        optimization_result = {
            'best_positions': best_positions,
            'best_reward': best_reward,
            'original_positions': self.env.original_positions,
            'optimization_time': total_time,
            'iterations': len(self.optimization_history),
            'quality_report': final_quality,
            'optimization_history': self.optimization_history,
            'config': {
                'ppo_config': self.ppo_config,
                'gurobi_config': self.gurobi_config,
                'hybrid_config': self.hybrid_config
            }
        }
        
        # 保存结果
        if self.hybrid_config['save_intermediate_results']:
            self._save_results(optimization_result)
        
        logger.info(f"\n混合优化完成!")
        logger.info(f"总用时: {total_time:.2f}秒")
        logger.info(f"最终奖励: {best_reward:.6f}")
        logger.info(f"覆盖率改善: {final_quality['quality_metrics']['coverage_improvement_percent']:.2f}%")
        
        return optimization_result
    
    def _save_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """保存优化结果"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"hybrid_optimization_results_{timestamp}.pkl"
        
        save_path = Path(filename)
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"优化结果已保存到: {save_path}")


def test_hybrid_optimizer():
    """测试混合优化器"""
    logger.info("开始测试混合优化器...")
    
    try:
        # 创建混合优化器（使用较少的训练步数进行快速测试）
        hybrid_config = {
            'max_iterations': 2,
            'ppo_episodes_per_iteration': 5000,
            'gurobi_optimization_radius': 0.001,
            'convergence_tolerance': 1e-4,
            'use_warm_start': True,
            'save_intermediate_results': True
        }
        
        optimizer = HybridOptimizer(
            population_csv_path="./populaiton/温州_population_grid.csv",
            bus_stops_shp_path="./公交站点shp/0577温州.shp",
            hybrid_config=hybrid_config
        )
        
        # 执行优化
        results = optimizer.hybrid_optimize()
        
        # 显示结果摘要
        logger.info("\n=== 优化结果摘要 ===")
        logger.info(f"最佳奖励: {results['best_reward']:.6f}")
        logger.info(f"优化时间: {results['optimization_time']:.2f}秒")
        logger.info(f"迭代次数: {results['iterations']}")
        
        quality = results['quality_report']['quality_metrics']
        logger.info(f"覆盖率改善: {quality['coverage_improvement_percent']:.2f}%")
        logger.info(f"平均移动距离: {quality['average_movement_km']:.3f}km")
        logger.info(f"移动站点数: {quality['moved_stations_count']}")
        
        logger.info("混合优化器测试完成!")
        
        return results
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    test_hybrid_optimizer()