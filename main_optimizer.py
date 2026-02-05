"""
主优化训练脚本
整合所有模块，提供完整的公交站点优化解决方案
"""

import numpy as np
import pandas as pd
import logging
import time
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

# 导入所有模块
from data_preprocessing import DataProcessor
from ultra_fast_optimizer import UltraFastOptimizer
from hybrid_optimizer import HybridOptimizer
from visualization import OptimizationVisualizer
from acceleration_utils import test_acceleration_functions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MainOptimizer:
    """
    主优化器 - 集成所有优化算法和可视化功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化主优化器
        
        Args:
            config: 配置参数字典
        """
        self.config = config
        self.population_csv_path = config['data']['population_csv_path']
        self.bus_stops_shp_path = config['data']['bus_stops_shp_path']
        self.output_dir = Path(config.get('output_dir', './results'))
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载数据
        logger.info("加载和预处理数据...")
        processor = DataProcessor(self.population_csv_path, self.bus_stops_shp_path)
        self.population_data, self.bus_stops_data, self.data_analysis = processor.get_processed_data()
        
        # 创建可视化器
        self.visualizer = OptimizationVisualizer(
            self.population_data, 
            self.bus_stops_data,
            config.get('coverage_radius', 0.005)
        )
        
        # 原始位置
        self.original_positions = self.bus_stops_data[['longitude', 'latitude']].values
        
        logger.info(f"主优化器初始化完成")
        logger.info(f"数据: {len(self.population_data)} 人口点, {len(self.bus_stops_data)} 站点")
        logger.info(f"输出目录: {self.output_dir}")
    
    def run_ultra_fast_optimization(self) -> Dict[str, Any]:
        """运行超高速优化算法"""
        logger.info("=== 开始超高速优化 ===")
        
        config = self.config.get('ultra_fast', {})
        
        optimizer = UltraFastOptimizer(self.population_csv_path, self.bus_stops_shp_path)
        
        results = {}
        
        # 遗传算法
        if config.get('run_genetic_algorithm', True):
            logger.info("运行遗传算法...")
            ga_result = optimizer.genetic_algorithm_optimize(
                pop_size=config.get('ga_population_size', 100),
                generations=config.get('ga_generations', 100)
            )
            results['genetic_algorithm'] = ga_result
            
            # 保存结果
            self._save_result(ga_result, 'genetic_algorithm_result.json')
            
            # 创建可视化
            self.visualizer.create_optimization_comparison_map(
                self.original_positions,
                ga_result['best_positions'],
                ga_result,
                str(self.output_dir / 'genetic_algorithm_map.html')
            )
        
        # 模拟退火
        if config.get('run_simulated_annealing', True):
            logger.info("运行并行模拟退火...")
            sa_result = optimizer.simulated_annealing_optimize(
                n_chains=config.get('sa_chains', 8),
                iterations_per_chain=config.get('sa_iterations_per_chain', 2000)
            )
            results['simulated_annealing'] = sa_result
            
            # 保存结果
            self._save_result(sa_result, 'simulated_annealing_result.json')
            
            # 创建可视化
            self.visualizer.create_optimization_comparison_map(
                self.original_positions,
                sa_result['best_positions'],
                sa_result,
                str(self.output_dir / 'simulated_annealing_map.html')
            )
        
        # 比较算法性能
        self._compare_algorithms(results)
        
        return results
    
    def run_hybrid_optimization(self) -> Dict[str, Any]:
        """运行混合优化（PPO + Gurobi）"""
        logger.info("=== 开始混合优化（PPO + Gurobi） ===")
        
        config = self.config.get('hybrid', {})
        
        # 配置参数
        ppo_config = config.get('ppo_config', {
            'total_timesteps': 30000,  # 减少训练时间进行测试
            'learning_rate': 3e-4,
            'n_steps': 1024,
            'batch_size': 64
        })
        
        hybrid_config = config.get('hybrid_config', {
            'max_iterations': 3,  # 减少迭代次数进行测试
            'ppo_episodes_per_iteration': 10000,
            'gurobi_optimization_radius': 0.002
        })
        
        optimizer = HybridOptimizer(
            self.population_csv_path,
            self.bus_stops_shp_path,
            ppo_config=ppo_config,
            hybrid_config=hybrid_config
        )
        
        # 执行混合优化
        result = optimizer.hybrid_optimize()
        
        # 保存结果
        self._save_result(result, 'hybrid_optimization_result.json')
        
        # 创建可视化
        self.visualizer.create_optimization_comparison_map(
            self.original_positions,
            result['best_positions'],
            result,
            str(self.output_dir / 'hybrid_optimization_map.html')
        )
        
        # 创建优化历史图表
        if 'optimization_history' in result:
            self.visualizer.create_optimization_history_plot(
                result['optimization_history'],
                str(self.output_dir / 'hybrid_optimization_history.png')
            )
        
        return result
    
    def run_hyperparameter_optimization(self) -> Dict[str, Any]:
        """运行超参数优化"""
        logger.info("=== 开始超参数优化 ===")
        
        # 定义超参数搜索空间
        hyperparameter_space = {
            'ga_population_size': [50, 100, 200],
            'ga_generations': [50, 100, 150],
            'sa_chains': [4, 8, 12],
            'sa_iterations_per_chain': [1000, 2000, 3000]
        }
        
        best_config = None
        best_score = -np.inf
        all_results = []
        
        # 网格搜索（简化版本）
        configs_to_test = [
            {'ga_population_size': 100, 'ga_generations': 50},
            {'ga_population_size': 50, 'ga_generations': 100},
            {'sa_chains': 8, 'sa_iterations_per_chain': 1500},
            {'sa_chains': 4, 'sa_iterations_per_chain': 2000}
        ]
        
        for i, config in enumerate(configs_to_test):
            logger.info(f"测试配置 {i+1}/{len(configs_to_test)}: {config}")
            
            try:
                # 运行优化
                optimizer = UltraFastOptimizer(self.population_csv_path, self.bus_stops_shp_path)
                
                if 'ga_population_size' in config:
                    result = optimizer.genetic_algorithm_optimize(
                        pop_size=config['ga_population_size'],
                        generations=config['ga_generations']
                    )
                else:
                    result = optimizer.simulated_annealing_optimize(
                        n_chains=config['sa_chains'],
                        iterations_per_chain=config['sa_iterations_per_chain']
                    )
                
                # 评估结果
                score = result.get('final_coverage', result.get('best_energy', 0))
                
                result_record = {
                    'config': config,
                    'score': score,
                    'optimization_time': result['optimization_time'],
                    'method': result['method']
                }
                all_results.append(result_record)
                
                if score > best_score:
                    best_score = score
                    best_config = config
                    logger.info(f"发现更优配置！分数: {score:.6f}")
                
            except Exception as e:
                logger.error(f"配置 {config} 测试失败: {e}")
        
        # 保存超参数优化结果
        hyperopt_result = {
            'best_config': best_config,
            'best_score': best_score,
            'all_results': all_results,
            'search_space': hyperparameter_space
        }
        
        self._save_result(hyperopt_result, 'hyperparameter_optimization.json')
        
        logger.info(f"超参数优化完成！最佳配置: {best_config}")
        logger.info(f"最佳分数: {best_score:.6f}")
        
        return hyperopt_result
    
    def create_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """创建综合优化报告"""
        logger.info("生成综合优化报告...")
        
        report_path = self.output_dir / 'optimization_report.html'
        
        # HTML报告模板
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>公交站点优化报告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #2c3e50; }}
                .section {{ margin: 30px 0; padding: 20px; border-left: 4px solid #3498db; background-color: #f8f9fa; }}
                .result-box {{ background-color: white; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .metric {{ display: inline-block; margin: 10px 20px; padding: 10px; background-color: #e8f4fd; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .improvement {{ color: #27ae60; font-weight: bold; }}
                .time {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚌 公交站点优化报告</h1>
                <h3>温州市公交站点布局优化分析</h3>
                <p>生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 数据概览</h2>
                <div class="result-box">
                    <div class="metric">总人口点数: <strong>{len(self.population_data)}</strong></div>
                    <div class="metric">公交站点数: <strong>{len(self.bus_stops_data)}</strong></div>
                    <div class="metric">总人口: <strong>{self.population_data['population'].sum():.0f}</strong></div>
                    <div class="metric">服务区域: <strong>{self._calculate_service_area():.2f} km²</strong></div>
                </div>
            </div>
            
            <div class="section">
                <h2>🚀 优化结果对比</h2>
                {self._generate_results_comparison_html(results)}
            </div>
            
            <div class="section">
                <h2>📈 性能分析</h2>
                {self._generate_performance_analysis_html(results)}
            </div>
            
            <div class="section">
                <h2>🗺️ 可视化结果</h2>
                <p>以下交互式地图展示了优化结果：</p>
                <ul>
                    <li><a href="genetic_algorithm_map.html">遗传算法优化地图</a></li>
                    <li><a href="simulated_annealing_map.html">模拟退火优化地图</a></li>
                    <li><a href="hybrid_optimization_map.html">混合优化地图</a> (如果运行了混合优化)</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>💡 优化建议</h2>
                {self._generate_recommendations_html(results)}
            </div>
        </body>
        </html>
        """
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"综合报告已保存到: {report_path}")
        return str(report_path)
    
    def _save_result(self, result: Dict[str, Any], filename: str) -> None:
        """保存结果到JSON文件"""
        # 转换numpy数组为列表
        result_copy = result.copy()
        if 'best_positions' in result_copy:
            result_copy['best_positions'] = result_copy['best_positions'].tolist()
        if 'original_positions' in result_copy:
            result_copy['original_positions'] = result_copy['original_positions'].tolist()
        
        with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(result_copy, f, ensure_ascii=False, indent=2)
    
    def _compare_algorithms(self, results: Dict[str, Any]) -> None:
        """比较算法性能"""
        logger.info("\n=== 算法性能对比 ===")
        
        comparison_data = []
        for method, result in results.items():
            comparison_data.append({
                'method': method,
                'coverage': result.get('final_coverage', 0),
                'time': result.get('optimization_time', 0),
                'movement_km': result.get('total_movement_km', 0),
                'moved_stations': result.get('moved_stations', 0)
            })
        
        # 按覆盖率排序
        comparison_data.sort(key=lambda x: x['coverage'], reverse=True)
        
        logger.info("排名 | 方法 | 覆盖率 | 时间(秒) | 移动距离(km) | 移动站点数")
        logger.info("-" * 70)
        
        for i, data in enumerate(comparison_data, 1):
            logger.info(f"{i:2d} | {data['method']:20s} | {data['coverage']:.4f} | "
                       f"{data['time']:8.2f} | {data['movement_km']:11.2f} | {data['moved_stations']:9d}")
    
    def _calculate_service_area(self) -> float:
        """计算服务区域面积（平方公里）"""
        bounds = {
            'min_lon': min(self.population_data['longitude'].min(), self.bus_stops_data['longitude'].min()),
            'max_lon': max(self.population_data['longitude'].max(), self.bus_stops_data['longitude'].max()),
            'min_lat': min(self.population_data['latitude'].min(), self.bus_stops_data['latitude'].min()),
            'max_lat': max(self.population_data['latitude'].max(), self.bus_stops_data['latitude'].max())
        }
        
        # 简化的面积计算（矩形近似）
        width_km = (bounds['max_lon'] - bounds['min_lon']) * 111.32
        height_km = (bounds['max_lat'] - bounds['min_lat']) * 111.32
        
        return width_km * height_km
    
    def _generate_results_comparison_html(self, results: Dict[str, Any]) -> str:
        """生成结果对比HTML"""
        if not results:
            return "<p>没有可用的优化结果。</p>"
        
        html = "<table><tr><th>优化方法</th><th>覆盖率</th><th>优化时间</th><th>移动距离</th><th>改善程度</th></tr>"
        
        for method, result in results.items():
            coverage = result.get('final_coverage', 0)
            time_cost = result.get('optimization_time', 0)
            movement = result.get('total_movement_km', 0)
            improvement = "优" if coverage > 0.75 else "中" if coverage > 0.65 else "待改进"
            
            html += f"""
            <tr>
                <td>{method}</td>
                <td class="improvement">{coverage:.4f}</td>
                <td class="time">{time_cost:.2f}秒</td>
                <td>{movement:.2f}km</td>
                <td>{improvement}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_performance_analysis_html(self, results: Dict[str, Any]) -> str:
        """生成性能分析HTML"""
        if not results:
            return "<p>没有可用的性能数据。</p>"
        
        # 找到最佳方法
        best_method = max(results.keys(), key=lambda x: results[x].get('final_coverage', 0))
        best_result = results[best_method]
        
        return f"""
        <div class="result-box">
            <h3>🏆 最佳优化方法: {best_method}</h3>
            <div class="metric">最高覆盖率: <span class="improvement">{best_result.get('final_coverage', 0):.4f}</span></div>
            <div class="metric">优化时间: <span class="time">{best_result.get('optimization_time', 0):.2f}秒</span></div>
            <div class="metric">移动效率: {best_result.get('moved_stations', 0)} 站点移动 {best_result.get('total_movement_km', 0):.2f}km</div>
        </div>
        """
    
    def _generate_recommendations_html(self, results: Dict[str, Any]) -> str:
        """生成优化建议HTML"""
        recommendations = [
            "🎯 基于分析结果，建议优先考虑覆盖率最高的优化方案",
            "⚡ 对于实时应用，推荐使用超高速算法(遗传算法或模拟退火)",
            "🔄 对于高精度要求，推荐使用混合优化策略(PPO+Gurobi)",
            "📊 建议定期重新评估人口分布变化，动态调整站点布局",
            "🚌 考虑结合实际公交线路规划，不仅仅优化站点位置"
        ]
        
        html = "<ul>"
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ul>"
        
        return html


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        'data': {
            'population_csv_path': './populaiton/温州_population_grid.csv',
            'bus_stops_shp_path': './公交站点shp/0577温州.shp'
        },
        'output_dir': './optimization_results',
        'coverage_radius': 0.005,
        'ultra_fast': {
            'run_genetic_algorithm': True,
            'ga_population_size': 100,
            'ga_generations': 100,
            'run_simulated_annealing': True,
            'sa_chains': 8,
            'sa_iterations_per_chain': 2000
        },
        'hybrid': {
            'run_hybrid_optimization': False,  # 默认关闭，因为时间较长
            'ppo_config': {
                'total_timesteps': 50000,
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64
            },
            'hybrid_config': {
                'max_iterations': 5,
                'ppo_episodes_per_iteration': 10000,
                'gurobi_optimization_radius': 0.002
            }
        },
        'run_hyperparameter_optimization': False,  # 默认关闭
        'generate_comprehensive_report': True
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='公交站点优化系统')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--test', action='store_true', help='运行快速测试')
    parser.add_argument('--hybrid', action='store_true', help='运行混合优化')
    parser.add_argument('--hyperopt', action='store_true', help='运行超参数优化')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # 根据命令行参数调整配置
    if args.test:
        config['ultra_fast']['ga_generations'] = 20
        config['ultra_fast']['sa_iterations_per_chain'] = 500
    
    if args.hybrid:
        config['hybrid']['run_hybrid_optimization'] = True
    
    if args.hyperopt:
        config['run_hyperparameter_optimization'] = True
    
    try:
        # 测试加速函数
        logger.info("测试加速优化函数...")
        test_acceleration_functions()
        
        # 创建主优化器
        main_optimizer = MainOptimizer(config)
        
        # 运行优化
        all_results = {}
        
        # 超高速优化
        ultra_results = main_optimizer.run_ultra_fast_optimization()
        all_results.update(ultra_results)
        
        # 混合优化（可选）
        if config['hybrid'].get('run_hybrid_optimization', False):
            hybrid_result = main_optimizer.run_hybrid_optimization()
            all_results['hybrid_optimization'] = hybrid_result
        
        # 超参数优化（可选）
        if config.get('run_hyperparameter_optimization', False):
            hyperopt_result = main_optimizer.run_hyperparameter_optimization()
            all_results['hyperparameter_optimization'] = hyperopt_result
        
        # 生成综合报告
        if config.get('generate_comprehensive_report', True):
            report_path = main_optimizer.create_comprehensive_report(all_results)
            logger.info(f"综合报告生成完成: {report_path}")
        
        logger.info("🎉 所有优化任务完成!")
        logger.info(f"结果保存在: {main_optimizer.output_dir}")
        
    except Exception as e:
        logger.error(f"优化过程出错: {e}")
        raise


if __name__ == "__main__":
    # 无参数运行时使用默认配置进行快速测试
    import sys
    if len(sys.argv) == 1:
        sys.argv.append('--test')
    
    main()