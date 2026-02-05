"""
可视化模块
使用folium创建交互式地图，展示优化过程和结果
"""

import numpy as np
import pandas as pd
import folium
from folium import plugins
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import seaborn as sns
import logging
from typing import Dict, Any, List, Optional, Tuple
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationVisualizer:
    """优化结果可视化器"""
    
    def __init__(self, population_data: pd.DataFrame, 
                 bus_stops_data: pd.DataFrame,
                 coverage_radius: float = 0.005):
        """
        初始化可视化器
        
        Args:
            population_data: 人口数据
            bus_stops_data: 公交站点数据  
            coverage_radius: 站点覆盖半径
        """
        self.population_data = population_data
        self.bus_stops_data = bus_stops_data
        self.coverage_radius = coverage_radius
        
        # 计算地图中心和边界
        all_lons = np.concatenate([
            population_data['longitude'].values,
            bus_stops_data['longitude'].values
        ])
        all_lats = np.concatenate([
            population_data['latitude'].values,
            bus_stops_data['latitude'].values
        ])
        
        self.center_lat = np.mean(all_lats)
        self.center_lon = np.mean(all_lons)
        self.bounds = {
            'min_lat': np.min(all_lats),
            'max_lat': np.max(all_lats),
            'min_lon': np.min(all_lons),
            'max_lon': np.max(all_lons)
        }
        
        logger.info(f"可视化器初始化完成，地图中心: ({self.center_lat:.4f}, {self.center_lon:.4f})")
    
    def create_optimization_comparison_map(self, 
                                         original_positions: np.ndarray,
                                         optimized_positions: np.ndarray,
                                         optimization_result: Dict[str, Any],
                                         filename: str = "optimization_comparison.html") -> folium.Map:
        """
        创建优化前后对比地图
        
        Args:
            original_positions: 原始站点位置
            optimized_positions: 优化后站点位置
            optimization_result: 优化结果字典
            filename: 保存文件名
            
        Returns:
            folium地图对象
        """
        logger.info("创建优化前后对比地图...")
        
        # 创建基础地图
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # 添加人口密度热力图
        self._add_population_heatmap(m)
        
        # 添加原始站点
        self._add_bus_stops(m, original_positions, 'original', color='red')
        
        # 添加优化后站点
        self._add_bus_stops(m, optimized_positions, 'optimized', color='blue')
        
        # 添加移动轨迹
        self._add_movement_arrows(m, original_positions, optimized_positions)
        
        # 添加覆盖范围（可选择显示）
        coverage_group = folium.FeatureGroup(name="覆盖范围 (点击查看)", show=False)
        self._add_coverage_circles(coverage_group, optimized_positions, color='blue', alpha=0.2)
        m.add_child(coverage_group)
        
        # 添加优化结果信息
        self._add_optimization_info_panel(m, optimization_result)
        
        # 添加图层控制
        folium.LayerControl(collapsed=False).add_to(m)
        
        # 保存地图
        m.save(filename)
        logger.info(f"地图已保存到: {filename}")
        
        return m
    
    def create_coverage_analysis_map(self, 
                                   positions: np.ndarray,
                                   filename: str = "coverage_analysis.html") -> folium.Map:
        """
        创建覆盖范围分析地图
        """
        logger.info("创建覆盖范围分析地图...")
        
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=12,
            tiles='CartoDB positron'
        )
        
        # 添加人口点（按权重着色）
        self._add_population_points(m, show_weights=True)
        
        # 添加站点和覆盖范围
        self._add_bus_stops(m, positions, 'stations', color='red', show_coverage=True)
        
        # 计算并显示覆盖统计
        coverage_stats = self._calculate_coverage_statistics(positions)
        self._add_coverage_statistics_panel(m, coverage_stats)
        
        m.save(filename)
        logger.info(f"覆盖分析地图已保存到: {filename}")
        
        return m
    
    def create_optimization_history_plot(self, 
                                       history: List[Dict],
                                       filename: str = "optimization_history.png") -> None:
        """创建优化历史图表"""
        logger.info("创建优化历史图表...")
        
        if not history:
            logger.warning("没有历史数据可绘制")
            return
        
        # 提取数据
        iterations = [h.get('iteration', i) for i, h in enumerate(history)]
        rewards = [h.get('reward', h.get('fitness', h.get('energy', 0))) for h in history]
        coverages = [h.get('coverage_ratio', 0) for h in history if 'coverage_ratio' in h]
        movements = [h.get('total_movement', 0) for h in history if 'total_movement' in h]
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('优化过程分析', fontsize=16, fontweight='bold')
        
        # 奖励/适应度曲线
        axes[0, 0].plot(iterations, rewards, 'b-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_title('奖励/适应度变化')
        axes[0, 0].set_xlabel('迭代次数')
        axes[0, 0].set_ylabel('奖励值')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 覆盖率变化（如果有数据）
        if coverages and len(coverages) > 1:
            axes[0, 1].plot(iterations[:len(coverages)], coverages, 'g-', linewidth=2, marker='s', markersize=4)
            axes[0, 1].set_title('人口覆盖率变化')
            axes[0, 1].set_xlabel('迭代次数')
            axes[0, 1].set_ylabel('覆盖率')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, '覆盖率数据不足', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('人口覆盖率变化')
        
        # 移动距离变化（如果有数据）
        if movements and len(movements) > 1:
            axes[1, 0].plot(iterations[:len(movements)], movements, 'r-', linewidth=2, marker='^', markersize=4)
            axes[1, 0].set_title('总移动距离变化')
            axes[1, 0].set_xlabel('迭代次数')
            axes[1, 0].set_ylabel('移动距离')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, '移动距离数据不足', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('总移动距离变化')
        
        # 收敛性分析
        if len(rewards) > 10:
            # 计算滑动平均
            window_size = min(10, len(rewards) // 4)
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            moving_iterations = iterations[window_size-1:]
            
            axes[1, 1].plot(iterations, rewards, 'lightblue', alpha=0.6, label='原始数据')
            axes[1, 1].plot(moving_iterations, moving_avg, 'navy', linewidth=3, label=f'{window_size}点滑动平均')
            axes[1, 1].set_title('收敛性分析')
            axes[1, 1].set_xlabel('迭代次数')
            axes[1, 1].set_ylabel('奖励值')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, '数据不足进行收敛分析', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('收敛性分析')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"优化历史图表已保存到: {filename}")
    
    def _add_population_heatmap(self, m: folium.Map) -> None:
        """添加人口密度热力图"""
        heat_data = []
        for _, row in self.population_data.iterrows():
            # [纬度, 经度, 权重]
            heat_data.append([row['latitude'], row['longitude'], row['population']])
        
        heatmap = plugins.HeatMap(
            heat_data,
            name="人口密度热力图",
            min_opacity=0.2,
            max_zoom=18,
            radius=15,
            blur=10,
            gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
        )
        
        m.add_child(heatmap)
    
    def _add_population_points(self, m: folium.Map, show_weights: bool = False) -> None:
        """添加人口点"""
        if show_weights:
            # 按人口权重着色
            pop_max = self.population_data['population'].max()
            
            for _, row in self.population_data.iterrows():
                weight_ratio = row['population'] / pop_max
                color = plt.cm.YlOrRd(weight_ratio)  # 黄到红的渐变
                color_hex = mcolors.rgb2hex(color)
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3 + weight_ratio * 5,  # 半径随权重变化
                    color=color_hex,
                    fill=True,
                    fillOpacity=0.6,
                    popup=f"人口: {row['population']:.1f}",
                    tooltip=f"人口密度: {row['population']:.1f}"
                ).add_to(m)
    
    def _add_bus_stops(self, m: folium.Map, positions: np.ndarray, 
                      group_name: str, color: str = 'red', show_coverage: bool = False) -> None:
        """添加公交站点"""
        feature_group = folium.FeatureGroup(name=f"{group_name} 站点")
        
        for i, (lon, lat) in enumerate(positions):
            # 站点标记
            folium.Marker(
                location=[lat, lon],
                popup=f"{group_name} 站点 {i+1}<br>坐标: ({lon:.6f}, {lat:.6f})",
                tooltip=f"站点 {i+1}",
                icon=folium.Icon(color=color, icon='bus', prefix='fa')
            ).add_to(feature_group)
            
            # 覆盖范围圆圈
            if show_coverage:
                folium.Circle(
                    location=[lat, lon],
                    radius=self.coverage_radius * 111320,  # 转换为米
                    color=color,
                    fillColor=color,
                    fillOpacity=0.1,
                    weight=1,
                    popup=f"站点 {i+1} 覆盖范围"
                ).add_to(feature_group)
        
        m.add_child(feature_group)
    
    def _add_coverage_circles(self, feature_group: folium.FeatureGroup, 
                            positions: np.ndarray, color: str = 'blue', alpha: float = 0.2) -> None:
        """添加覆盖范围圆圈"""
        for i, (lon, lat) in enumerate(positions):
            folium.Circle(
                location=[lat, lon],
                radius=self.coverage_radius * 111320,  # 转换为米
                color=color,
                fillColor=color,
                fillOpacity=alpha,
                weight=2,
                popup=f"站点 {i+1} 服务范围",
                tooltip=f"覆盖半径: {self.coverage_radius * 111.32:.1f}km"
            ).add_to(feature_group)
    
    def _add_movement_arrows(self, m: folium.Map, 
                           original_positions: np.ndarray, 
                           optimized_positions: np.ndarray) -> None:
        """添加移动轨迹箭头"""
        movement_group = folium.FeatureGroup(name="站点移动轨迹")
        
        for i, (orig, opt) in enumerate(zip(original_positions, optimized_positions)):
            # 计算移动距离
            distance = np.sqrt((opt[0] - orig[0])**2 + (opt[1] - orig[1])**2) * 111.32  # km
            
            # 只显示显著移动的站点（>10米）
            if distance > 0.01:  
                # 移动轨迹线
                folium.PolyLine(
                    locations=[[orig[1], orig[0]], [opt[1], opt[0]]],
                    color='purple',
                    weight=3,
                    opacity=0.8,
                    popup=f"站点 {i+1} 移动 {distance:.2f}km",
                    tooltip=f"移动距离: {distance:.2f}km"
                ).add_to(movement_group)
                
                # 箭头标记（在终点）
                folium.Marker(
                    location=[opt[1], opt[0]],
                    icon=folium.Icon(color='purple', icon='arrow-up', prefix='fa'),
                    popup=f"站点 {i+1} 新位置<br>移动距离: {distance:.2f}km"
                ).add_to(movement_group)
        
        m.add_child(movement_group)
    
    def _add_optimization_info_panel(self, m: folium.Map, result: Dict[str, Any]) -> None:
        """添加优化结果信息面板"""
        info_html = f"""
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 300px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px; border-radius: 10px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.2);">
        <h4><b>优化结果摘要</b></h4>
        <p><b>算法:</b> {result.get('method', '未知')}</p>
        <p><b>优化时间:</b> {result.get('optimization_time', 0):.2f} 秒</p>
        <p><b>最终覆盖率:</b> {result.get('final_coverage', 0):.4f}</p>
        <p><b>总移动距离:</b> {result.get('total_movement_km', 0):.2f} km</p>
        <p><b>移动站点数:</b> {result.get('moved_stations', 0)}</p>
        <p><b>最佳分数:</b> {result.get('best_fitness', result.get('best_energy', 0)):.6f}</p>
        </div>
        """
        
        m.get_root().html.add_child(folium.Element(info_html))
    
    def _calculate_coverage_statistics(self, positions: np.ndarray) -> Dict[str, Any]:
        """计算覆盖统计信息"""
        from acceleration_utils import fast_population_coverage
        
        coverage_ratio = fast_population_coverage(
            positions, 
            self.population_data[['longitude', 'latitude']].values,
            self.population_data['population'].values,
            self.coverage_radius
        )
        
        total_population = self.population_data['population'].sum()
        covered_population = coverage_ratio * total_population
        
        return {
            'coverage_ratio': coverage_ratio,
            'total_population': total_population,
            'covered_population': covered_population,
            'uncovered_population': total_population - covered_population,
            'n_stations': len(positions)
        }
    
    def _add_coverage_statistics_panel(self, m: folium.Map, stats: Dict[str, Any]) -> None:
        """添加覆盖统计信息面板"""
        info_html = f"""
        <div style="position: fixed; 
                    bottom: 10px; right: 10px; width: 300px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px; border-radius: 10px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.2);">
        <h4><b>覆盖统计</b></h4>
        <p><b>总人口:</b> {stats['total_population']:.0f}</p>
        <p><b>已覆盖人口:</b> {stats['covered_population']:.0f}</p>
        <p><b>未覆盖人口:</b> {stats['uncovered_population']:.0f}</p>
        <p><b>覆盖率:</b> {stats['coverage_ratio']:.2%}</p>
        <p><b>站点数量:</b> {stats['n_stations']}</p>
        </div>
        """
        
        m.get_root().html.add_child(folium.Element(info_html))


def test_visualization():
    """测试可视化功能"""
    logger.info("开始测试可视化功能...")
    
    try:
        from data_preprocessing import DataProcessor
        
        # 加载数据
        processor = DataProcessor(
            "./populaiton/温州_population_grid.csv",
            "./公交站点shp/0577温州.shp"
        )
        pop_data, bus_data, _ = processor.get_processed_data()
        
        # 创建可视化器
        visualizer = OptimizationVisualizer(pop_data, bus_data)
        
        # 生成模拟优化结果
        original_positions = bus_data[['longitude', 'latitude']].values
        optimized_positions = original_positions.copy()
        
        # 随机移动几个站点进行演示
        n_moved = min(5, len(original_positions))
        indices_to_move = np.random.choice(len(original_positions), n_moved, replace=False)
        
        for idx in indices_to_move:
            perturbation = np.random.normal(0, 0.002, 2)
            optimized_positions[idx] += perturbation
        
        # 模拟优化结果
        mock_result = {
            'method': 'visualization_test',
            'optimization_time': 15.5,
            'final_coverage': 0.8234,
            'total_movement_km': 2.45,
            'moved_stations': n_moved,
            'best_fitness': 0.7123
        }
        
        # 创建对比地图
        comparison_map = visualizer.create_optimization_comparison_map(
            original_positions, optimized_positions, mock_result,
            "test_comparison_map.html"
        )
        
        # 创建覆盖分析地图
        coverage_map = visualizer.create_coverage_analysis_map(
            optimized_positions, "test_coverage_map.html"
        )
        
        # 创建模拟优化历史图表
        mock_history = []
        for i in range(50):
            mock_history.append({
                'iteration': i + 1,
                'reward': 0.5 + 0.3 * np.exp(-i/20) + 0.02 * np.random.randn(),
                'coverage_ratio': 0.7 + 0.1 * (1 - np.exp(-i/30)) + 0.01 * np.random.randn(),
                'total_movement': 5.0 * np.exp(-i/25) + 0.1 * np.random.randn()
            })
        
        visualizer.create_optimization_history_plot(mock_history, "test_optimization_history.png")
        
        logger.info("可视化测试完成!")
        logger.info("生成的文件:")
        logger.info("- test_comparison_map.html (对比地图)")
        logger.info("- test_coverage_map.html (覆盖分析地图)")
        logger.info("- test_optimization_history.png (优化历史图表)")
        
        return True
        
    except Exception as e:
        logger.error(f"可视化测试失败: {e}")
        raise


if __name__ == "__main__":
    test_visualization()