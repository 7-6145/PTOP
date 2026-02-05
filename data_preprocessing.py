"""
数据预处理模块
加载和分析人口网格数据和公交站点数据
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """数据预处理类"""
    
    def __init__(self, population_csv_path: str, bus_stops_shp_path: str):
        self.population_csv_path = population_csv_path
        self.bus_stops_shp_path = bus_stops_shp_path
        
        self.population_data = None
        self.bus_stops_data = None
        self.population_gdf = None
        
    def load_population_data(self) -> pd.DataFrame:
        """加载人口网格数据"""
        logger.info("加载人口网格数据...")
        
        self.population_data = pd.read_csv(self.population_csv_path)
        
        # 数据基本信息
        logger.info(f"人口数据形状: {self.population_data.shape}")
        logger.info(f"人口统计: 总人口 {self.population_data['population'].sum():.2f}")
        logger.info(f"经度范围: {self.population_data['longitude'].min():.4f} - {self.population_data['longitude'].max():.4f}")
        logger.info(f"纬度范围: {self.population_data['latitude'].min():.4f} - {self.population_data['latitude'].max():.4f}")
        
        # 转换为GeoDataFrame
        geometry = gpd.points_from_xy(self.population_data.longitude, self.population_data.latitude)
        self.population_gdf = gpd.GeoDataFrame(self.population_data, geometry=geometry, crs='EPSG:4326')
        
        return self.population_data
    
    def load_bus_stops_data(self) -> gpd.GeoDataFrame:
        """加载公交站点数据"""
        logger.info("加载公交站点数据...")
        
        try:
            self.bus_stops_data = gpd.read_file(self.bus_stops_shp_path)
            
            # 确保坐标系统一
            if self.bus_stops_data.crs != 'EPSG:4326':
                self.bus_stops_data = self.bus_stops_data.to_crs('EPSG:4326')
            
            logger.info(f"公交站点数量: {len(self.bus_stops_data)}")
            logger.info(f"公交站点字段: {list(self.bus_stops_data.columns)}")
            
            # 提取坐标
            self.bus_stops_data['longitude'] = self.bus_stops_data.geometry.x
            self.bus_stops_data['latitude'] = self.bus_stops_data.geometry.y
            
            logger.info(f"站点经度范围: {self.bus_stops_data['longitude'].min():.4f} - {self.bus_stops_data['longitude'].max():.4f}")
            logger.info(f"站点纬度范围: {self.bus_stops_data['latitude'].min():.4f} - {self.bus_stops_data['latitude'].max():.4f}")
            
            return self.bus_stops_data
            
        except Exception as e:
            logger.error(f"加载公交站点数据失败: {e}")
            raise
    
    def analyze_data_overlap(self) -> Dict[str, Any]:
        """分析数据重叠区域"""
        if self.population_data is None or self.bus_stops_data is None:
            raise ValueError("请先加载数据")
        
        # 计算边界框
        pop_bounds = {
            'min_lon': self.population_data['longitude'].min(),
            'max_lon': self.population_data['longitude'].max(),
            'min_lat': self.population_data['latitude'].min(),
            'max_lat': self.population_data['latitude'].max()
        }
        
        stop_bounds = {
            'min_lon': self.bus_stops_data['longitude'].min(),
            'max_lon': self.bus_stops_data['longitude'].max(),
            'min_lat': self.bus_stops_data['latitude'].min(),
            'max_lat': self.bus_stops_data['latitude'].max()
        }
        
        # 计算重叠区域
        overlap_bounds = {
            'min_lon': max(pop_bounds['min_lon'], stop_bounds['min_lon']),
            'max_lon': min(pop_bounds['max_lon'], stop_bounds['max_lon']),
            'min_lat': max(pop_bounds['min_lat'], stop_bounds['min_lat']),
            'max_lat': min(pop_bounds['max_lat'], stop_bounds['max_lat'])
        }
        
        # 检查是否有有效重叠
        has_overlap = (overlap_bounds['max_lon'] > overlap_bounds['min_lon'] and 
                      overlap_bounds['max_lat'] > overlap_bounds['min_lat'])
        
        analysis = {
            'population_bounds': pop_bounds,
            'bus_stops_bounds': stop_bounds,
            'overlap_bounds': overlap_bounds,
            'has_overlap': has_overlap,
            'population_count': len(self.population_data),
            'bus_stops_count': len(self.bus_stops_data),
            'total_population': self.population_data['population'].sum()
        }
        
        logger.info(f"数据重叠分析: {has_overlap}")
        if has_overlap:
            logger.info(f"重叠区域: 经度 {overlap_bounds['min_lon']:.4f} - {overlap_bounds['max_lon']:.4f}")
            logger.info(f"重叠区域: 纬度 {overlap_bounds['min_lat']:.4f} - {overlap_bounds['max_lat']:.4f}")
        
        return analysis
    
    def filter_to_overlap_region(self, analysis: Dict[str, Any]) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """筛选重叠区域的数据"""
        if not analysis['has_overlap']:
            logger.warning("没有数据重叠，使用全部数据")
            return self.population_data, self.bus_stops_data
        
        bounds = analysis['overlap_bounds']
        
        # 筛选人口数据
        pop_mask = (
            (self.population_data['longitude'] >= bounds['min_lon']) &
            (self.population_data['longitude'] <= bounds['max_lon']) &
            (self.population_data['latitude'] >= bounds['min_lat']) &
            (self.population_data['latitude'] <= bounds['max_lat'])
        )
        filtered_population = self.population_data[pop_mask].copy()
        
        # 筛选公交站点数据
        stop_mask = (
            (self.bus_stops_data['longitude'] >= bounds['min_lon']) &
            (self.bus_stops_data['longitude'] <= bounds['max_lon']) &
            (self.bus_stops_data['latitude'] >= bounds['min_lat']) &
            (self.bus_stops_data['latitude'] <= bounds['max_lat'])
        )
        filtered_stops = self.bus_stops_data[stop_mask].copy()
        
        logger.info(f"筛选后人口网格数: {len(filtered_population)}")
        logger.info(f"筛选后公交站点数: {len(filtered_stops)}")
        logger.info(f"筛选区域总人口: {filtered_population['population'].sum():.2f}")
        
        return filtered_population, filtered_stops
    
    def get_processed_data(self) -> Tuple[pd.DataFrame, gpd.GeoDataFrame, Dict[str, Any]]:
        """获取预处理后的数据"""
        # 加载数据
        self.load_population_data()
        self.load_bus_stops_data()
        
        # 分析重叠区域
        analysis = self.analyze_data_overlap()
        
        # 筛选到重叠区域
        population_data, bus_stops_data = self.filter_to_overlap_region(analysis)
        
        return population_data, bus_stops_data, analysis


def main():
    """主函数 - 测试数据加载"""
    
    # 数据路径
    population_path = "./populaiton/温州_population_grid.csv"
    bus_stops_path = "./公交站点shp/0577温州.shp"
    
    # 创建数据处理器
    processor = DataProcessor(population_path, bus_stops_path)
    
    try:
        # 处理数据
        population_data, bus_stops_data, analysis = processor.get_processed_data()
        
        logger.info("数据预处理完成!")
        logger.info(f"有效人口网格: {len(population_data)}")
        logger.info(f"有效公交站点: {len(bus_stops_data)}")
        
        # 保存预处理结果
        population_data.to_csv("processed_population_data.csv", index=False)
        bus_stops_data.to_file("processed_bus_stops.shp")
        
        logger.info("预处理数据已保存")
        
        return population_data, bus_stops_data, analysis
        
    except Exception as e:
        logger.error(f"数据预处理失败: {e}")
        raise


if __name__ == "__main__":
    main()