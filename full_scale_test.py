"""
全规模优化测试脚本
提供不同规模的测试选项，确保系统稳定运行
"""

import numpy as np
import pandas as pd
import logging
import time
import psutil
import multiprocessing as mp
from pathlib import Path
import argparse

from full_scale_fixed import QuickFullScaleOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局工作函数，避免pickle问题
def simple_test_worker(x):
    """简单测试工作函数"""
    return x * x

class FullScaleTestManager:
    """全规模优化测试管理器"""
    
    def __init__(self):
        """初始化测试管理器"""
        self.system_info = self._get_system_info()
        self._display_system_info()
    
    def _get_system_info(self) -> dict:
        """获取系统信息"""
        memory = psutil.virtual_memory()
        return {
            'cpu_count': mp.cpu_count(),
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'memory_percent': memory.percent
        }
    
    def _display_system_info(self):
        """显示系统信息"""
        info = self.system_info
        logger.info(f"🖥️  系统配置检查:")
        logger.info(f"   CPU核心: {info['cpu_count']}")
        logger.info(f"   内存总量: {info['memory_total_gb']:.1f}GB")
        logger.info(f"   可用内存: {info['memory_available_gb']:.1f}GB ({100-info['memory_percent']:.1f}%)")
        
        # 给出建议
        if info['memory_available_gb'] < 4:
            logger.warning("⚠️  可用内存不足4GB，建议使用小规模测试模式")
        elif info['memory_available_gb'] < 8:
            logger.info("💡 可用内存适中，建议使用中规模测试模式")
        else:
            logger.info("✅ 内存充足，可以运行全规模优化")
    
    def quick_test(self) -> dict:
        """快速测试：验证系统功能"""
        logger.info("🚀 === 快速功能测试 ===")
        
        try:
            from data_preprocessing import DataProcessor
            
            # 快速数据加载测试
            logger.info("📊 测试数据加载...")
            processor = DataProcessor(
                "./populaiton/温州_population_grid.csv",
                "./公交站点shp/0577温州.shp"
            )
            pop_data, stop_data, _ = processor.get_processed_data()
            
            logger.info(f"✅ 数据加载成功: {len(pop_data)}人口点, {len(stop_data)}站点")
            
            # 内存使用测试
            memory_usage = psutil.virtual_memory().percent
            logger.info(f"💾 内存使用率: {memory_usage:.1f}%")
            
            # 多进程测试
            logger.info("🔄 测试多进程环境...")
            
            with mp.Pool(processes=min(2, mp.cpu_count())) as pool:
                results = pool.map(simple_test_worker, range(10))
            
            logger.info("✅ 多进程测试成功")
            
            return {
                'status': 'success',
                'data_size': {'population': len(pop_data), 'stops': len(stop_data)},
                'memory_usage': memory_usage,
                'multiprocessing_ok': True
            }
            
        except Exception as e:
            logger.error(f"❌ 快速测试失败: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def small_scale_test(self) -> dict:
        """小规模测试：处理部分数据"""
        logger.info("🎯 === 小规模优化测试 ===")
        
        try:
            # 创建限制版本的优化器
            optimizer = QuickFullScaleOptimizer(
                "./populaiton/温州_population_grid.csv",
                "./公交站点shp/0577温州.shp"
            )
            
            # 限制处理规模
            original_blocks = optimizer.spatial_blocks
            optimizer.spatial_blocks = original_blocks[:min(4, len(original_blocks))]  # 最多4个区域
            optimizer.n_processes = min(2, mp.cpu_count())  # 最多2个进程
            
            logger.info(f"📦 限制处理规模: {len(optimizer.spatial_blocks)} 个区域")
            
            # 运行优化
            start_time = time.time()
            result = optimizer.optimize_full_scale()
            test_time = time.time() - start_time
            
            logger.info(f"✅ 小规模测试完成，用时 {test_time:.1f}秒")
            
            # 预估全规模时间
            total_blocks = len(original_blocks)
            estimated_full_time = test_time * (total_blocks / len(optimizer.spatial_blocks)) / optimizer.n_processes
            
            logger.info(f"⏱️  预估全规模时间: {estimated_full_time:.1f}秒 ({estimated_full_time/60:.1f}分钟)")
            
            return {
                'status': 'success',
                'test_time': test_time,
                'estimated_full_time': estimated_full_time,
                'blocks_tested': len(optimizer.spatial_blocks),
                'total_blocks': total_blocks,
                'result': result['global_metrics']
            }
            
        except Exception as e:
            logger.error(f"❌ 小规模测试失败: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def medium_scale_test(self) -> dict:
        """中规模测试：处理大部分数据"""
        logger.info("🎯 === 中规模优化测试 ===")
        
        try:
            optimizer = QuickFullScaleOptimizer(
                "./populaiton/温州_population_grid.csv",
                "./公交站点shp/0577温州.shp"
            )
            
            # 中等规模限制
            original_blocks = optimizer.spatial_blocks
            max_blocks = min(len(original_blocks), max(8, len(original_blocks) // 2))
            optimizer.spatial_blocks = original_blocks[:max_blocks]
            optimizer.n_processes = min(4, mp.cpu_count())
            
            logger.info(f"📦 处理规模: {len(optimizer.spatial_blocks)}/{len(original_blocks)} 个区域")
            
            # 运行优化
            start_time = time.time()
            result = optimizer.optimize_full_scale()
            test_time = time.time() - start_time
            
            # 创建可视化 (QuickFullScaleOptimizer 暂不支持)
            # map_path = optimizer.create_full_scale_visualization(result, sample_ratio=0.15)
            map_path = "暂不支持可视化"
            
            logger.info(f"✅ 中规模测试完成，用时 {test_time:.1f}秒")
            # logger.info(f"🗺️  可视化地图: {map_path}")
            
            return {
                'status': 'success',
                'test_time': test_time,
                'blocks_processed': len(optimizer.spatial_blocks),
                'total_blocks': len(original_blocks),
                'result': result['global_metrics'],
                'map_path': map_path
            }
            
        except Exception as e:
            logger.error(f"❌ 中规模测试失败: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_full_scale_with_monitoring(self) -> dict:
        """运行全规模优化（带监控）"""
        logger.info("🚀 === 全规模优化（监控模式） ===")
        
        try:
            # 预检查
            if self.system_info['memory_available_gb'] < 6:
                logger.warning("⚠️  内存可能不足，建议先运行中规模测试")
                return {'status': 'skipped', 'reason': 'insufficient_memory'}
            
            # 创建优化器
            optimizer = QuickFullScaleOptimizer(
                "./populaiton/温州_population_grid.csv",
                "./公交站点shp/0577温州.shp"
            )
            
            # 性能监控
            start_memory = psutil.virtual_memory().percent
            start_time = time.time()
            
            logger.info(f"🔍 开始监控 - 初始内存使用: {start_memory:.1f}%")
            
            # 运行优化
            result = optimizer.optimize_full_scale()
            
            end_time = time.time()
            end_memory = psutil.virtual_memory().percent
            
            # 创建完整可视化 (QuickFullScaleOptimizer 暂不支持)
            # map_path = optimizer.create_full_scale_visualization(result, sample_ratio=0.2)
            map_path = "暂不支持可视化"
            
            logger.info(f"✅ 全规模优化完成!")
            logger.info(f"⏱️  总用时: {end_time - start_time:.1f}秒")
            logger.info(f"💾 内存变化: {start_memory:.1f}% → {end_memory:.1f}%")
            # logger.info(f"🗺️  完整地图: {map_path}")
            
            return {
                'status': 'success',
                'total_time': end_time - start_time,
                'memory_change': end_memory - start_memory,
                'result': result,
                'map_path': map_path
            }
            
        except Exception as e:
            logger.error(f"❌ 全规模优化失败: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_progressive_test(self) -> dict:
        """渐进式测试：从小到大逐步测试"""
        logger.info("📈 === 渐进式测试流程 ===")
        
        results = {}
        
        # 1. 快速测试
        logger.info("\n🔸 步骤1: 快速功能测试")
        quick_result = self.quick_test()
        results['quick_test'] = quick_result
        
        if quick_result['status'] != 'success':
            logger.error("❌ 快速测试失败，停止后续测试")
            return results
        
        # 2. 小规模测试
        logger.info("\n🔸 步骤2: 小规模优化测试")
        small_result = self.small_scale_test()
        results['small_scale'] = small_result
        
        if small_result['status'] != 'success':
            logger.error("❌ 小规模测试失败，停止后续测试")
            return results
        
        # 3. 根据结果决定是否进行更大规模测试
        estimated_time = small_result.get('estimated_full_time', 0)
        
        if estimated_time > 1800:  # 超过30分钟
            logger.warning(f"⚠️  预估全规模时间过长({estimated_time/60:.1f}分钟)，建议使用中规模测试")
            
            logger.info("\n🔸 步骤3: 中规模优化测试")
            medium_result = self.medium_scale_test()
            results['medium_scale'] = medium_result
            
        else:
            logger.info(f"✅ 预估时间合理({estimated_time/60:.1f}分钟)，可以进行全规模优化")
            
            logger.info("\n🔸 步骤3: 全规模优化")
            full_result = self.run_full_scale_with_monitoring()
            results['full_scale'] = full_result
        
        # 总结
        self._display_progressive_summary(results)
        
        return results
    
    def _display_progressive_summary(self, results: dict):
        """显示渐进式测试总结"""
        logger.info(f"\n📋 === 渐进式测试总结 ===")
        
        for test_name, result in results.items():
            if result['status'] == 'success':
                logger.info(f"✅ {test_name}: 成功")
                if 'test_time' in result:
                    logger.info(f"   ⏱️  用时: {result['test_time']:.1f}秒")
            else:
                logger.info(f"❌ {test_name}: 失败")
        
        # 推荐
        if 'full_scale' in results and results['full_scale']['status'] == 'success':
            logger.info("\n🏆 推荐: 全规模优化已成功完成！")
        elif 'medium_scale' in results and results['medium_scale']['status'] == 'success':
            logger.info("\n💡 推荐: 中规模测试表现良好，可以尝试全规模优化")
        elif 'small_scale' in results and results['small_scale']['status'] == 'success':
            logger.info("\n⚠️  推荐: 建议优化系统资源后再尝试大规模优化")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='温州全规模公交优化测试')
    parser.add_argument('--mode', choices=['quick', 'small', 'medium', 'full', 'progressive'], 
                       default='progressive', help='测试模式')
    parser.add_argument('--force', action='store_true', help='强制运行（忽略资源警告）')
    
    args = parser.parse_args()
    
    # 设置多进程方式
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 已经设置过了
    
    # 创建测试管理器
    test_manager = FullScaleTestManager()
    
    try:
        if args.mode == 'quick':
            result = test_manager.quick_test()
        elif args.mode == 'small':
            result = test_manager.small_scale_test()
        elif args.mode == 'medium':
            result = test_manager.medium_scale_test()
        elif args.mode == 'full':
            result = test_manager.run_full_scale_with_monitoring()
        elif args.mode == 'progressive':
            result = test_manager.run_progressive_test()
        else:
            logger.error(f"未知测试模式: {args.mode}")
            return
        
        logger.info(f"\n🎉 测试完成！模式: {args.mode}")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  用户中断测试")
    except Exception as e:
        logger.error(f"\n❌ 测试过程出错: {e}")


if __name__ == "__main__":
    main()