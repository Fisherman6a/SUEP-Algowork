"""
实验协调器模块
协调整个实验流程
"""

import os
import pickle
import numpy as np
import time
from datetime import datetime
from typing import List, Dict

from .traditional_solvers import GreedySolver, LocalSearchSolver, VNSSolver, TabuSearchSolver
from .lhns_runner import LHNSRunner
from .visualizer import ExperimentVisualizer
from .result_manager import ResultManager


class ExperimentCoordinator:
    """实验协调器：管理整个实验流程"""

    def __init__(self, project_root: str, experiment_dir: str = None):
        self.project_root = project_root

        # 如果指定了experiment_dir，加载已有实验；否则创建新实验
        if experiment_dir:
            self.output_dir = experiment_dir
            print(f"\n加载已有实验: {experiment_dir}")
        else:
            # 生成带时间戳的输出目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"experiment_{timestamp}"
            print(f"\n创建新实验: {self.output_dir}")

        # 创建目录结构
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "data"), exist_ok=True)

        # 实验结果存储
        self.baseline_results = {}
        self.lhns_results = {}

        # 初始化模块
        self.visualizer = ExperimentVisualizer(self.output_dir)
        self.result_manager = ResultManager(self.output_dir)

        # 加载LHNS的instances.pkl数据集（统一数据源）
        self._load_dataset()

        print(f"\n实验结果将保存到: {self.output_dir}/")

    def _load_dataset(self):
        """加载LHNS数据集"""
        instances_path = os.path.join(
            self.project_root, "src", "lhns", "problems",
            "optimization", "tsp_construct", "instances.pkl"
        )
        print(f"\n加载LHNS数据集: {instances_path}")
        with open(instances_path, 'rb') as f:
            self.lhns_instances = pickle.load(f)[:8]  # 使用前8个实例，与LHNS一致
        print(f"✓ 已加载 {len(self.lhns_instances)} 个TSP50实例（与LHNS完全相同）")

    def run_baseline_algorithms(self):
        """运行传统baseline算法（使用LHNS相同的数据集）"""
        print("\n" + "="*60)
        print("【第一部分】运行传统算法（Baseline）")
        print("="*60)
        print("\n✓ 使用与LHNS完全相同的TSP50数据集（8个实例）")

        algorithms = [
            (GreedySolver, {}),
            (LocalSearchSolver, {}),
            (VNSSolver, {'max_iterations': 50, 'max_time': 10}),  # 增加迭代和时间，确保收敛
            (TabuSearchSolver, {'max_iterations': 100, 'tabu_tenure': 10})  # 增加迭代次数
        ]

        # 对每个实例运行传统算法
        print(f"\n【TSP50】使用LHNS数据集的8个实例...")

        # 存储所有实例的平均结果
        instance_name = "TSP50"
        self.baseline_results[instance_name] = {
            'coords_list': [],
            'distance_matrices': [],
            'algorithms': {
                algo_class(np.zeros((50, 50))).name: {
                    'distances': [],
                    'times': [],
                    'tours': []
                } for algo_class, _ in algorithms
            }
        }

        # 遍历8个实例
        for idx, (coords, distance_matrix) in enumerate(self.lhns_instances):
            print(f"\n  实例 {idx+1}/8:")
            self.baseline_results[instance_name]['coords_list'].append(coords)
            self.baseline_results[instance_name]['distance_matrices'].append(distance_matrix)

            for solver_class, params in algorithms:
                solver = solver_class(distance_matrix)
                solver_name = solver.name

                print(f"    运行 {solver_name}...", end=" ")
                start_time = time.time()
                tour, distance = solver.solve(**params)
                end_time = time.time()

                self.baseline_results[instance_name]['algorithms'][solver_name]['distances'].append(distance)
                self.baseline_results[instance_name]['algorithms'][solver_name]['times'].append(end_time - start_time)
                self.baseline_results[instance_name]['algorithms'][solver_name]['tours'].append(tour)

                print(f"✓ 距离: {distance:.4f}, 时间: {end_time - start_time:.4f}s")

        # 计算每个算法的平均性能
        print(f"\n【平均性能统计】")
        for algo_name, results in self.baseline_results[instance_name]['algorithms'].items():
            avg_distance = np.mean(results['distances'])
            avg_time = np.mean(results['times'])
            std_distance = np.std(results['distances'])

            # 保存统计信息
            results['avg_distance'] = avg_distance
            results['avg_time'] = avg_time
            results['std_distance'] = std_distance
            results['history'] = [avg_distance]  # 为兼容性保留

            print(f"  {algo_name:20s}: 平均距离={avg_distance:.4f}±{std_distance:.4f}, 平均时间={avg_time:.4f}s")

    def run_lhns_algorithms(self):
        """运行LHNS算法"""
        print("\n" + "="*60)
        print("【第二部分】运行LHNS算法（调用项目代码）")
        print("="*60)

        runner = LHNSRunner(self.project_root, experiment_dir=self.output_dir)  # 传递experiment_dir

        # 运行三种LHNS变体
        for heuristic_type in ['vns', 'ils', 'ts']:
            result = runner.run_lhns_experiment(
                heuristic_type=heuristic_type,
                iterations=20
            )

            if result:
                self.lhns_results[f'LHNS-{heuristic_type.upper()}'] = result

    def visualize_results(self):
        """生成可视化图表"""
        print("\n" + "="*60)
        print("【第三部分】生成可视化图表")
        print("="*60)

        # 1. 绘制传统算法性能对比
        print("\n1. 绘制传统算法性能对比...")
        self.visualizer.plot_performance_comparison(self.baseline_results)

        # 2. 绘制传统算法 vs LHNS 对比图表
        if self.lhns_results:
            print("\n2. 绘制传统算法 vs LHNS 对比图...")
            self.visualizer.plot_quality_comparison(self.baseline_results, self.lhns_results)
            self.visualizer.plot_time_comparison(self.baseline_results, self.lhns_results)
            self.visualizer.plot_radar_chart(self.baseline_results, self.lhns_results)
            self.visualizer.plot_performance_heatmap(self.baseline_results, self.lhns_results)
        else:
            print("\n⚠️ 没有LHNS结果，跳过对比图")

        print("\n✓ 所有图表生成完成！")

    def save_and_summarize(self):
        """保存结果并生成摘要"""
        self.result_manager.save_results(self.baseline_results, self.lhns_results)
        self.result_manager.generate_summary(self.baseline_results, self.lhns_results)

        print("\n" + "="*60)
        print("实验完成！")
        print(f"结果保存在 {self.output_dir}/ 目录")
        print("="*60 + "\n")

    def load_existing_results(self):
        """加载已有实验的结果（用于重新生成图表）"""
        print("\n" + "="*60)
        print("【加载已有实验数据】")
        print("="*60)

        import json

        # 1. 加载传统算法结果
        baseline_file = os.path.join(self.output_dir, "data", "baseline_results.json")
        if os.path.exists(baseline_file):
            with open(baseline_file, 'r', encoding='utf-8') as f:
                self.baseline_results = json.load(f)
            print(f"\n✓ 已加载传统算法结果: {len(self.baseline_results)} 个问题")
        else:
            print(f"\n✗ 未找到传统算法结果文件: {baseline_file}")
            return False

        # 2. 加载或解析LHNS结果
        lhns_logs_dir = os.path.join(self.output_dir, "lhns_logs")
        if os.path.exists(lhns_logs_dir):
            print(f"\n✓ 找到LHNS logs目录，解析结果...")
            runner = LHNSRunner(self.project_root, experiment_dir=self.output_dir)

            for heuristic_type in ['vns', 'ils', 'ts']:
                log_dir = os.path.join(lhns_logs_dir, f"lhns_{heuristic_type}")
                if os.path.exists(log_dir):
                    result_data = runner.parse_log_results(log_dir)
                    if result_data and result_data.get('best_obj', 0) > 0:
                        result_data['method'] = f'LHNS-{heuristic_type.upper()}'
                        self.lhns_results[f'LHNS-{heuristic_type.upper()}'] = result_data
                        print(f"  → LHNS-{heuristic_type.upper()}: {result_data.get('best_obj', 0):.4f}")
                    else:
                        print(f"  ⚠️ LHNS-{heuristic_type.upper()}: 解析失败或结果为0")
                else:
                    print(f"  ⚠️ 未找到LHNS-{heuristic_type.upper()} log")
        else:
            print(f"\n⚠️ 未找到LHNS logs目录: {lhns_logs_dir}")

        print(f"\n✓ 加载完成 - 传统算法: {len(self.baseline_results)}, LHNS: {len(self.lhns_results)}")
        return True
