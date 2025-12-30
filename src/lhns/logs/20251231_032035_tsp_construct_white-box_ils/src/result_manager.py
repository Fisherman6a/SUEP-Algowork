"""
实验结果管理模块
负责保存和生成结果摘要
"""

import os
import json
import numpy as np
from typing import Dict


class ResultManager:
    """实验结果管理器"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.data_dir = os.path.join(output_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

    def save_results(self, baseline_results: Dict, lhns_results: Dict):
        """保存实验结果到JSON文件"""
        print("\n保存实验数据...")

        # 保存baseline结果
        baseline_data = {}
        for instance, data in baseline_results.items():
            baseline_data[instance] = {}
            for algo, result in data['algorithms'].items():
                baseline_data[instance][algo] = {
                    'avg_distance': float(result['avg_distance']),
                    'std_distance': float(result['std_distance']),
                    'avg_time': float(result['avg_time']),
                    'all_distances': [float(d) for d in result['distances']],
                    'all_times': [float(t) for t in result['times']]
                }

        with open(os.path.join(self.data_dir, 'baseline_results.json'), 'w', encoding='utf-8') as f:
            json.dump(baseline_data, f, indent=2, ensure_ascii=False)

        # 保存LHNS结果
        lhns_data = {}
        for method, result in lhns_results.items():
            lhns_data[method] = {
                'best_obj': result.get('best_obj', 0),
                'mean_obj': result.get('mean_obj', 0),
                'time': result.get('time', 0)
            }

        with open(os.path.join(self.data_dir, 'lhns_results.json'), 'w', encoding='utf-8') as f:
            json.dump(lhns_data, f, indent=2, ensure_ascii=False)

        print(f"✓ 结果已保存到 {self.data_dir}/")

    def generate_summary(self, baseline_results: Dict, lhns_results: Dict):
        """生成文本摘要"""
        print("\n" + "="*60)
        print("实验结果摘要")
        print("="*60)

        print("\n【传统算法结果（8个TSP50实例平均）】")
        for instance_name, data in baseline_results.items():
            print(f"\n{instance_name}:")
            best = min(data['algorithms'].items(), key=lambda x: x[1]['avg_distance'])
            fastest = min(data['algorithms'].items(), key=lambda x: x[1]['avg_time'])

            print(f"  最优解: {best[0]} (平均距离: {best[1]['avg_distance']:.4f}±{best[1]['std_distance']:.4f})")
            print(f"  最快: {fastest[0]} (平均时间: {fastest[1]['avg_time']:.4f}s)")

            for algo, result in sorted(data['algorithms'].items(), key=lambda x: x[1]['avg_distance']):
                print(f"    {algo:20s}: 平均距离={result['avg_distance']:8.4f}±{result['std_distance']:.4f}, "
                      f"平均时间={result['avg_time']:.4f}s")

        if lhns_results:
            print("\n【LHNS算法结果】")
            for method, result in lhns_results.items():
                print(f"  {method}: 最优目标={result.get('best_obj', 0):.4f}, "
                      f"平均目标={result.get('mean_obj', 0):.4f}, 时间={result.get('time', 0):.2f}s")

        # 保存摘要到文件
        summary_file = os.path.join(self.output_dir, 'SUMMARY.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("实验结果摘要\n")
            f.write("="*60 + "\n\n")

            f.write("【传统算法结果（8个TSP50实例平均）】\n")
            for instance_name, data in baseline_results.items():
                f.write(f"\n{instance_name}:\n")
                for algo, result in sorted(data['algorithms'].items(), key=lambda x: x[1]['avg_distance']):
                    f.write(f"  {algo:20s}: 平均距离={result['avg_distance']:8.4f}±{result['std_distance']:.4f}, "
                           f"平均时间={result['avg_time']:.4f}s\n")

            if lhns_results:
                f.write("\n【LHNS算法结果】\n")
                for method, result in lhns_results.items():
                    f.write(f"  {method}: 最优目标={result.get('best_obj', 0):.4f}, "
                           f"平均目标={result.get('mean_obj', 0):.4f}, 时间={result.get('time', 0):.2f}s\n")

        print(f"\n✓ 摘要已保存到 {summary_file}")
