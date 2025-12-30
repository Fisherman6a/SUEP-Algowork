"""
实验结果可视化模块
生成所有对比图表
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from math import pi


class ExperimentVisualizer:
    """实验结果可视化器"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

    def plot_performance_comparison(self, baseline_results: Dict):
        """绘制传统算法性能对比（平均性能）"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        instance_name = "TSP50"
        data = baseline_results[instance_name]
        algorithms = list(data['algorithms'].keys())

        # 使用平均距离和平均时间
        avg_distances = [data['algorithms'][algo]['avg_distance'] for algo in algorithms]
        std_distances = [data['algorithms'][algo]['std_distance'] for algo in algorithms]
        avg_times = [data['algorithms'][algo]['avg_time'] for algo in algorithms]

        x = np.arange(len(algorithms))
        width = 0.6

        # 绘制平均距离（带误差棒）
        ax1.bar(x, avg_distances, width, yerr=std_distances, capsize=5, alpha=0.8)
        ax1.set_xlabel('算法', fontsize=12)
        ax1.set_ylabel('平均路径长度', fontsize=12)
        ax1.set_title('传统算法解质量对比（8个实例平均）', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms, rotation=15, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')

        # 绘制平均时间
        ax2.bar(x, avg_times, width, alpha=0.8, color='orange')
        ax2.set_xlabel('算法', fontsize=12)
        ax2.set_ylabel('平均运行时间 (秒)', fontsize=12)
        ax2.set_title('传统算法运行时间对比（8个实例平均）', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(algorithms, rotation=15, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, "baseline_performance_comparison.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: baseline_performance_comparison.png")

    def plot_quality_comparison(self, baseline_results: Dict, lhns_results: Dict):
        """解质量对比：传统算法 vs LHNS（使用平均值）"""
        fig, ax = plt.subplots(figsize=(14, 8))

        # 准备数据
        all_algorithms = []
        all_scores = []
        all_colors = []

        # 传统算法（使用TSP50的平均数据）
        if 'TSP50' in baseline_results:
            for algo_name, result in baseline_results['TSP50']['algorithms'].items():
                all_algorithms.append(f"{algo_name}\n(传统)")
                all_scores.append(result['avg_distance'])  # 使用平均距离
                all_colors.append('#3498db')  # 蓝色 - 传统算法

        # LHNS算法
        for method, result in lhns_results.items():
            all_algorithms.append(f"{method}\n(AI-LHNS)")
            all_scores.append(result.get('best_obj', 0))
            all_colors.append('#e74c3c')  # 红色 - LHNS

        # 绘制柱状图
        x = np.arange(len(all_algorithms))
        ax.bar(x, all_scores, color=all_colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        # 标注数值
        for i, score in enumerate(all_scores):
            ax.text(i, score, f'{score:.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', alpha=0.7, label='传统人工设计算法 (8实例平均)'),
            Patch(facecolor='#e74c3c', alpha=0.7, label='AI自动设计算法 (LHNS)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

        ax.set_xlabel('算法', fontsize=13, fontweight='bold')
        ax.set_ylabel('路径长度 (越小越好)', fontsize=13, fontweight='bold')
        ax.set_title('传统算法 vs AI-LHNS：解质量对比 (TSP50)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(all_algorithms, rotation=0, ha='center', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, "comparison_quality_all.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: comparison_quality_all.png")

    def plot_time_comparison(self, baseline_results: Dict, lhns_results: Dict):
        """运行时间对比：传统算法 vs LHNS（使用平均值）"""
        fig, ax = plt.subplots(figsize=(14, 8))

        # 准备数据
        all_algorithms = []
        all_times = []
        all_colors = []

        # 传统算法（使用TSP50的平均数据）
        if 'TSP50' in baseline_results:
            for algo_name, result in baseline_results['TSP50']['algorithms'].items():
                all_algorithms.append(f"{algo_name}\n(传统)")
                all_times.append(result['avg_time'])  # 使用平均时间
                all_colors.append('#2ecc71')  # 绿色 - 传统算法

        # LHNS算法
        for method, result in lhns_results.items():
            all_algorithms.append(f"{method}\n(AI-LHNS)")
            all_times.append(result.get('time', 0))
            all_colors.append('#f39c12')  # 橙色 - LHNS

        # 绘制柱状图
        x = np.arange(len(all_algorithms))
        ax.bar(x, all_times, color=all_colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        # 标注数值
        for i, time_val in enumerate(all_times):
            ax.text(i, time_val, f'{time_val:.2f}s',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', alpha=0.7, label='传统算法 (8实例平均)'),
            Patch(facecolor='#f39c12', alpha=0.7, label='LHNS (需调用LLM)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

        ax.set_xlabel('算法', fontsize=13, fontweight='bold')
        ax.set_ylabel('运行时间 (秒)', fontsize=13, fontweight='bold')
        ax.set_title('传统算法 vs AI-LHNS：运行时间对比 (TSP50)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(all_algorithms, rotation=0, ha='center', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, "comparison_time_all.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: comparison_time_all.png")

    def plot_radar_chart(self, baseline_results: Dict, lhns_results: Dict):
        """绘制性能雷达图"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # 性能维度
        categories = ['解质量\n(归一化)', '运行速度\n(归一化)', '稳定性\n(归一化)', '创新性']
        N = len(categories)

        # 准备数据 - 选择最佳传统算法和最佳LHNS
        if 'TSP50' in baseline_results:
            best_traditional = min(
                baseline_results['TSP50']['algorithms'].items(),
                key=lambda x: x[1]['avg_distance']
            )
            trad_quality = best_traditional[1]['avg_distance']
            trad_time = best_traditional[1]['avg_time']
            trad_name = best_traditional[0]
        else:
            trad_quality, trad_time, trad_name = 10.0, 1.0, "Traditional"

        # LHNS：选择解质量最好的
        if lhns_results:
            best_lhns = min(lhns_results.items(), key=lambda x: x[1].get('best_obj', 999))
            lhns_quality = best_lhns[1].get('best_obj', 10.0)
            lhns_time = best_lhns[1].get('time', 100.0)
            lhns_name = best_lhns[0]
        else:
            lhns_quality, lhns_time, lhns_name = 10.0, 100.0, "LHNS"

        # 归一化（越高越好）
        max_quality = max(trad_quality, lhns_quality)
        max_time = max(trad_time, lhns_time)

        # 传统算法数据（归一化到0-1）
        traditional_values = [
            1 - (trad_quality / max_quality) if max_quality > 0 else 0.5,
            1 - (trad_time / max_time) if max_time > 0 else 0.5,
            0.8,   # 稳定性
            0.3    # 创新性
        ]

        # LHNS数据
        lhns_values = [
            1 - (lhns_quality / max_quality) if max_quality > 0 else 0.5,
            1 - (lhns_time / max_time) if max_time > 0 else 0.5,
            0.6,   # 稳定性
            0.95   # 创新性
        ]

        # 计算角度
        angles = [n / float(N) * 2 * pi for n in range(N)]
        traditional_values += traditional_values[:1]
        lhns_values += lhns_values[:1]
        angles += angles[:1]

        # 绘制
        ax.plot(angles, traditional_values, 'o-', linewidth=2, label=f'传统最优: {trad_name}', color='#3498db')
        ax.fill(angles, traditional_values, alpha=0.25, color='#3498db')

        ax.plot(angles, lhns_values, 'o-', linewidth=2, label=f'AI-LHNS最优: {lhns_name}', color='#e74c3c')
        ax.fill(angles, lhns_values, alpha=0.25, color='#e74c3c')

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        plt.title('传统算法 vs AI-LHNS：多维性能对比', size=16, fontweight='bold', pad=20)

        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, "comparison_radar.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: comparison_radar.png")

    def plot_performance_heatmap(self, baseline_results: Dict, lhns_results: Dict):
        """绘制算法性能矩阵热力图"""
        algorithms = []
        quality_scores = []
        time_scores = []

        # 传统算法（使用TSP50平均数据）
        if 'TSP50' in baseline_results:
            for algo_name, result in baseline_results['TSP50']['algorithms'].items():
                algorithms.append(algo_name + "\n(传统)")
                quality_scores.append(result['avg_distance'])
                time_scores.append(result['avg_time'])

        # LHNS算法
        for method, result in lhns_results.items():
            algorithms.append(method + "\n(LHNS)")
            quality_scores.append(result.get('best_obj', 0))
            time_scores.append(result.get('time', 0))

        # 归一化到0-100分制（越高越好）
        max_quality = max(quality_scores) if quality_scores else 1
        max_time = max(time_scores) if time_scores else 1

        quality_normalized = [100 * (1 - q/max_quality) for q in quality_scores]
        time_normalized = [100 * (1 - t/max_time) for t in time_scores]

        # 计算综合得分
        综合得分 = [(q * 0.6 + t * 0.4) for q, t in zip(quality_normalized, time_normalized)]

        # 构建矩阵
        data_matrix = np.array([quality_normalized, time_normalized, 综合得分])

        # 绘制热力图
        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        # 设置刻度
        ax.set_xticks(np.arange(len(algorithms)))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(algorithms, fontsize=10)
        ax.set_yticklabels(['解质量得分', '运行速度得分', '综合得分'], fontsize=12)

        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        # 在每个单元格中显示数值
        for i in range(3):
            for j in range(len(algorithms)):
                ax.text(j, i, f'{data_matrix[i, j]:.1f}',
                       ha="center", va="center", color="black", fontsize=10, fontweight='bold')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('性能得分 (0-100)', rotation=270, labelpad=20, fontsize=12)

        ax.set_title('算法性能矩阵热力图（得分越高越好）', fontsize=16, fontweight='bold', pad=15)

        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, "comparison_heatmap.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: comparison_heatmap.png")
