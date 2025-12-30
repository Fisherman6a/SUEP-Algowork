"""
完整实验主程序 - 交互式版本
支持：
1. 运行新实验
2. 加载已有实验重新生成图表
"""

import os
import sys
from experiment_modules import ExperimentCoordinator


def show_menu():
    """显示交互式菜单"""
    print("\n" + "="*60)
    print("   TSP传统算法 vs AI-LHNS 对比实验系统")
    print("="*60)
    print("\n请选择操作：")
    print("  1 - 运行新实验（传统算法 + LHNS，仅存储数据）")
    print("  2 - 加载已有实验（从已有数据生成图表）")
    print("  0 - 退出")
    print()


def list_existing_experiments():
    """列出所有已有的实验目录"""
    experiments = []
    for item in os.listdir('.'):
        if os.path.isdir(item) and item.startswith('experiment_'):
            experiments.append(item)

    if not experiments:
        print("\n⚠️  未找到任何已有实验")
        return None

    experiments.sort(reverse=True)  # 最新的在前面

    print("\n找到以下实验：")
    for idx, exp in enumerate(experiments, 1):
        print(f"  {idx}. {exp}")

    return experiments


def run_new_experiment():
    """运行新实验（只跑数据，不生成图表）"""
    print("\n" + "="*60)
    print("【运行新实验】")
    print("="*60)

    # 询问用户要运行哪些LHNS算法
    print("\n请选择要运行的LHNS算法：")
    print("  1 - 只运行 VNS（快速测试，推荐）")
    print("  2 - 运行全部（VNS + ILS + TS）")

    while True:
        choice = input("\n请选择 (1/2): ").strip()
        if choice == '1':
            heuristic_types = ['vns']
            break
        elif choice == '2':
            heuristic_types = ['vns', 'ils', 'ts']
            break
        else:
            print("❌ 无效选项，请输入 1 或 2")

    project_root = os.path.dirname(os.path.abspath(__file__))
    coordinator = ExperimentCoordinator(project_root)

    # 只跑数据，不生成图表
    coordinator.run_baseline_algorithms()
    coordinator.run_lhns_algorithms(heuristic_types=heuristic_types)
    coordinator.save_and_summarize()  # 只保存数据

    print(f"\n✓ 数据收集完成！结果保存在: {coordinator.output_dir}/")
    print(f"提示: 选择操作2可以为此实验生成图表")


def load_existing_experiment():
    """加载已有实验"""
    experiments = list_existing_experiments()

    if not experiments:
        return

    while True:
        choice = input("\n请输入实验编号（或输入实验名称/时间戳，0返回）: ").strip()

        if choice == '0':
            return

        # 尝试作为编号解析
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(experiments):
                experiment_dir = experiments[idx]
                break
        # 尝试作为实验名称解析
        elif choice.startswith('experiment_'):
            if choice in experiments:
                experiment_dir = choice
                break
            elif os.path.isdir(choice):
                experiment_dir = choice
                break
        # 尝试作为时间戳解析 (20251231_032035 -> experiment_20251231_032035)
        elif '_' in choice and len(choice) == 15:  # 格式: YYYYMMDD_HHMMSS
            experiment_dir = f"experiment_{choice}"
            if experiment_dir in experiments or os.path.isdir(experiment_dir):
                break

        print("❌ 无效输入，请重新输入")

    print(f"\n正在加载实验: {experiment_dir}")

    project_root = os.path.dirname(os.path.abspath(__file__))
    coordinator = ExperimentCoordinator(project_root, experiment_dir=experiment_dir)

    # 加载数据
    if coordinator.load_existing_results():
        # 重新生成图表
        coordinator.visualize_results()
        print(f"\n✓ 图表重新生成完成！保存在: {experiment_dir}/figures/")
    else:
        print("\n❌ 加载实验失败")


def main():
    """主函数"""
    while True:
        show_menu()

        choice = input("请输入选项 (0/1/2): ").strip()

        if choice == '1':
            run_new_experiment()
        elif choice == '2':
            load_existing_experiment()
        elif choice == '0':
            print("\n再见！\n")
            break
        else:
            print("\n❌ 无效选项，请重新输入")


if __name__ == "__main__":
    main()
