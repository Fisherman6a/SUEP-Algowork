"""
LHNS运行器模块 - 改进版
直接调用Python模块，无需修改文件
"""

import os
import sys
import time
import re
from typing import Dict, Optional


class LHNSRunner:
    """运行项目的LHNS实现（直接导入Python模块）"""

    def __init__(self, project_root: str, experiment_dir: str = None):
        self.project_root = project_root
        self.lhns_dir = os.path.join(project_root, "src", "lhns")
        self.experiment_dir = experiment_dir  # 统一的实验输出目录

    def run_lhns_experiment(self, heuristic_type: str = "vns", iterations: int = 20) -> Optional[Dict]:
        """
        运行LHNS实验（直接修改paras对象，无需修改文件）

        参数:
            heuristic_type: 启发式类型 ['vns', 'ils', 'ts']
            iterations: 迭代次数
        """
        print(f"\n{'='*60}")
        print(f"运行LHNS实验: tsp_construct - {heuristic_type}")
        print(f"{'='*60}")

        try:
            # 保存当前工作目录
            original_cwd = os.getcwd()
            original_path = sys.path.copy()

            # 切换到LHNS目录并设置路径（必须在导入之前）
            os.chdir(self.lhns_dir)
            sys.path.insert(0, self.lhns_dir)
            sys.path.insert(0, os.path.join(self.lhns_dir, ".."))
            sys.path.insert(0, os.path.join(self.lhns_dir, "..", ".."))

            # 直接导入（路径已设置，支持相对导入）
            from src.lhns import lhns
            from src.lhns.utils.getParas import Paras
            from log_utils import create_logger, copy_all_src, get_result_folder, set_result_folder_ini
            from logging import getLogger

            print(f"  配置参数: {heuristic_type}, iterations={iterations}")

            # 创建参数对象并设置
            paras = Paras()
            paras.set_paras(
                method="lhns",
                ini_ratio=0.5, 
                heuristic_type=heuristic_type,  # 动态设置
                problem="tsp_construct",
                problem_type="white-box",
                llm_api_endpoint="api.siliconflow.cn",
                llm_api_key="",
                llm_model="deepseek-ai/DeepSeek-V3.2",
                iterations=iterations,  # 动态设置
                rounds=1,
                exp_debug_mode=False
            )

            # 设置日志
            logger = getLogger('root')
            logger_params = {
                'log_file': {
                    'desc': f"{paras.problem}_{paras.problem_type}_{paras.heuristic_type}",
                    'filename': 'run_log'
                }
            }
            create_logger(**logger_params)

            # 打印配置
            logger.info(f"problem: {paras.problem}")
            logger.info(f"problem_type: {paras.problem_type}")
            logger.info(f"heuristic_type: {paras.heuristic_type}")
            logger.info(f"ini_ratio: {paras.ini_ratio}")

            copy_all_src(get_result_folder())

            # 运行LHNS
            print(f"  开始运行LHNS...")
            start_time = time.time()

            # 正确的调用方式：创建EVOL对象并运行
            import numpy as np

            set_result_folder_ini()

            # 设置exp_seed_path
            if paras.problem_type == 'white-box':
                paras.exp_seed_path = f"problems/optimization/{paras.problem}/ini_state.json"
            else:
                paras.exp_seed_path = f"problems/optimization/{paras.problem}/ini_state_black.json"

            # 创建进化对象
            evolution = lhns.EVOL(paras)

            # 运行实验
            history_mean = []
            history_best = []
            for i in range(paras.rounds):
                m, b, test_results, times = evolution.run(i, logger_params['log_file'].get('filepath', 'run_log.log'))
                history_mean.append(m)
                history_best.append(b)

            # 记录结果
            logger.info('=================================================================')
            logger.info('Test Done !')
            logger.info(f'Average Obj: {np.mean(history_mean)}')
            logger.info(f'Best Obj List: {[e for e in history_best]}')
            logger.info(f'Average Best Obj: {np.mean(history_best)}')

            end_time = time.time()
            print(f"  LHNS运行完成，耗时: {end_time - start_time:.2f}s")

            # 解析结果 - 在重命名之前先获取并解析log
            logs_dir = os.path.join(self.lhns_dir, "logs")
            latest_log = self.get_latest_log_dir(logs_dir)

            if latest_log:
                # ⚠️ 重要：先解析结果，再重命名！
                result_data = self.parse_log_results(latest_log)
                result_data['method'] = f'LHNS-{heuristic_type.upper()}'
                result_data['time'] = end_time - start_time
                result_data['iterations'] = iterations
                result_data['log_dir'] = latest_log  # 保存原始log目录路径

                print(f"\n✓ LHNS完成 - 最优目标: {result_data.get('best_obj', 0):.4f}, 时间: {result_data['time']:.2f}s")

                # 注意：不在这里重命名！留到所有实验完成后统一重命名
                # 这样可以避免重命名后时间戳混乱导致后续实验拿到错误的log

                # 恢复工作目录和路径
                os.chdir(original_cwd)
                sys.path = original_path

                return result_data
            else:
                print(f"✗ 未找到log输出")
                os.chdir(original_cwd)
                sys.path = original_path
                return None

        except Exception as e:
            print(f"✗ LHNS运行出错: {str(e)}")
            import traceback
            traceback.print_exc()

            # 恢复工作目录和路径
            os.chdir(original_cwd)
            sys.path = original_path
            return None

    def get_latest_log_dir(self, logs_dir: str) -> Optional[str]:
        """获取最新的log目录"""
        if not os.path.exists(logs_dir):
            return None

        log_dirs = [d for d in os.listdir(logs_dir)
                   if os.path.isdir(os.path.join(logs_dir, d))]

        if not log_dirs:
            return None

        # 按时间戳排序，取最新的
        log_dirs.sort(reverse=True)
        return os.path.join(logs_dir, log_dirs[0])

    def parse_log_results(self, log_dir: str) -> Dict:
        """从log目录解析结果"""
        result = {
            'best_obj': 0,
            'mean_obj': 0,
            'test_results': []
        }

        # 查找log文件（注意：文件名是run_log，没有.log后缀）
        log_file = os.path.join(log_dir, "run_log")
        if not os.path.exists(log_file):
            # 也尝试.log后缀（兼容性）
            log_file = os.path.join(log_dir, "run_log.log")

        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()

                # 提取Best Obj
                best_match = re.search(r'Average Best Obj:\s*([\d.]+)', content)
                if best_match:
                    result['best_obj'] = float(best_match.group(1))

                mean_match = re.search(r'Average Obj:\s*([\d.]+)', content)
                if mean_match:
                    result['mean_obj'] = float(mean_match.group(1))
        else:
            print(f"  ⚠️ 警告：未找到log文件：{log_file}")

        return result

    def rename_all_latest_logs(self, experiment_timestamp: str):
        """
        在实验完成后，批量重命名所有最新的LHNS日志文件夹

        参数:
            experiment_timestamp: 实验时间戳，格式为 "20251231_032035"

        功能:
            找到 src/lhns/logs/ 下日期时间最大的所有日志文件夹，
            将它们的时间戳统一替换为 experiment_timestamp
        """
        import shutil

        logs_dir = os.path.join(self.lhns_dir, "logs")
        if not os.path.exists(logs_dir):
            print(f"⚠️ logs目录不存在: {logs_dir}")
            return

        # 获取所有log目录
        all_logs = [d for d in os.listdir(logs_dir)
                   if os.path.isdir(os.path.join(logs_dir, d))]

        if not all_logs:
            print(f"⚠️ logs目录为空: {logs_dir}")
            return

        # 按时间戳排序，找到最大的时间戳
        all_logs.sort(reverse=True)
        latest_timestamp = all_logs[0].split('_')[0] + '_' + all_logs[0].split('_')[1]  # 提取 20251231_032159

        print(f"\n📝 开始重命名日志文件...")
        print(f"   实验时间戳: {experiment_timestamp}")
        print(f"   最新日志时间戳: {latest_timestamp}")

        # 找到所有具有最新时间戳的log目录
        latest_logs = [d for d in all_logs if d.startswith(latest_timestamp)]

        if not latest_logs:
            print(f"⚠️ 未找到时间戳为 {latest_timestamp} 的日志文件")
            return

        print(f"   找到 {len(latest_logs)} 个需要重命名的日志文件夹:")

        renamed_count = 0
        for log_name in latest_logs:
            old_path = os.path.join(logs_dir, log_name)

            # 替换时间戳: 20251231_032159_xxx -> 20251231_032035_xxx
            new_log_name = re.sub(r'^\d{8}_\d{6}', experiment_timestamp, log_name)
            new_path = os.path.join(logs_dir, new_log_name)

            # 如果新路径已存在，先删除
            if os.path.exists(new_path):
                print(f"   ⚠️ 删除已存在的目录: {new_log_name}")
                shutil.rmtree(new_path)

            # 重命名
            try:
                os.rename(old_path, new_path)
                print(f"   ✓ {log_name} → {new_log_name}")
                renamed_count += 1
            except Exception as e:
                print(f"   ✗ 重命名失败 {log_name}: {str(e)}")

        print(f"\n✓ 日志重命名完成！共重命名 {renamed_count} 个文件夹")
        print(f"  现在可以使用时间戳 '{experiment_timestamp}' 统一匹配：")
        print(f"  - experiment_{experiment_timestamp}/")
        print(f"  - src/lhns/logs/{experiment_timestamp}_*/")
