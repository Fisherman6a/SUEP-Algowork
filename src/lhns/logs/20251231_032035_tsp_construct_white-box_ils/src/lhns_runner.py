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
                llm_api_key="sk-ahfzvfjnwvlwxlwiygoxllgddpcanwpvullyfgwtlcuskduk",
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

            # 解析结果
            logs_dir = os.path.join(self.lhns_dir, "logs")
            latest_log = self.get_latest_log_dir(logs_dir)

            if latest_log:
                result_data = self.parse_log_results(latest_log)
                result_data['method'] = f'LHNS-{heuristic_type.upper()}'
                result_data['time'] = end_time - start_time
                result_data['iterations'] = iterations
                result_data['log_dir'] = latest_log  # 保存log目录路径

                print(f"\n✓ LHNS完成 - 最优目标: {result_data.get('best_obj', 0):.4f}, 时间: {result_data['time']:.2f}s")

                # 如果指定了experiment_dir，复制log到统一位置
                if self.experiment_dir:
                    self._copy_log_to_experiment(latest_log, heuristic_type)

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

    def _copy_log_to_experiment(self, log_dir: str, heuristic_type: str):
        """将LHNS log复制到experiment目录"""
        import shutil

        # 创建lhns_logs子目录
        lhns_logs_dir = os.path.join(self.experiment_dir, "lhns_logs")
        os.makedirs(lhns_logs_dir, exist_ok=True)

        # 目标目录名
        dest_dir = os.path.join(lhns_logs_dir, f"lhns_{heuristic_type}")

        # 如果已存在，删除
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)

        # 复制整个log目录
        shutil.copytree(log_dir, dest_dir)
        print(f"  → LHNS log已复制到: {dest_dir}")
