"""
传统TSP求解算法模块
包含：Greedy, Local Search (2-opt), VNS, Tabu Search
"""

import numpy as np
import random
import time
from typing import List, Tuple
from collections import deque


class TSPSolver:
    """TSP求解器基类"""

    def __init__(self, distance_matrix: np.ndarray, name: str = "Base"):
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
        self.name = name
        self.best_tour = None
        self.best_distance = float('inf')
        self.history = []

    def calculate_tour_length(self, tour: List[int]) -> float:
        """计算路径长度"""
        distance = 0
        for i in range(len(tour)):
            distance += self.distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
        return distance

    def generate_random_tour(self) -> List[int]:
        """生成随机路径"""
        tour = list(range(self.n_cities))
        random.shuffle(tour)
        return tour


class GreedySolver(TSPSolver):
    """贪心算法 - 最近邻算法"""

    def __init__(self, distance_matrix: np.ndarray):
        super().__init__(distance_matrix, "Greedy")

    def solve(self, start_city: int = 0) -> Tuple[List[int], float]:
        """最近邻贪心算法"""
        n = self.n_cities
        unvisited = set(range(n))
        tour = [start_city]
        unvisited.remove(start_city)

        current = start_city
        while unvisited:
            nearest = min(unvisited, key=lambda city: self.distance_matrix[current][city])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        distance = self.calculate_tour_length(tour)
        self.best_tour = tour
        self.best_distance = distance
        self.history = [distance]

        return tour, distance


class LocalSearchSolver(TSPSolver):
    """局部搜索 - 2-opt算法"""

    def __init__(self, distance_matrix: np.ndarray):
        super().__init__(distance_matrix, "2-opt Local Search")

    def two_opt_swap(self, tour: List[int], i: int, j: int) -> List[int]:
        """执行2-opt交换"""
        new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
        return new_tour

    def two_opt(self, tour: List[int], max_iterations: int = 1000) -> Tuple[List[int], float]:
        """2-opt改进算法"""
        best_tour = tour[:]
        best_distance = self.calculate_tour_length(best_tour)
        improved = True
        iteration = 0

        self.history = [best_distance]

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for i in range(1, self.n_cities - 1):
                for j in range(i + 1, self.n_cities):
                    new_tour = self.two_opt_swap(best_tour, i, j)
                    new_distance = self.calculate_tour_length(new_tour)

                    if new_distance < best_distance:
                        best_tour = new_tour
                        best_distance = new_distance
                        improved = True
                        self.history.append(best_distance)
                        break
                if improved:
                    break

        self.best_tour = best_tour
        self.best_distance = best_distance
        return best_tour, best_distance

    def solve(self, initial_tour: List[int] = None) -> Tuple[List[int], float]:
        """使用贪心算法生成初始解，然后用2-opt改进"""
        if initial_tour is None:
            greedy = GreedySolver(self.distance_matrix)
            initial_tour, _ = greedy.solve()

        return self.two_opt(initial_tour)


class VNSSolver(TSPSolver):
    """变邻域搜索算法"""

    def __init__(self, distance_matrix: np.ndarray):
        super().__init__(distance_matrix, "VNS")
        self.neighborhood_sizes = [2, 3, 4]

    def shake(self, tour: List[int], k: int) -> List[int]:
        """扰动操作：随机交换k对城市"""
        new_tour = tour[:]
        for _ in range(k):
            i, j = random.sample(range(len(tour)), 2)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour

    def local_search(self, tour: List[int]) -> Tuple[List[int], float]:
        """局部搜索（2-opt）- 充分收敛"""
        solver = LocalSearchSolver(self.distance_matrix)
        return solver.two_opt(tour, max_iterations=100)  # 恢复充分迭代

    def solve(self, max_iterations: int = 100, max_time: int = 60) -> Tuple[List[int], float]:
        """变邻域搜索算法"""
        greedy = GreedySolver(self.distance_matrix)
        current_tour, current_distance = greedy.solve()

        best_tour = current_tour[:]
        best_distance = current_distance

        self.history = [best_distance]
        start_time = time.time()

        for iteration in range(max_iterations):
            if time.time() - start_time > max_time:
                break

            k_index = 0
            while k_index < len(self.neighborhood_sizes):
                k = self.neighborhood_sizes[k_index]
                new_tour = self.shake(current_tour, k)
                new_tour, new_distance = self.local_search(new_tour)

                if new_distance < best_distance:
                    best_tour = new_tour
                    best_distance = new_distance
                    current_tour = new_tour
                    current_distance = new_distance
                    k_index = 0
                    self.history.append(best_distance)
                else:
                    k_index += 1

        self.best_tour = best_tour
        self.best_distance = best_distance
        return best_tour, best_distance


class TabuSearchSolver(TSPSolver):
    """禁忌搜索算法"""

    def __init__(self, distance_matrix: np.ndarray):
        super().__init__(distance_matrix, "Tabu Search")

    def get_neighbors(self, tour: List[int], n_neighbors: int = 20) -> List[List[int]]:
        """生成邻域解（使用2-opt）"""
        neighbors = []
        for _ in range(n_neighbors):
            i, j = sorted(random.sample(range(1, len(tour)), 2))
            neighbor = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
            neighbors.append(neighbor)
        return neighbors

    def solve(self, max_iterations: int = 100, tabu_tenure: int = 10) -> Tuple[List[int], float]:
        """禁忌搜索算法"""
        greedy = GreedySolver(self.distance_matrix)
        current_tour, current_distance = greedy.solve()

        best_tour = current_tour[:]
        best_distance = current_distance

        tabu_list = deque(maxlen=tabu_tenure)
        self.history = [best_distance]

        for iteration in range(max_iterations):
            neighbors = self.get_neighbors(current_tour)

            best_neighbor = None
            best_neighbor_distance = float('inf')

            for neighbor in neighbors:
                neighbor_tuple = tuple(neighbor)
                neighbor_distance = self.calculate_tour_length(neighbor)

                if neighbor_distance < best_distance:
                    best_neighbor = neighbor
                    best_neighbor_distance = neighbor_distance
                    break

                if neighbor_tuple not in tabu_list and neighbor_distance < best_neighbor_distance:
                    best_neighbor = neighbor
                    best_neighbor_distance = neighbor_distance

            if best_neighbor is not None:
                current_tour = best_neighbor
                current_distance = best_neighbor_distance
                tabu_list.append(tuple(current_tour))

                if current_distance < best_distance:
                    best_tour = current_tour[:]
                    best_distance = current_distance

                self.history.append(best_distance)

        self.best_tour = best_tour
        self.best_distance = best_distance
        return best_tour, best_distance
