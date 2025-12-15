# Write the implemented EA into the provided template file.

import Reporter
import numpy as np
import random
from typing import List, Tuple

# Modify the class name to match your student number.
class r0123456:

    def __init__(self,
                 N_RUNS: int = 500,
                 lam: int = 100,           # seed population size λ
                 mu: int = 100,            # offspring size μ
                 k_tournament: int = 5,    # tournament size k
                 mutation_rate: float = 0.2,
                 use_cscx = True,
                 rng_seed: int | None = None):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.N_RUNS = int(N_RUNS)
        self.lam = int(lam)
        self.mu = int(mu)
        self.k_tournament = int(k_tournament)
        self.mutation_rate = float(mutation_rate)
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)
        self.use_cscx = use_cscx

    # ---- Core EA ----
    def optimize(self, filename: str):
        # Read distance matrix from file.
        with open(filename) as file:
            distanceMatrix = np.loadtxt(file, delimiter=",")

        n = distanceMatrix.shape[0]
        D = distanceMatrix.astype(np.float64, copy=False)

        # Initialize λ random tours (cycles).
        population = [self._random_permutation(n) for _ in range(self.lam)]
        fitness = np.array([self._tour_length(t, D) for t in population], dtype=np.float64)

        # Live header
        print("generation,best_cost", flush=True)

        # Main loop
        for gen in range(1, self.N_RUNS + 1):
            # Reproduction: create μ offspring via k-tournament selection + CSCX + swap mutation
            offspring: List[np.ndarray] = []
            while len(offspring) < self.mu:
                p1 = self._k_tournament(population, fitness, self.k_tournament)
                p2 = self._k_tournament(population, fitness, self.k_tournament)
                if self.use_cscx:
                    c1 = self._cscx(p1, p2)
                    c2 = self._cscx(p2, p1)
                else:
                    c1 = self._erx(p1, p2)
                    c2 = self._erx(p2, p1)
                if self.rng.random() < self.mutation_rate:
                    self._light_two_opt(c1, D)
                if self.rng.random() < self.mutation_rate:
                    self._light_two_opt(c2, D)
                offspring.append(c1)
                if len(offspring) < self.mu:
                    offspring.append(c2)

            offspring_f = np.array([self._tour_length(t, D) for t in offspring], dtype=np.float64)

            # (λ+μ)-elimination: keep best λ from union
            combined = population + offspring
            combined_f = np.concatenate([fitness, offspring_f])
            keep_idx = np.argsort(combined_f)[:self.lam]
            population = [combined[i] for i in keep_idx]
            fitness = combined_f[keep_idx]

            # Reporting
            meanObjective = float(np.mean(fitness))
            bestObjective = float(fitness.min())
            bestSolution = population[int(np.argmin(fitness))].copy()
            bestSolution = self._rotate_to_start(bestSolution, 0)

            # Live print of generation and best cost
            print(f"{gen},{bestObjective}", flush=True)

            # Course reporter
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

        return 0


    # ---- Helpers ----
    def _random_permutation(self, n: int) -> np.ndarray:
        perm = np.arange(n)
        self.np_rng.shuffle(perm)
        return perm

    def _tour_length(self, tour: np.ndarray, D: np.ndarray) -> float:
        # length = sum of directed edges including return to start
        # vectorized wrap-around
        idx_from = tour
        idx_to = np.roll(tour, -1)
        return float(np.sum(D[idx_from, idx_to]))

    def _k_tournament(self, population: List[np.ndarray], fitness: np.ndarray, k: int) -> np.ndarray:
        # select k distinct random candidates
        k = max(1, min(k, len(population)))
        candidates = self.rng.sample(range(len(population)), k)
        # pick the one with minimal fitness (since we minimise tour length)
        best_idx = min(candidates, key=lambda i: fitness[i])
        return population[best_idx]

    def _two_opt_once(self, tour: np.ndarray, D: np.ndarray) -> bool:
        """执行一次2-opt改进，如果找到改进则返回True"""
        n = len(tour)
        best_delta = 0.0
        best_i = best_j = -1
        # 采样若干对 (i,j) 而非全枚举，提速
        tries = min(2000, n * 20)  # 可调
        for _ in range(tries):
            i = self.np_rng.integers(0, n-3)
            j = self.np_rng.integers(i+2, n-1)
            a, b = tour[i], tour[(i+1) % n]
            c, d = tour[j], tour[(j+1) % n]
            delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
            if delta < best_delta:
                best_delta = delta
                best_i, best_j = i, j
        if best_delta < 0:
            # 反转 i+1..j
            tour[best_i+1:best_j+1] = tour[best_i+1:best_j+1][::-1]
            return True
        return False

    def _light_two_opt(self, tour: np.ndarray, D: np.ndarray, max_steps: int = 200) -> bool:
        """执行轻量级2-opt局部搜索"""
        improved = False
        for _ in range(max_steps):
            if not self._two_opt_once(tour, D):
                break
            improved = True
        return improved

    def _rotate_to_start(self, tour: np.ndarray, start_city: int) -> np.ndarray:
        # rotate array so that start_city is first
        n = len(tour)
        pos = int(np.where(tour == start_city)[0][0])
        return np.concatenate([tour[pos:], tour[:pos]])

    def _erx(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        Edge Recombination Crossover (ERX).
        Builds an edge map from both parents, then constructs the child
        by repeatedly selecting the next city that shares the fewest remaining edges.
        Preserves adjacency information from both parents.
        """
        n = len(p1)
        # build edge map: city -> set of neighbors in both parents
        edge_map = {city: set() for city in p1}
        for parent in (p1, p2):
            for i in range(n):
                left = parent[i - 1]
                right = parent[(i + 1) % n]
                edge_map[parent[i]].update((left, right))

        child = []
        current = int(self.rng.choice(p1))
        child.append(current)

        while len(child) < n:
            # remove current from all neighbor sets
            for s in edge_map.values():
                s.discard(current)

            # choose next city
            neighbors = edge_map[current]
            if neighbors:
                # pick neighbor with fewest edges
                next_city = min(neighbors, key=lambda c: len(edge_map[c]))
            else:
                # all used, pick random unused city
                remaining = [c for c in p1 if c not in child]
                next_city = self.rng.choice(remaining)

            child.append(next_city)
            current = next_city

        return np.array(child, dtype=int)

    # ---- CSCX recombination ----
    def _cscx(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        Cycle-Subsequence Crossover (CSCX).
        Idea:
          - Choose a random start city s.
          - Follow the order of p1 from s, appending cities until we encounter a city already in child.
          - Switch to p2 and continue from the last appended city’s successor in p2.
          - Alternate whenever a duplicate would occur.
          - Continue until child contains all cities.
        Properties:
          - Produces a valid permutation.
          - Preserves contiguous subsequences from both parents.
        """
        n = len(p1)
        child = np.full(n, -1, dtype=int)
        in_child = np.zeros(n, dtype=bool)

        # successors maps for quick "next after city" lookup
        next_in = lambda parent: np.roll(parent, -1)

        p1_next = next_in(p1)
        p2_next = next_in(p2)

        # index lookup city -> position
        pos1 = np.empty(n, dtype=int)
        pos2 = np.empty(n, dtype=int)
        pos1[p1] = np.arange(n)
        pos2[p2] = np.arange(n)

        # start city
        s = p1[self.rng.randrange(n)]
        cur = s
        use_parent = 1  # 1 => p1_next, 2 => p2_next
        t = 0
        while t < n:
            if not in_child[cur]:
                child[t] = cur
                in_child[cur] = True
                t += 1
                # advance along current parent
                if use_parent == 1:
                    cur = p1_next[pos1[cur]]
                else:
                    cur = p2_next[pos2[cur]]
            else:
                # city already present, switch parent and move to its successor
                use_parent = 2 if use_parent == 1 else 1
                if use_parent == 1:
                    cur = p1_next[pos1[cur]]
                else:
                    cur = p2_next[pos2[cur]]

            # if we get stuck on already-used cities, jump to first unused city
            if t < n and in_child[cur]:
                # quick check to avoid infinite loops
                unused = np.where(~in_child)[0]
                if unused.size > 0 and self.rng.random() < 0.1:
                    cur = int(unused[self.rng.randrange(unused.size)])

        return child


if __name__ == "__main__":
    ea = r0123456(N_RUNS=10000, lam=1000, mu=500, use_cscx=True)
    ea.optimize("tour250.csv")