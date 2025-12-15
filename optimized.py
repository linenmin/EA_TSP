# Write the implemented EA into the provided template file.

import Reporter
import numpy as np
from typing import List
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

if NUMBA_OK:
    @njit(cache=True, fastmath=True)
    def _erx_jit(p1, p2):
        n = p1.size
        child = np.empty(n, np.int32)
        used = np.zeros(n, np.uint8)
        neighbors = np.full((n, 4), -1, np.int32)
        deg = np.zeros(n, np.int32)

        def add_edge(u, v):
            if v == u:
                return
            for k in range(deg[u]):
                if neighbors[u, k] == v:
                    return
            if deg[u] < 4:
                neighbors[u, deg[u]] = v
                deg[u] += 1

        for parent in (p1, p2):
            for i in range(n):
                c = parent[i]
                add_edge(c, parent[(i - 1) % n])
                add_edge(c, parent[(i + 1) % n])

        cur = p1[0]
        next_scan = 0

        for t in range(n):
            child[t] = cur
            used[cur] = 1

            best = -1
            best_score = 1_000_000
            for k in range(deg[cur]):
                nb = neighbors[cur, k]
                if nb == -1 or used[nb] == 1:
                    continue
                cnt = 0
                for j in range(deg[nb]):
                    x = neighbors[nb, j]
                    if x != -1 and used[x] == 0:
                        cnt += 1
                if cnt < best_score:
                    best_score = cnt
                    best = nb

            if best != -1:
                cur = best
            else:
                while next_scan < n and used[next_scan] == 1:
                    next_scan += 1
                if next_scan < n:
                    cur = next_scan
                else:
                    for r in range(n):
                        if used[r] == 0:
                            cur = r
                            break
        return child

# -------- numba kernels --------
if NUMBA_OK:
    @njit(cache=True, fastmath=True)
    def tour_length_jit(tour, D):
        n = tour.shape[0]
        s = 0.0
        for i in range(n - 1):
            s += D[tour[i], tour[i+1]]
        s += D[tour[n-1], tour[0]]
        return s

    @njit(cache=True, fastmath=True)
    def batch_lengths_jit(pop2d, D, out):
        m, n = pop2d.shape
        for r in range(m):
            s = 0.0
            row = pop2d[r]
            for i in range(n - 1):
                s += D[row[i], row[i+1]]
            s += D[row[n-1], row[0]]
            out[r] = s

def tour_length_np(tour: np.ndarray, D: np.ndarray) -> float:
    idx_from = tour
    idx_to = np.roll(tour, -1)
    return float(np.sum(D[idx_from, idx_to]))

class r0123456:

    def __init__(self,
                 N_RUNS: int = 500,
                 lam: int = 100,
                 mu: int = 100,
                 k_tournament: int = 5,
                 mutation_rate: float = 0.3,
                 use_cscx: bool = True,
                 rng_seed: int | None = None):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.N_RUNS = int(N_RUNS)
        self.lam = int(lam)
        self.mu = int(mu)
        self.k_tournament = int(k_tournament)
        self.mutation_rate = float(mutation_rate)
        self.use_cscx = use_cscx
        self.rng = np.random.default_rng(rng_seed)

    # ---- Core EA ----
    def optimize(self, filename: str):
        with open(filename) as file:
            D = np.loadtxt(file, delimiter=",", dtype=np.float64, ndmin=2)
        n = D.shape[0]
        D = np.ascontiguousarray(D)

        # population as 2D array for fast batch fitness
        population = np.empty((self.lam, n), dtype=np.int32)
        for i in range(self.lam):
            population[i] = self._random_permutation(n)

        fitness = np.empty(self.lam, dtype=np.float64)
        self._eval_batch(population, D, fitness)

        # prealloc buffers
        off_count = self.mu
        offspring = np.empty((off_count, n), dtype=np.int32)
        offspring_f = np.empty(off_count, dtype=np.float64)

        # union buffers for truncation
        union_size = self.lam + self.mu
        union_fit = np.empty(union_size, dtype=np.float64)

        print("generation,best_cost", flush=True)

        for gen in range(1, self.N_RUNS + 1):
            # --- reproduction ---
            o = 0
            while o < off_count:
                p1 = population[self._k_tournament_idx(fitness, self.k_tournament)]
                p2 = population[self._k_tournament_idx(fitness, self.k_tournament)]
                if self.use_cscx:
                    c1 = self._cscx(p1, p2)
                    c2 = self._cscx(p2, p1)
                else:
                    c1 = self._erx(p1, p2)
                    c2 = self._erx(p2, p1)
                if self.rng.random() < self.mutation_rate:
                    self._swap_mutation_inplace(c1)
                if self.rng.random() < self.mutation_rate:
                    self._swap_mutation_inplace(c2)
                offspring[o] = c1; o += 1
                if o < off_count:
                    offspring[o] = c2; o += 1

            # --- fitness ---
            self._eval_batch(offspring, D, offspring_f)

            # --- (λ+μ) truncation with argpartition (no full sort) ---
            # build virtual union view
            union_fit[:self.lam] = fitness
            union_fit[self.lam:] = offspring_f

            # indices into virtual union: 0..lam-1 are parents, lam..lam+mu-1 are offspring
            keep = np.argpartition(union_fit, self.lam - 1)[:self.lam]
            # materialize next population
            next_pop = np.empty_like(population)
            next_fit = np.empty_like(fitness)
            p_i = 0
            for idx in keep:
                if idx < self.lam:
                    next_pop[p_i] = population[idx]
                    next_fit[p_i] = fitness[idx]
                else:
                    next_pop[p_i] = offspring[idx - self.lam]
                    next_fit[p_i] = offspring_f[idx - self.lam]
                p_i += 1
            population, fitness = next_pop, next_fit

            # --- report ---
            best_idx = int(np.argmin(fitness))
            bestObjective = float(fitness[best_idx])
            bestSolution = self._rotate_to_start(population[best_idx].copy(), 0)
            meanObjective = float(fitness.mean())

            print(f"{gen},{bestObjective}", flush=True)

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

        return 0

    # ---- Helpers ----
    def _eval_batch(self, pop2d: np.ndarray, D: np.ndarray, out: np.ndarray) -> None:
        if NUMBA_OK:
            batch_lengths_jit(pop2d, D, out)
        else:
            # vectorized fallback
            for i in range(pop2d.shape[0]):
                out[i] = tour_length_np(pop2d[i], D)

    def _random_permutation(self, n: int) -> np.ndarray:
        return self.rng.permutation(n).astype(np.int32, copy=False)

    def _k_tournament_idx(self, fitness: np.ndarray, k: int) -> int:
        k = 1 if k < 1 else k
        k = min(k, fitness.shape[0])
        cand = self.rng.choice(fitness.shape[0], size=k, replace=False)
        # return index of minimal fitness among candidates
        best_local = np.argmin(fitness[cand])
        return int(cand[best_local])

    def _swap_mutation_inplace(self, tour: np.ndarray) -> None:
        n = tour.shape[0]
        i = int(self.rng.integers(n))
        j = int(self.rng.integers(n - 1))
        if j >= i:
            j += 1
        tour[i], tour[j] = tour[j], tour[i]

    def _rotate_to_start(self, tour: np.ndarray, start_city: int) -> np.ndarray:
        pos = int(np.where(tour == start_city)[0][0])
        if pos == 0:
            return tour
        return np.concatenate([tour[pos:], tour[:pos]])

    # ---- ERX ----
    def _erx(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        # Use JIT-compiled ERX if available
        if NUMBA_OK:
            return _erx_jit(p1.astype(np.int32, copy=False),
                            p2.astype(np.int32, copy=False))
        # fallback: simple array-based ERX (if numba missing)
        n = p1.shape[0]
        edge_map = {int(c): set() for c in p1.tolist()}
        for parent in (p1, p2):
            for i in range(n):
                c = int(parent[i])
                edge_map[c].add(int(parent[i - 1]))
                edge_map[c].add(int(parent[(i + 1) % n]))
        child = np.empty(n, dtype=np.int32)
        used = set()
        cur = int(p1[0])
        for t in range(n):
            child[t] = cur
            used.add(cur)
            for s in edge_map.values():
                s.discard(cur)
            neigh = edge_map[cur]
            if neigh:
                cur = min(neigh, key=lambda c: len(edge_map[c]))
            else:
                remaining = [c for c in p1 if c not in used]
                cur = remaining[0] if remaining else cur
        return child

    # ---- CSCX ----
    def _cscx(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        n = p1.shape[0]
        child = np.full(n, -1, dtype=np.int32)
        in_child = np.zeros(n, dtype=np.bool_)
        p1_next = np.roll(p1, -1)
        p2_next = np.roll(p2, -1)
        pos1 = np.empty(n, dtype=np.int32); pos1[p1] = np.arange(n, dtype=np.int32)
        pos2 = np.empty(n, dtype=np.int32); pos2[p2] = np.arange(n, dtype=np.int32)
        cur = int(p1[self.rng.integers(n)])
        use_parent = 1
        t = 0
        while t < n:
            if not in_child[cur]:
                child[t] = cur
                in_child[cur] = True
                t += 1
                cur = (p1_next[pos1[cur]] if use_parent == 1 else p2_next[pos2[cur]])
            else:
                use_parent = 2 if use_parent == 1 else 1
                cur = (p1_next[pos1[cur]] if use_parent == 1 else p2_next[pos2[cur]])
                if t < n and in_child[cur]:
                    # jump to a random unused city with small probability
                    if self.rng.random() < 0.1:
                        # scan for first unused (fast enough)
                        for c in range(n):
                            if not in_child[c]:
                                cur = c
                                break
        return child


if __name__ == "__main__":
    ea = r0123456(N_RUNS=10000000, lam=6000, mu=4000, use_cscx=False, mutation_rate=0.3)
    ea.optimize("tour500.csv")
