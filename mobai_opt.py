"""Warehouse optimisation for the MobAI WMS dataset.

Three modules, following the design we presented at the hackathon:
  A. inbound placement   (receipt -> reserve slot), weighted score + optimal assignment
  B. pick-face slotting  (fast movers close to expedition and at low levels)
  C. multi-trolley routing on a grid: A* with space-time reservations

The real location codes are used (picking: <level><row>-<bay>-<pos>, reserve:
B07-N<floor>-<row><slot>), but the metric geometry is NOT in the data, so the
grid below is a schematic reconstruction (1 cell ~ 1 m). All distances are in
grid steps.
"""
from __future__ import annotations
import heapq, re
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

# ----------------------------------------------------------------- layout
ROWS = list('ABCDEFGHIKMNPQRSTVWX')          # picking rack rows present in the data
W, Hh = 19, 32                                # ground-floor grid size
EXPEDITION, RECEPTION, ELEVATOR = (0, 31), (18, 0), (18, 31)
LEVEL_COST = 3                                # extra steps-equivalent per rack level (reach time)
FLOOR_COST = 12                               # steps-equivalent for one elevator floor


def rack_y(row_idx):
    pair, side = divmod(row_idx, 2)
    return 3 * pair + 1 + 2 * side            # back-to-back racks, aisle between each pair


def aisle_y(row_idx):
    return 3 * (row_idx // 2) + 2


def build_grid():
    """0 = free floor, 1 = rack. Cross aisles at x in {0,1,17,18}, y in {0,31}."""
    g = np.zeros((Hh, W), dtype=np.int8)
    for k in range(len(ROWS)):
        g[rack_y(k), 2:17] = 1
    return g


def parse_locations(emp: pd.DataFrame) -> pd.DataFrame:
    """Turn location codes into coordinates."""
    out = []
    for _, r in emp.iterrows():
        code = str(r.code_emplacement)
        m = re.fullmatch(r'(\d)([A-Z])-(\d\d)-(\d\d)', code)
        if m and r.type_emplacement == 'PICKING':
            lvl, row, bay, pos = int(m[1]), m[2], int(m[3]), int(m[4])
            if row not in ROWS: continue
            k = ROWS.index(row)
            x = min(2 + (bay - 1) * 3 + (pos - 1), 16)
            out.append(dict(code=code, kind='PICK', floor=0, level=lvl, x=x, y=aisle_y(k), rack_y=rack_y(k)))
            continue
        m = re.fullmatch(r'B07-N(\d)-([A-Z])(\d+)', code)
        if m and r.type_emplacement == 'RESERVE':
            fl, row, slot = int(m[1]), m[2], int(m[3])
            ri = ord(row) - ord('A')
            out.append(dict(code=code, kind='RESERVE', floor=fl, level=0, x=slot, y=3 * ri + 2, rack_y=None))
    return pd.DataFrame(out)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ------------------------------------------------- A. inbound placement
def reserve_distances(L: pd.DataFrame) -> pd.DataFrame:
    """Normalised distances for reserve slots (other floors are reached by elevator)."""
    R = L[L.kind == 'RESERVE'].copy()
    elev_local = (0, 2)                                   # elevator door on each upper floor
    R['d_elev'] = [manhattan((x, y), elev_local) for x, y in zip(R.x, R.y)]
    v = R.floor * FLOOR_COST
    R['d_exp'] = R.d_elev + v + manhattan(ELEVATOR, EXPEDITION)
    R['d_rec'] = R.d_elev + v + manhattan(ELEVATOR, RECEPTION)
    for c in ['d_exp', 'd_rec', 'd_elev']:
        R[c + '_n'] = (R[c] - R[c].min()) / (R[c].max() - R[c].min())
    return R.reset_index(drop=True)


def placement_cost_matrix(P: pd.DataFrame, R: pd.DataFrame, alpha=1.0, beta=0.2, gamma=0.3, delta=0.5):
    """score(p,c) = freq_p*(a*d_exp + b*d_rec + g*d_elev) + d*bulk_p*floor_c
    freq makes fast movers pay more for distance; bulk keeps big pallets low."""
    base = alpha * R.d_exp_n.values + beta * R.d_rec_n.values + gamma * R.d_elev_n.values
    fl = R.floor.values / R.floor.max()
    return (P.freq.values[:, None] * base[None, :]
            + delta * P.bulk.values[:, None] * fl[None, :])


def assign_random(C, seed=0):
    rng = np.random.default_rng(seed)
    return rng.permutation(C.shape[1])[:C.shape[0]]


def assign_greedy(C, order):
    """Pallets in the given order each take their cheapest free slot (online rule)."""
    free = np.ones(C.shape[1], bool); out = np.empty(C.shape[0], int)
    for i in order:
        j = np.where(free)[0][np.argmin(C[i, free])]
        out[i] = j; free[j] = False
    return out


def assign_optimal(C):
    """Exact minimum-cost assignment (Hungarian algorithm)."""
    r, c = linear_sum_assignment(C)
    out = np.empty(C.shape[0], int); out[r] = c
    return out


# ------------------------------------------------- B. pick-face slotting
def pick_cost(L: pd.DataFrame) -> pd.Series:
    Pk = L[L.kind == 'PICK']
    d = [2 * manhattan(EXPEDITION, (x, y)) + LEVEL_COST * lv for x, y, lv in zip(Pk.x, Pk.y, Pk.level)]
    return pd.Series(d, index=Pk.index)


def slot_by_frequency(freq: np.ndarray, cost: np.ndarray):
    """Rearrangement inequality: highest frequency -> cheapest slot is optimal
    for a separable sum(freq_i * cost_slot(i))."""
    return np.argsort(cost)[np.argsort(np.argsort(-freq))]


# ------------------------------------------------- C. routing
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]


def astar(grid, start, goal, t0=0, reserved=None, edges=None, free_cells=(), horizon=400):
    """Space-time A*. reserved: set of (x,y,t); edges: set of (x1,y1,x2,y2,t).
    Cells in free_cells (docks) may be shared. Returns list of (x,y) from t0."""
    reserved = reserved or set(); edges = edges or set()
    h = lambda p: manhattan(p, goal)
    openh = [(h(start), 0, start, t0)]; came = {(start, t0): None}
    while openh:
        f, g, p, t = heapq.heappop(openh)
        if p == goal:
            path = []; node = (p, t)
            while node: path.append(node[0]); node = came[node]
            return path[::-1]
        if t - t0 > horizon: continue
        for dx, dy in MOVES:
            q = (p[0] + dx, p[1] + dy); nt = t + 1
            if not (0 <= q[0] < grid.shape[1] and 0 <= q[1] < grid.shape[0]): continue
            if grid[q[1], q[0]]: continue
            if q not in free_cells and (q[0], q[1], nt) in reserved: continue
            if (q[0], q[1], p[0], p[1], t) in edges: continue           # head-on swap
            if (q, nt) in came: continue
            came[(q, nt)] = (p, t)
            heapq.heappush(openh, (nt - t0 + h(q) + (0.01 if (dx, dy) == (0, 0) else 0), g + 1, q, nt))
    return None


def tour_order(points, start):
    """Nearest-neighbour order of pick faces, then back to start."""
    left = list(dict.fromkeys(points)); cur = start; out = []
    while left:
        nxt = min(left, key=lambda q: manhattan(cur, q)); out.append(nxt); left.remove(nxt); cur = nxt
    return out + [start]


def plan_fleet(grid, tours, starts, cooperative=True, free_cells=()):
    """Plan every trolley's full tour. Prioritised planning: trolley k plans after
    0..k-1 and respects their reservations. Returns paths and statistics."""
    reserved, edges, paths = set(), set(), []
    for tour, (s, t0) in zip(tours, starts):
        path = [s]; t = t0; cur = s
        for wp in tour:
            seg = astar(grid, cur, wp, t, reserved if cooperative else None,
                        edges if cooperative else None, free_cells)
            if seg is None: raise RuntimeError('no path')
            path += seg[1:]; t += len(seg) - 1; cur = wp
        for i, (x, y) in enumerate(path):
            tt = t0 + i; reserved.add((x, y, tt))
            if i: px, py = path[i - 1]; edges.add((px, py, x, y, tt - 1))
        paths.append((t0, path))
    return paths


def count_conflicts(paths, free_cells=()):
    occ = {}; conflicts = 0
    for k, (t0, p) in enumerate(paths):
        for i, c in enumerate(p):
            if c in free_cells: continue
            key = (c, t0 + i)
            if key in occ and occ[key] != k: conflicts += 1
            occ[key] = k
    swaps = 0
    mv = {}
    for k, (t0, p) in enumerate(paths):
        for i in range(1, len(p)):
            mv[(p[i - 1], p[i], t0 + i)] = k
    for (a, b, t), k in mv.items():
        if a != b and (b, a, t) in mv and mv[(b, a, t)] != k: swaps += 1
    return conflicts + swaps // 2


def path_stats(paths):
    steps = sum(len(p) - 1 for _, p in paths)
    waits = sum(sum(1 for i in range(1, len(p)) if p[i] == p[i - 1]) for _, p in paths)
    makespan = max(t0 + len(p) - 1 for t0, p in paths) - min(t0 for t0, _ in paths)
    return dict(time_steps=steps, waits=waits, makespan=makespan)
