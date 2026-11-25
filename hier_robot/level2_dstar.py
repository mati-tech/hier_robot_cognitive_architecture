"""LEVEL 2 (middle): geometric planner D* Lite on the RTAB-Map occupancy grid.
Middle level of the hierarchy: D* Lite geometric planner on an occupancy grid.

D* Lite (Koenig & Likhachev, 2002) searches backwards from the goal to the robot.
When some cells of the map change (SLAM adds a wall, the LiDAR sees a new obstacle,
or the robot moves), it repairs only the part of the search that is affected
instead of planning from scratch. That is why it suits a map that RTAB-Map keeps
growing while the robot drives.

Grid costs (per cell):
  * lethal (inf):   occupied cells inflated by the robot radius, and no-go zones
                    (e.g. the forbidden server room) passed in by the top level
  * high:           unknown cells (allowed, but avoided when a known route exists)
  * graded:         cells close to obstacles (keeps the path off the walls)
  * 1:              free cells
Moves are 8-connected; the cost of a move is its length times the mean cell cost.
"""
from __future__ import annotations

import heapq
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import time

import numpy as np

from .level3_lqr import Path, smooth_corners

INF = float("inf")
Cell = Tuple[int, int]  # (row, col)
NEIGH = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


# ============================================================================ costmap
def costmap_from_grid(occ: np.ndarray, resolution: float, robot_radius: float = 0.20,
                      inflation: float = 0.45, unknown_cost: float = 5.0,
                      no_go: Optional[np.ndarray] = None) -> np.ndarray:
    """occ: int8 occupancy (ROS convention: -1 unknown, 0..100 occupied probability).
    Returns float cost per cell (1 = free, inf = lethal)."""
    occupied = occ >= 50
    unknown = occ < 0
    dist = _distance_to(occupied) * resolution            # metres to the nearest obstacle
    cost = np.ones(occ.shape, float)
    near = (dist > robot_radius) & (dist < inflation)
    cost[near] += 4.0 * (inflation - dist[near]) / (inflation - robot_radius)
    cost[unknown] = np.maximum(cost[unknown], unknown_cost)
    cost[dist <= robot_radius] = INF
    if no_go is not None:
        cost[no_go] = INF
    return cost


def _distance_to(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (in cells) to the nearest True cell; two-pass chamfer approximation."""
    big = 1e9
    d = np.where(mask, 0.0, big)
    H, W = d.shape
    a, b = 1.0, math.sqrt(2)
    for r in range(H):                     # forward pass
        row = d[r]
        if r > 0:
            up = d[r - 1]
            row = np.minimum(row, up + a)
            row[1:] = np.minimum(row[1:], up[:-1] + b)
            row[:-1] = np.minimum(row[:-1], up[1:] + b)
        for c in range(1, W):
            if row[c - 1] + a < row[c]:
                row[c] = row[c - 1] + a
        d[r] = row
    for r in range(H - 1, -1, -1):         # backward pass
        row = d[r]
        if r < H - 1:
            dn = d[r + 1]
            row = np.minimum(row, dn + a)
            row[1:] = np.minimum(row[1:], dn[:-1] + b)
            row[:-1] = np.minimum(row[:-1], dn[1:] + b)
        for c in range(W - 2, -1, -1):
            if row[c + 1] + a < row[c]:
                row[c] = row[c + 1] + a
        d[r] = row
    return d


# ============================================================================ D* Lite
class DStarLite:
    def __init__(self, cost: np.ndarray, start: Cell, goal: Cell):
        self.cost = cost.astype(float).copy()
        self.H, self.W = cost.shape
        self.s_start, self.s_goal = start, goal
        self.s_last = start
        self.km = 0.0
        self.g: Dict[Cell, float] = {}
        self.rhs: Dict[Cell, float] = {goal: 0.0}
        self.U: List[Tuple[Tuple[float, float], Cell]] = []
        self.in_U: Dict[Cell, Tuple[float, float]] = {}
        self._push(goal, self._key(goal))
        self.expanded = 0

    # ---- helpers ------------------------------------------------------------
    def _h(self, a: Cell, b: Cell) -> float:
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)    # octile distance

    def _gv(self, s: Cell) -> float:
        return self.g.get(s, INF)

    def _rv(self, s: Cell) -> float:
        return self.rhs.get(s, INF)

    def _key(self, s: Cell) -> Tuple[float, float]:
        m = min(self._gv(s), self._rv(s))
        # rounded: ties in the first element are common on a grid, and float noise in km
        # must not decide the lexicographic comparison (it made the repair stop too early)
        return (round(m + self._h(self.s_start, s) + self.km, 6), round(m, 6))

    def _push(self, s: Cell, k):
        self.in_U[s] = k
        heapq.heappush(self.U, (k, s))

    def _neighbours(self, s: Cell) -> Iterable[Cell]:
        r, c = s
        for dr, dc in NEIGH:
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.H and 0 <= cc < self.W:
                yield (rr, cc)

    def _c(self, a: Cell, b: Cell) -> float:
        ca, cb = self.cost[a], self.cost[b]
        if ca == INF or cb == INF:
            return INF
        step = math.sqrt(2) if (a[0] != b[0] and a[1] != b[1]) else 1.0
        if step > 1.0:   # no corner cutting past a lethal cell
            if self.cost[a[0], b[1]] == INF or self.cost[b[0], a[1]] == INF:
                return INF
        return step * 0.5 * (ca + cb)

    def _update_vertex(self, u: Cell):
        if u != self.s_goal:
            best = INF
            for s in self._neighbours(u):
                v = self._c(u, s) + self._gv(s)
                if v < best:
                    best = v
            self.rhs[u] = best
        self.in_U.pop(u, None)          # lazy removal: stale heap entries are skipped
        if self._gv(u) != self._rv(u):
            self._push(u, self._key(u))

    def _top(self):
        while self.U:
            k, s = self.U[0]
            if self.in_U.get(s) == k:
                return k, s
            heapq.heappop(self.U)
        return (INF, INF), None

    def compute(self, max_expansions: int = 2_000_000, target: Optional[Cell] = None) -> bool:
        """ComputeShortestPath. `target` (default: the robot cell) must end locally consistent
        with every queued key above its own, so its g value is the true cost-to-goal."""
        t = self.s_start if target is None else target
        while True:
            k_old, u = self._top()
            if u is None:
                break
            if not (k_old < self._key(t) or self._rv(t) != self._gv(t)):
                break
            if self.expanded >= max_expansions:
                return False
            self.expanded += 1
            heapq.heappop(self.U)
            del self.in_U[u]
            k_new = self._key(u)
            if k_old < k_new:
                self._push(u, k_new)
            elif self._gv(u) > self._rv(u):
                self.g[u] = self._rv(u)
                for s in self._neighbours(u):
                    self._update_vertex(s)
            else:
                self.g[u] = INF
                self._update_vertex(u)
                for s in self._neighbours(u):
                    self._update_vertex(s)
        return self._gv(t) < INF

    # ---- incremental API ------------------------------------------------------
    def move_start(self, new_start: Cell):
        """The robot moved: D* Lite keeps the search and only shifts the key offset."""
        self.km += self._h(self.s_last, new_start)
        self.s_last = self.s_start = new_start

    def update_costs(self, changes: Dict[Cell, float]):
        """Some cells changed cost (new obstacle, map update). Repair only what is affected."""
        touched = set()
        for cell, new_cost in changes.items():
            if self.cost[cell] == new_cost:
                continue
            self.cost[cell] = new_cost
            touched.add(cell)
            touched.update(self._neighbours(cell))
        for u in touched:
            self._update_vertex(u)

    def path(self, max_len: int = 100_000) -> List[Cell]:
        if self._gv(self.s_start) == INF:
            return []
        s, out = self.s_start, [self.s_start]
        while s != self.s_goal and len(out) < max_len:
            best, nxt = INF, None
            for n in self._neighbours(s):
                if self._c(s, n) == INF:
                    continue
                # after a cost change cells far from the robot may still hold stale g values;
                # make every candidate consistent first (usually a no-op)
                self.compute(target=n)
                v = self._c(s, n) + self._gv(n)
                if v < best:
                    best, nxt = v, n
            if nxt is None or best == INF:
                return []
            s = nxt
            out.append(s)
        return out


# ============================================================================ helpers
class GridFrame:
    """Conversion between map metres and grid cells (ROS OccupancyGrid convention:
    data row 0 = lowest y)."""

    def __init__(self, origin_x: float, origin_y: float, resolution: float, H: int, W: int):
        self.ox, self.oy, self.res, self.H, self.W = origin_x, origin_y, resolution, H, W

    def to_cell(self, x: float, y: float) -> Cell:
        c = int((x - self.ox) / self.res)
        r = int((y - self.oy) / self.res)
        return (min(max(r, 0), self.H - 1), min(max(c, 0), self.W - 1))

    def to_xy(self, cell: Cell) -> Tuple[float, float]:
        r, c = cell
        return (self.ox + (c + 0.5) * self.res, self.oy + (r + 0.5) * self.res)


def nearest_free(cost: np.ndarray, cell: Cell, max_r: int = 20) -> Optional[Cell]:
    """Closest non-lethal cell (the robot or a goal may sit on an inflated edge)."""
    if cost[cell] < INF:
        return cell
    r0, c0 = cell
    for rad in range(1, max_r + 1):
        best = None
        for r in range(r0 - rad, r0 + rad + 1):
            for c in range(c0 - rad, c0 + rad + 1):
                if 0 <= r < cost.shape[0] and 0 <= c < cost.shape[1] and cost[r, c] < INF:
                    d = (r - r0) ** 2 + (c - c0) ** 2
                    if best is None or d < best[0]:
                        best = (d, (r, c))
        if best:
            return best[1]
    return None


def simplify(points: Sequence[Tuple[float, float]], every: int = 6) -> List[Tuple[float, float]]:
    """Keep every n-th point of a cell path (plus the end) before smoothing."""
    if len(points) <= 2:
        return list(points)
    out = list(points[::every])
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


def polygon_mask(frame: GridFrame, polys: Sequence[Sequence[Tuple[float, float]]]) -> np.ndarray:
    """Boolean grid mask of cells inside any of the polygons (e.g. forbidden rooms)."""
    mask = np.zeros((frame.H, frame.W), bool)
    ys = frame.oy + (np.arange(frame.H) + 0.5) * frame.res
    xs = frame.ox + (np.arange(frame.W) + 0.5) * frame.res
    X, Y = np.meshgrid(xs, ys)
    for poly in polys:
        inside = np.zeros_like(mask)
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            cond = ((y1 > Y) != (y2 > Y)) & (X < (x2 - x1) * (Y - y1) / (y2 - y1 + 1e-12) + x1)
            inside ^= cond
        mask |= inside
    return mask


def load_map_yaml(path: str):
    """Read a map_server map (yaml + pgm) into ROS OccupancyGrid convention:
    int8 grid, row 0 = lowest y, -1 unknown, 0 free, 100 occupied. Returns (occ, GridFrame)."""
    import os
    import yaml
    with open(path) as f:
        meta = yaml.safe_load(f)
    img_path = os.path.join(os.path.dirname(path), meta["image"])
    with open(img_path, "rb") as f:
        data = f.read()
    # binary PGM (P5) header: magic, width, height, maxval (comments not used by our writer)
    parts, pos = [], 0
    while len(parts) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            pos = data.index(b"\n", pos) + 1
            continue
        end = pos
        while not data[end:end + 1].isspace():
            end += 1
        parts.append(data[pos:end])
        pos = end
    W, H, maxval = int(parts[1]), int(parts[2]), int(parts[3])
    pix = np.frombuffer(data[pos + 1:pos + 1 + W * H], np.uint8).reshape(H, W).astype(float)
    p = (maxval - pix) / maxval if not meta.get("negate", 0) else pix / maxval
    occ = np.full((H, W), -1, np.int8)
    occ[p > meta.get("occupied_thresh", 0.65)] = 100
    occ[p < meta.get("free_thresh", 0.25)] = 0
    occ = occ[::-1].copy()                     # image row 0 is the top (highest y)
    ox, oy = meta["origin"][:2]
    return occ, GridFrame(ox, oy, meta["resolution"], H, W)


def plan_path(cost: np.ndarray, frame: GridFrame, start_xy, goal_xy):
    """One-shot helper: D* Lite from start to goal, returns (list of (x, y), planner) or ([], planner)."""
    s = nearest_free(cost, frame.to_cell(*start_xy))
    g = nearest_free(cost, frame.to_cell(*goal_xy))
    if s is None or g is None:
        return [], None
    ds = DStarLite(cost, s, g)
    ds.compute()
    return [frame.to_xy(c) for c in ds.path()], ds


# ============================================================================ layer used by the robot
class GeometricLayer:
    """D* Lite on a costmap made from the SLAM occupancy grid."""

    def __init__(self, occ: np.ndarray, frame: GridFrame, no_go_polys: Sequence = (),
                 robot_radius: float = 0.20, inflation: float = 0.45, allowed_polys: Sequence = ()):
        self.frame, self.robot_radius, self.inflation = frame, robot_radius, inflation
        self.no_go = polygon_mask(frame, no_go_polys) if no_go_polys else None
        # everything outside the building (outside all room polygons) is a no-go zone too:
        # SLAM has not seen it, so it is "unknown", and D* must never plan a way through it
        if allowed_polys:
            outside = ~polygon_mask(frame, allowed_polys)
            self.no_go = outside if self.no_go is None else (self.no_go | outside)
        self.occ = occ.copy()
        self.cost = costmap_from_grid(self.occ, frame.res, robot_radius, inflation, no_go=self.no_go)
        self.ds: Optional[DStarLite] = None
        self.goal_xy = None
        self.stats = {"plans": 0, "replans": 0, "expanded_initial": 0, "expanded_replan": 0,
                      "t_initial_s": 0.0, "t_replan_s": 0.0}

    def in_no_go(self, x: float, y: float) -> bool:
        return self.no_go is not None and bool(self.no_go[self.frame.to_cell(x, y)])

    def plan(self, start_xy, goal_xy) -> Tuple[Optional[Path], str]:
        if self.in_no_go(*goal_xy):
            return None, "D*: цель в запретной зоне"
        s = nearest_free(self.cost, self.frame.to_cell(*start_xy))
        g = nearest_free(self.cost, self.frame.to_cell(*goal_xy), max_r=16)
        if s is None or g is None:
            return None, "D*: старт или цель внутри препятствия"
        t0 = time.time()
        self.ds = DStarLite(self.cost, s, g)
        ok = self.ds.compute()
        self.goal_xy = goal_xy
        self.stats["plans"] += 1
        self.stats["expanded_initial"] += self.ds.expanded
        self.stats["t_initial_s"] += time.time() - t0
        if not ok:
            return None, "D*: путь не найден"
        path = self._path()
        return (path, "ok") if path is not None else (None, "D*: цель совпадает со стартом")

    def _path(self) -> Optional[Path]:
        cells = self.ds.path()
        if len(cells) < 2:
            return None
        pts = simplify([self.frame.to_xy(c) for c in cells], every=4)
        pts[-1] = tuple(self.goal_xy)
        return Path(smooth_corners(pts, iters=2))

    def update_map(self, occ: np.ndarray, robot_xy) -> Tuple[Optional[Path], str, int]:
        """New occupancy grid (SLAM update or LiDAR obstacles): repair the D* search
        incrementally from the robot's current cell. Returns (path, msg, changed_cells)."""
        self.occ = occ.copy()
        new_cost = costmap_from_grid(self.occ, self.frame.res, self.robot_radius, self.inflation, no_go=self.no_go)
        diff = np.argwhere(new_cost != self.cost)
        self.cost = new_cost
        if self.ds is None or len(diff) == 0:
            return None, "без изменений", 0
        t0 = time.time()
        e0 = self.ds.expanded
        cell = nearest_free(new_cost, self.frame.to_cell(*robot_xy))
        if cell is None:
            return None, "D*: робот внутри препятствия", len(diff)
        if new_cost[self.ds.s_goal] == INF:
            # the goal turned out to be inside an obstacle that SLAM just mapped (e.g. a shelf):
            # choose the nearest free cell as the new goal and plan again
            path, msg = self.plan(robot_xy, self.goal_xy)
            self.stats["replans"] += 1
            return path, msg, len(diff)
        self.ds.move_start(cell)
        self.ds.update_costs({(int(r), int(c)): float(new_cost[r, c]) for r, c in diff})
        ok = self.ds.compute()
        self.stats["replans"] += 1
        self.stats["expanded_replan"] += self.ds.expanded - e0
        self.stats["t_replan_s"] += time.time() - t0
        if not ok:
            return None, "D*: путь перекрыт, обхода нет", len(diff)
        return self._path(), "ok", len(diff)
