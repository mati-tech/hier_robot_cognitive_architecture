"""LEVEL 3 (bottom): LQR path tracker + Kalman filter (EKF) for the pose.

LQR keeps the robot on the path that D* produced: the tracking error (along, across,
heading) is fed back through the gain from the discrete Riccati equation.
The EKF estimates the pose: prediction from wheel odometry, correction from the
RTAB-Map pose.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import solve_discrete_are

V_MAX, W_MAX = 0.22, 2.84  # TurtleBot3 Burger limits (m/s, rad/s)


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# ============================================================================ path
class Path:
    """Dense polyline with arc length, heading and curvature at every point."""

    def __init__(self, pts: Sequence[Tuple[float, float]], step: float = 0.02):
        pts = np.asarray(pts, float)
        dense = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(int(np.linalg.norm(b - a) / step), 1)
            for k in range(1, n + 1):
                dense.append(a + (b - a) * k / n)
        self.p = np.asarray(dense)
        d = np.diff(self.p, axis=0)
        seg = np.hypot(d[:, 0], d[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        th = np.arctan2(d[:, 1], d[:, 0])
        self.th = np.concatenate([th, th[-1:]])
        dth = np.array([wrap(b - a) for a, b in zip(self.th[:-1], self.th[1:])])
        ds = np.maximum(np.concatenate([seg, seg[-1:]]), 1e-6)
        kappa = np.concatenate([dth, [0.0]]) / ds
        # light smoothing of curvature so the feedforward is not spiky
        self.kappa = np.convolve(kappa, np.ones(9) / 9, mode="same")
        self.length = float(self.s[-1])

    def project(self, x: float, y: float, hint: int = 0, window: int = 200) -> int:
        lo, hi = max(hint - 20, 0), min(hint + window, len(self.p))
        d = np.hypot(self.p[lo:hi, 0] - x, self.p[lo:hi, 1] - y)
        return lo + int(np.argmin(d))

    def cross_track(self, x: float, y: float, i: int) -> float:
        dx, dy = x - self.p[i, 0], y - self.p[i, 1]
        return -math.sin(self.th[i]) * dx + math.cos(self.th[i]) * dy


def smooth_corners(pts: Sequence[Tuple[float, float]], iters: int = 3) -> List[Tuple[float, float]]:
    """Chaikin corner cutting: turns a waypoint polyline into a drivable curve."""
    p = [tuple(map(float, q)) for q in pts]
    for _ in range(iters):
        out = [p[0]]
        for a, b in zip(p[:-1], p[1:]):
            out.append((0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]))
            out.append((0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]))
        out.append(p[-1])
        p = out
    return p


# ============================================================================ LQR
@dataclass
class LqrTracker:
    dt: float = 0.05
    q: Tuple[float, float, float] = (2.0, 12.0, 4.0)   # weights on e1, e2 (cross-track), e3
    r: Tuple[float, float] = (1.0, 0.6)               # weights on u1 (speed), u2 (turn rate)
    v_ref: float = 0.18
    slow_down_dist: float = 0.4
    goal_tol: float = 0.05
    idx: int = 0

    def reset(self):
        self.idx = 0

    def gain(self, v_r: float, w_r: float) -> np.ndarray:
        A = np.array([[0, w_r, 0], [-w_r, 0, v_r], [0, 0, 0]])
        B = np.array([[1, 0], [0, 0], [0, 1]], float)
        Ad, Bd = np.eye(3) + A * self.dt, B * self.dt
        Q, R = np.diag(self.q), np.diag(self.r)
        P = solve_discrete_are(Ad, Bd, Q, R)
        return np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)

    def command(self, pose: Tuple[float, float, float], path: Path) -> Tuple[float, float, bool]:
        x, y, th = pose
        self.idx = path.project(x, y, self.idx)
        remaining = path.length - path.s[self.idx]
        gx, gy = path.p[-1]
        if remaining < self.goal_tol or (self.idx >= len(path.p) - 2 and math.hypot(gx - x, gy - y) < 2 * self.goal_tol):
            return 0.0, 0.0, True
        # reference a little ahead of the projection keeps the robot moving forward
        j = min(self.idx + 1, len(path.p) - 1)
        xr, yr, thr = path.p[j, 0], path.p[j, 1], path.th[j]
        v_r = self.v_ref * min(1.0, max(remaining / self.slow_down_dist, 0.25))
        w_r = v_r * path.kappa[j]
        e = np.array([
            math.cos(th) * (xr - x) + math.sin(th) * (yr - y),
            -math.sin(th) * (xr - x) + math.cos(th) * (yr - y),
            wrap(thr - th),
        ])
        u = -self.gain(v_r, w_r) @ e
        v = v_r * math.cos(e[2]) - u[0]
        w = w_r - u[1]
        return float(np.clip(v, 0.0, V_MAX)), float(np.clip(w, -W_MAX, W_MAX)), False


# ============================================================================ Kalman
@dataclass
class Ekf:
    """Extended Kalman filter for the unicycle state (x, y, th)."""
    q_v: float = 0.02      # std of odometry speed error, m/s
    q_w: float = 0.05      # std of odometry turn-rate error, rad/s
    x: np.ndarray = field(default_factory=lambda: np.zeros(3))
    P: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.05)
    initialised: bool = False

    def init(self, pose, cov=None):
        self.x = np.array(pose, float)
        self.P = np.diag([0.05, 0.05, 0.05]) if cov is None else np.asarray(cov, float)
        self.initialised = True

    def predict(self, v: float, w: float, dt: float):
        th = self.x[2]
        self.x = self.x + np.array([v * math.cos(th) * dt, v * math.sin(th) * dt, w * dt])
        self.x[2] = wrap(self.x[2])
        F = np.array([[1, 0, -v * math.sin(th) * dt], [0, 1, v * math.cos(th) * dt], [0, 0, 1]])
        G = np.array([[math.cos(th) * dt, 0], [math.sin(th) * dt, 0], [0, dt]])
        Qu = np.diag([self.q_v ** 2, self.q_w ** 2])
        self.P = F @ self.P @ F.T + G @ Qu @ G.T + np.eye(3) * 1e-6

    def update(self, z: Sequence[float], R: np.ndarray):
        z = np.asarray(z, float)
        y = z - self.x
        y[2] = wrap(y[2])
        S = self.P + R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(3) - K) @ self.P

    @property
    def pose(self) -> Tuple[float, float, float]:
        return float(self.x[0]), float(self.x[1]), float(self.x[2])
