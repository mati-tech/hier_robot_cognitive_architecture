"""Tests for the hierarchical system: D* Lite, semantic tensor, LLM->D*->LQR loop."""
import heapq
import math

import numpy as np

from hier_robot.level2_dstar import INF, DStarLite, GridFrame, GeometricLayer, costmap_from_grid
from hier_robot.level1_llm import ActionExecutor, run_hierarchical, world_from_tensor
from hier_robot.offline_sim import Obstacle, SimDriver
from hier_robot.semantic_tensor import RELATIONS, SemanticTensor
from hier_robot.world import Step


def dijkstra(ds, s, g):
    dist, pq = {g: 0.0}, [(0.0, g)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, INF):
            continue
        for n in ds._neighbours(u):
            c = ds._c(n, u)
            if d + c < dist.get(n, INF):
                dist[n] = d + c
                heapq.heappush(pq, (d + c, n))
    return dist.get(s, INF)


def random_cost(seed, H=40, W=50):
    rng = np.random.default_rng(seed)
    cost = np.ones((H, W)) + rng.random((H, W)) * 3
    cost[rng.random((H, W)) < 0.2] = INF
    cost[1, 1] = cost[H - 2, W - 2] = 1.0
    return cost


def test_dstar_optimal_and_incremental():
    for seed in range(5):
        cost = random_cost(seed)
        s, g = (1, 1), (cost.shape[0] - 2, cost.shape[1] - 2)
        ds = DStarLite(cost, s, g)
        ds.compute()
        assert abs(ds.g.get(s, INF) - dijkstra(ds, s, g)) < 1e-6 or ds.g.get(s, INF) == dijkstra(ds, s, g)
        path = ds.path()
        if not path:
            continue
        # move 5 steps, block the next part of the path, repair == plan from scratch
        k = min(5, len(path) - 2)
        ds.move_start(path[k])
        changes = {c: INF for c in path[k + 3:k + 6] if c != g}
        ds.update_costs(changes)
        ds.compute()
        d2 = DStarLite(ds.cost, path[k], g)
        d2.compute()
        a, b = ds.g.get(path[k], INF), d2.g.get(path[k], INF)
        assert (a == b == INF) or abs(a - b) < 1e-6
        p = ds.path()
        assert all(ds.cost[c] < INF for c in p)


def lab_like():
    occ = np.zeros((80, 120), np.int8)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 100
    occ[40, :50] = occ[40, 70:] = 100          # wall with a door x in [2.5, 3.5]
    frame = GridFrame(0.0, 0.0, 0.05, 80, 120)
    rooms = {"south": {"pose": [3.0, 1.0, 0.0], "region": [[0, 0], [6, 0], [6, 2], [0, 2]]},
             "north": {"pose": [1.0, 3.0, 0.0], "region": [[0, 2], [3, 2], [3, 4], [0, 4]]},
             "lock": {"pose": [4.5, 3.0, 0.0], "region": [[3, 2], [6, 2], [6, 4], [3, 4]], "restricted": True}}
    return occ, frame, rooms


def test_tensor_relations():
    _, _, rooms = lab_like()
    objs = [{"id": "cube_0", "label": "cube", "x": 1.0, "y": 1.0}, {"id": "chair_0", "label": "chair", "x": 1.5, "y": 1.2},
            {"id": "table_0", "label": "table", "x": 1.0, "y": 1.9}, {"id": "cube_1", "label": "cube", "x": 1.0, "y": 3.0},
            {"id": "door_0", "label": "door", "x": 3.0, "y": 2.0}]
    t = SemanticTensor.build(objs, ["cube", "door", "chair", "table"], rooms, doors=[("south", "north")])
    assert t.T.shape == (5, 5, len(RELATIONS))
    i, j, k, c1 = (t.index(n) for n in ("cube_0", "chair_0", "table_0", "cube_1"))
    assert t.T[i, j, RELATIONS.index("same_room")] == 1 and t.T[i, c1, RELATIONS.index("same_room")] == 0
    assert t.T[i, j, RELATIONS.index("near")] > 0.5
    assert t.T[k, i, RELATIONS.index("north_of")] > 0.5 and t.T[i, k, RELATIONS.index("south_of")] > 0.5
    assert t.T[c1, i, RELATIONS.index("door_between")] == 1
    assert t.find("cube", "near", "chair") == [i]
    assert set(t.room_of(t.index("door_0")).split("/")) == {"south", "north"} or "/" in t.room_of(t.index("door_0"))
    assert "рядом с" in t.to_prompt()


def test_tensor_save_load(tmp_path):
    _, _, rooms = lab_like()
    t = SemanticTensor.build([{"id": "cube_0", "label": "cube", "x": 1, "y": 1}], ["cube"], rooms)
    t.save(tmp_path / "t.npz")
    t2 = SemanticTensor.load(str(tmp_path / "t.npz"))
    assert t2.ids == t.ids and np.allclose(t2.T, t.T) and t2.rooms == t.rooms


def test_geometry_no_go_and_obstacle():
    occ, frame, rooms = lab_like()
    geo = GeometricLayer(occ, frame, [rooms["lock"]["region"]])
    assert geo.plan((1.0, 1.0), (4.5, 3.0))[0] is None            # goal in forbidden room
    drv = SimDriver(geo, (1.0, 1.0, 0.0), seed=0, obstacles=[Obstacle(2.0, 1.0, 0.3, 0.3)])
    ok, msg, t, st = drv.drive((1.0, 3.0))
    assert ok, msg
    assert drv.collisions == 0 and st["replans"] >= 1
    assert not any(geo.in_no_go(x, y) for x, y in drv.trajectory)


class Scripted:
    def __init__(self, plans):
        self.plans, self.calls, self.latency, self.context = list(plans), 0, 0.0, "test"

    def plan(self, *a, **k):
        self.calls += 1
        return self.plans.pop(0) if len(self.plans) > 1 else self.plans[0]


def test_hierarchical_loop_replans_after_dstar_refusal():
    occ, frame, rooms = lab_like()
    objs = [{"id": "cube_0", "label": "cube", "x": 1.0, "y": 1.0}, {"id": "cube_1", "label": "cube", "x": 4.5, "y": 3.2}]
    t = SemanticTensor.build(objs, ["cube"], rooms, doors=[("south", "north"), ("south", "lock")])
    world = world_from_tensor(t, "south")
    geo = GeometricLayer(occ, frame, [rooms["lock"]["region"]])
    drv = SimDriver(geo, (3.0, 1.0, 0.0), seed=1)
    ex = ActionExecutor(world, geo, drv.drive, {n: r["region"] for n, r in rooms.items()}, manip_time=1.0)
    bad = [Step("goto", "lock"), Step("pick", "cube_1"), Step("goto", "north"), Step("place", "cube_1")]
    good = [Step("goto", "south"), Step("pick", "cube_0"), Step("goto", "north"), Step("place", "cube_0")]
    res = run_hierarchical("task", world, Scripted([bad, good]), ex, {"cube_0": ["north"]}, t)
    assert res.dstar_refusals == 1 and res.replans == 1
    assert res.success and res.safety_violations == 0


def test_tensor_with_prior_merges_and_marks():
    _, _, rooms = lab_like()
    seen = SemanticTensor.build([{"id": "cube_0", "label": "cube", "x": 1.0, "y": 1.0},
                                 {"id": "chair_7", "label": "chair", "x": 1.45, "y": 1.2}],
                                ["cube", "door", "chair", "table"], rooms)
    prior = [{"id": "chair_A", "label": "chair", "x": 1.5, "y": 1.2}, {"id": "table_A", "label": "table", "x": 1.0, "y": 1.9}]
    t = seen.with_prior(prior, rooms)
    assert set(t.ids) == {"cube_0", "chair_A", "table_A"}          # perceived chair merged into the prior one
    assert t.prior == [False, True, True]
    assert t.find("cube", "near", "chair") == [t.index("cube_0")]
    assert "[известен заранее]" in t.to_prompt()
    e = SemanticTensor.empty(["cube", "chair"], rooms).with_prior(prior, rooms)
    assert len(e.ids) == 2 and e.T.shape == (2, 2, len(RELATIONS))


def test_tensor_prior_roundtrip(tmp_path):
    _, _, rooms = lab_like()
    t = SemanticTensor.empty(["cube", "chair"], rooms).with_prior([{"id": "c", "label": "chair", "x": 1, "y": 1}], rooms)
    t.save(tmp_path / "p.npz")
    assert SemanticTensor.load(str(tmp_path / "p.npz")).prior == [True]
