"""Tensor representation of the semantic scene (SLAM + SigLIP2).

The SLAM system (RTAB-Map) gives the robot pose and the occupancy grid; SigLIP2
says *what* the camera sees; the LiDAR says *how far*. The semantic mapper turns
that into landmarks (SemanticMap). This module packs those landmarks into tensors
that the top-level planner (LLM) reads and the middle level (D*) uses for goals:

    X  [N, C]     object features: mean SigLIP2 score per class over all sightings
                  (soft class distribution, not only the winning label)
    P  [N, 2]     object positions in the SLAM map frame, metres
    T  [N, N, K]  relation tensor; T[i, j, k] in [0, 1] = strength of relation k from i to j
    M  [N, R]     object-in-room membership
    A  [R, R]     room connectivity (through doors)
    r  [R]        1 = restricted room (no-go zone for D*)
    E  [N, D]     optional SigLIP2 image embeddings (mean), if the mapper recorded them

Relations K:
    same_room     i and j are in the same room                             (symmetric)
    near          1 - d/near_dist when d < near_dist                        (symmetric)
    north_of      i is north of j (+y in map) in the same room, graded       (antisymmetric pair)
    south_of
    east_of       i is east of j (+x)
    west_of
    door_between  i and j are in different rooms joined by a door             (symmetric)

The planner gets the tensor serialised as a compact list (objects + strongest
relations), so a 3B model can answer "the cube next to the pallet" or "the room
with the table" without seeing pixels. Saved as .npz for analysis.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

RELATIONS = ["same_room", "near", "north_of", "south_of", "east_of", "west_of", "door_between"]
REL_RU = {"same_room": "в одной комнате с", "near": "рядом с", "north_of": "севернее",
          "south_of": "южнее", "east_of": "восточнее", "west_of": "западнее",
          "door_between": "через дверь от"}


def _room_of(x: float, y: float, rooms: Dict[str, dict]) -> str:
    for name, r in rooms.items():
        reg = r.get("region")
        if reg and _in_poly(x, y, reg):
            return name
    return min(rooms, key=lambda n: math.hypot(rooms[n]["pose"][0] - x, rooms[n]["pose"][1] - y))


def _door_rooms(x: float, y: float, rooms: Dict[str, dict], probe: float = 0.35) -> List[str]:
    """The two rooms on either side of a door: probe the room regions around the door point;
    without regions fall back to the two closest room centres."""
    found = []
    for dx, dy in ((probe, 0), (-probe, 0), (0, probe), (0, -probe)):
        n = _room_of(x + dx, y + dy, rooms)
        if n not in found:
            found.append(n)
    if len(found) == 2:
        return found
    return sorted(rooms, key=lambda n: math.hypot(rooms[n]["pose"][0] - x, rooms[n]["pose"][1] - y))[:2]


def _in_poly(x, y, poly) -> bool:
    inside, n = False, len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


@dataclass
class SemanticTensor:
    ids: List[str]
    labels: List[str]            # winning class per object
    classes: List[str]
    rooms: List[str]
    X: np.ndarray
    P: np.ndarray
    T: np.ndarray
    M: np.ndarray
    A: np.ndarray
    r: np.ndarray
    E: Optional[np.ndarray] = None
    obs: Optional[np.ndarray] = None          # number of sightings per object
    room_poses: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    room_desc: Dict[str, str] = field(default_factory=dict)
    prior: Optional[List[bool]] = None        # True = object given as prior knowledge, not seen by SigLIP2

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, objects: Sequence[dict], classes: Sequence[str], rooms: Dict[str, dict],
              doors: Optional[Sequence[Tuple[str, str]]] = None, near_dist: float = 1.5,
              dir_min: float = 0.3) -> "SemanticTensor":
        """objects: [{"id", "label", "x", "y", "feat" (optional, len C), "n" (optional)}]
        rooms: name -> {"pose", "region"?, "restricted"?, "description"?}
        doors: room pairs joined by a door; if None they are taken from door landmarks."""
        classes = list(classes)
        room_names = list(rooms)
        N, C, R, K = len(objects), len(classes), len(room_names), len(RELATIONS)
        X = np.zeros((N, C))
        P = np.zeros((N, 2))
        M = np.zeros((N, R))
        obs = np.zeros(N)
        for i, o in enumerate(objects):
            P[i] = (o["x"], o["y"])
            f = o.get("feat")
            if f is not None and len(f) == C:
                X[i] = np.asarray(f, float)
            elif o["label"] in classes:
                X[i, classes.index(o["label"])] = float(o.get("score", 1.0)) or 1.0
            s = X[i].sum()
            if s > 0:
                X[i] /= s                         # soft class distribution
            if o["label"] == "door":   # a door lies on the boundary: half in each of the two rooms it joins
                for n in _door_rooms(o["x"], o["y"], rooms):
                    M[i, room_names.index(n)] = 0.5
            else:
                M[i, room_names.index(_room_of(o["x"], o["y"], rooms))] = 1.0
            obs[i] = o.get("n", 1)

        A = np.zeros((R, R))
        if doors is None:   # doors seen by the robot: a door joins the two closest rooms
            doors = []
            for o in objects:
                if o["label"] == "door":
                    near = _door_rooms(o["x"], o["y"], rooms)
                    if len(near) == 2:
                        doors.append(tuple(near))
        for a, b in doors:
            if a in room_names and b in room_names:
                ia, ib = room_names.index(a), room_names.index(b)
                A[ia, ib] = A[ib, ia] = 1.0
        rvec = np.array([1.0 if rooms[n].get("restricted") else 0.0 for n in room_names])

        T = np.zeros((N, N, K))
        room_idx = M.argmax(1) if N else np.zeros(0, int)
        is_door = [o["label"] == "door" for o in objects]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dx, dy = P[i] - P[j]
                d = math.hypot(dx, dy)
                if d < near_dist:
                    T[i, j, 1] = 1.0 - d / near_dist
                if is_door[i] or is_door[j]:
                    continue           # doors: only "near"; their rooms are already in M and A
                same = room_idx[i] == room_idx[j]
                T[i, j, 0] = float(same)
                if same and d < 3.0:   # directions only inside one room, where they are meaningful
                    g = lambda v: float(np.clip((v - dir_min) / 1.0, 0.0, 1.0)) if v > dir_min else 0.0  # noqa: E731
                    T[i, j, 2], T[i, j, 3] = g(dy), g(-dy)
                    T[i, j, 4], T[i, j, 5] = g(dx), g(-dx)
                if not same and A[room_idx[i], room_idx[j]] > 0:
                    T[i, j, 6] = 1.0
        return cls(ids=[o["id"] for o in objects], labels=[o["label"] for o in objects], classes=classes,
                   rooms=room_names, X=X, P=P, T=T, M=M, A=A, r=rvec, obs=obs,
                   room_poses={n: tuple(rooms[n]["pose"]) for n in room_names},
                   room_desc={n: rooms[n].get("description", "") for n in room_names})

    @classmethod
    def from_semantic_map(cls, smap, doors=None, near_dist: float = 1.5) -> "SemanticTensor":
        """From a live SemanticMap (confirmed landmarks only)."""
        cfg = smap.cfg
        classes = getattr(cfg, "object_classes", None) or [c for c in cfg.classes if c not in cfg.background]
        objs = [{"id": l.id, "label": l.label, "x": l.x, "y": l.y, "feat": _object_feat(l.feat, smap.cfg),
                 "n": l.n, "score": l.score} for l in smap.confirmed()]
        return cls.build(objs, classes, smap.rooms, doors=doors, near_dist=near_dist)

    def with_prior(self, prior_objects: Sequence[dict], rooms: Dict[str, dict],
                   doors: Optional[Sequence[Tuple[str, str]]] = None, dedup_dist: float = 0.6) -> "SemanticTensor":
        """Merge prior knowledge (e.g. the known furniture of the building) into the perceived tensor.
        Perceived objects keep their SigLIP2 features; a perceived object of the same class within
        dedup_dist of a prior one is treated as the same object (the prior position is kept).
        prior_objects: [{"id", "label", "x", "y"}]. Prior objects are marked in the prompt."""
        objs, flags = [], []
        known = [o for o in prior_objects]
        for i, oid in enumerate(self.ids):
            x, y = float(self.P[i, 0]), float(self.P[i, 1])
            if any(p["label"] == self.labels[i] and math.hypot(p["x"] - x, p["y"] - y) < dedup_dist for p in known):
                continue
            objs.append({"id": oid, "label": self.labels[i], "x": x, "y": y, "feat": self.X[i].tolist(),
                         "n": int(self.obs[i]) if self.obs is not None else 1})
            flags.append(bool(self.prior[i]) if self.prior else False)
        taken = {o["id"] for o in objs}
        for p in known:
            pid = p["id"] if p["id"] not in taken else p["id"] + "_prior"
            objs.append({"id": pid, "label": p["label"], "x": float(p["x"]), "y": float(p["y"]), "n": 0})
            flags.append(True)
        t = SemanticTensor.build(objs, self.classes, rooms, doors=doors)
        t.prior = flags
        return t

    @classmethod
    def empty(cls, classes: Sequence[str], rooms: Dict[str, dict], doors=None) -> "SemanticTensor":
        return cls.build([], classes, rooms, doors=doors)

    # ------------------------------------------------------------------ io
    def save(self, path: str):
        np.savez_compressed(path, X=self.X, P=self.P, T=self.T, M=self.M, A=self.A, r=self.r,
                            obs=self.obs if self.obs is not None else np.zeros(len(self.ids)),
                            E=self.E if self.E is not None else np.zeros((0, 0)),
                            meta=json.dumps({"ids": self.ids, "labels": self.labels, "classes": self.classes,
                                             "rooms": self.rooms, "relations": RELATIONS,
                                             "room_poses": self.room_poses, "room_desc": self.room_desc,
                                             "prior": self.prior},
                                            ensure_ascii=False))

    @classmethod
    def load(cls, path: str) -> "SemanticTensor":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        E = z["E"] if z["E"].size else None
        return cls(ids=meta["ids"], labels=meta["labels"], classes=meta["classes"], rooms=meta["rooms"],
                   X=z["X"], P=z["P"], T=z["T"], M=z["M"], A=z["A"], r=z["r"], E=E, obs=z["obs"],
                   room_poses={k: tuple(v) for k, v in meta.get("room_poses", {}).items()},
                   room_desc=meta.get("room_desc", {}), prior=meta.get("prior"))

    # ------------------------------------------------------------------ queries
    def room_of(self, i: int) -> str:
        if self.labels[i] == "door":
            return "/".join(self.rooms[k] for k in np.where(self.M[i] > 0)[0])
        return self.rooms[int(self.M[i].argmax())]

    def index(self, obj_id: str) -> int:
        return self.ids.index(obj_id)

    def related(self, i: int, rel: str, thr: float = 0.25) -> List[Tuple[int, float]]:
        k = RELATIONS.index(rel)
        js = np.where(self.T[i, :, k] > thr)[0]
        return sorted(((int(j), float(self.T[i, j, k])) for j in js), key=lambda t: -t[1])

    def find(self, label: str, rel: Optional[str] = None, other_label: Optional[str] = None,
             thr: float = 0.25) -> List[int]:
        """Objects of class `label` that have relation `rel` to some object of class `other_label`."""
        out = []
        for i, l in enumerate(self.labels):
            if l != label:
                continue
            if rel is None or any(self.labels[j] == other_label for j, _ in self.related(i, rel, thr)):
                out.append(i)
        return out

    def restricted_rooms(self) -> List[str]:
        return [n for n, v in zip(self.rooms, self.r) if v > 0]

    # ------------------------------------------------------------------ for the LLM
    def to_prompt(self, max_rel: int = 40, rel_thr: float = 0.2, with_relations: bool = True) -> str:
        """Compact text form of the tensor: objects, then the strongest relations as triples."""
        lines = ["Семантическая карта (SLAM + SigLIP2), объекты:"]
        for i, oid in enumerate(self.ids):
            top = np.argsort(-self.X[i])[:2]
            dist = ", ".join(f"{self.classes[c]} {self.X[i, c]:.2f}" for c in top if self.X[i, c] > 0)
            src = " [известен заранее]" if self.prior and self.prior[i] else ""
            lines.append(f"  [{i}] {oid}: класс {self.labels[i]} ({dist}), комната {self.room_of(i)}, "
                         f"x={self.P[i, 0]:.1f} y={self.P[i, 1]:.1f}{src}")
        doors = [f"{self.rooms[a]}—{self.rooms[b]}" for a in range(len(self.rooms))
                 for b in range(a + 1, len(self.rooms)) if self.A[a, b] > 0]
        if doors:
            lines.append("Двери между комнатами: " + ", ".join(doors))
        if not with_relations:
            return "\n".join(lines)
        triples = []
        for k, rel in enumerate(RELATIONS):
            if rel in ("same_room", "door_between", "south_of", "west_of"):
                continue   # implied by rooms / doors, or the mirror of north_of / east_of
            for i in range(len(self.ids)):
                for j in range(len(self.ids)):
                    v = self.T[i, j, k]
                    if v > rel_thr and (rel != "near" or i < j):
                        triples.append((v, f"{self.ids[i]} {REL_RU[rel]} {self.ids[j]} ({v:.2f})"))
        triples.sort(key=lambda t: -t[0])
        if triples:
            lines.append("Отношения (тензор T, сила 0..1):")
            lines += [f"  {t}" for _, t in triples[:max_rel]]
        return "\n".join(lines)

    def summary(self) -> str:
        K = len(RELATIONS)
        nz = {rel: int((self.T[:, :, k] > 0).sum()) for k, rel in enumerate(RELATIONS)}
        return (f"N={len(self.ids)} объектов, C={len(self.classes)} классов, R={len(self.rooms)} комнат, "
                f"K={K} отношений; T{tuple(self.T.shape)} ненулевых: {nz}")


def _object_feat(feat, cfg):
    """Mapper stores scores for all labels incl. background; keep the object classes only."""
    if feat is None:
        return None
    labels = list(cfg.classes)              # the scorer's label order (object classes + background)
    keep = [k for k, c in enumerate(labels) if c not in cfg.background]
    f = np.asarray(feat, float)
    return f[keep].tolist() if len(f) == len(labels) else None


# ------------------------------------------------------------------------ world adapter
def world_dict_from_tensor(st: SemanticTensor, start_room: str, speed: float = 0.18,
                           pickable: Sequence[str] = ("cube",)) -> dict:
    """world.yaml-style dict for the planner: rooms + the pickable objects the robot found,
    each with an approach pose computed from the tensor position."""
    rooms = {n: {"pose": list(st.room_poses[n]), "restricted": bool(st.r[k] > 0),
                 "description": st.room_desc.get(n, "")} for k, n in enumerate(st.rooms)}
    objects = {}
    for i, oid in enumerate(st.ids):
        if st.labels[i] in pickable:
            room = st.room_of(i)
            rp = st.room_poses[room]
            ang = math.atan2(st.P[i, 1] - rp[1], st.P[i, 0] - rp[0])
            objects[oid] = {"room": room, "pose": [round(st.P[i, 0] - 0.35 * math.cos(ang), 2),
                                                   round(st.P[i, 1] - 0.35 * math.sin(ang), 2), round(ang, 2)]}
    return {"start_room": start_room, "speed": speed, "rooms": rooms, "objects": objects}
