"""HW3: semantic layer for SLAM, built from SigLIP2 + LiDAR (no ROS imports).

Geometric SLAM (slam_toolbox or Cartographer) gives the map and the robot pose.
This module adds *what* is on the map:

  1. The camera image is cut into vertical strips. SigLIP2 scores every strip
     against text prompts ("a photo of a chair", ...) — zero-shot, no training.
  2. A strip's centre column gives a bearing; the LiDAR range at that bearing
     gives the distance. Robot pose + bearing + range = the object's map position.
  3. Detections are associated into landmarks (same class, closer than assoc_dist);
     a landmark is confirmed after `min_observations` detections.
  4. A dependency graph is built: object -в-> room, object -рядом с-> object,
     room -соединена через-> door -> room.
  5. The map is exported as JSON and as a world.yaml for the HW1 planner, so the
     LLM plans with objects the robot found itself.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


# ============================================================================ config
@dataclass
class SemanticConfig:
    model: str = "google/siglip2-base-patch16-224"
    classes: Dict[str, List[str]] = field(default_factory=dict)
    background: List[str] = field(default_factory=list)
    threshold: float = 0.12
    margin: float = 0.03
    strips: int = 6
    min_observations: int = 3
    assoc_dist: float = 0.45
    near_dist: float = 0.9
    max_range: float = 3.5
    ratio: float = 1.0     # winner must score >= ratio x the second-best object class (1.0 = off)

    @classmethod
    def from_yaml(cls, path: str) -> "SemanticConfig":
        with open(path, encoding="utf-8") as f:
            c = yaml.safe_load(f)
        classes = {k: v["prompts"] for k, v in c["classes"].items() if not v.get("background")}
        bg = [k for k, v in c["classes"].items() if v.get("background")]
        prompts = {k: v["prompts"] for k, v in c["classes"].items()}
        cfg = cls(model=c.get("model", cls.model), classes=prompts, background=bg,
                  **{k: c[k] for k in ("threshold", "margin", "strips", "min_observations",
                                       "assoc_dist", "near_dist", "max_range", "ratio") if k in c})
        cfg.object_classes = list(classes)
        return cfg


# ============================================================================ SigLIP2
class SiglipScorer:
    """Zero-shot strip classifier on top of SigLIP2 (transformers >= 4.49).

    SigLIP is trained with a sigmoid loss, so every (image, text) score is an
    independent probability — several classes can be present in one strip,
    and "nothing" is simply all scores being low.
    """

    def __init__(self, cfg: SemanticConfig, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch, self.device = torch, device
        self.processor = AutoProcessor.from_pretrained(cfg.model)
        self.model = AutoModel.from_pretrained(cfg.model).to(device).eval()
        self.labels = list(cfg.classes)
        # SigLIP was trained on lowercase text; every prompt is scored separately
        self.texts, owner = [], []
        for i, label in enumerate(self.labels):
            for p in cfg.classes[label]:
                self.texts.append(p.lower())
                owner.append(i)
        self.owner = torch.tensor(owner)
        self.last_logits = None

    def score(self, crops) -> np.ndarray:
        """crops: list of PIL images -> [n_crops, n_labels] sigmoid scores (max over a label's prompts).

        Uses the model's own forward pass, exactly as in the official SigLIP2
        example: processor(text=..., images=..., padding="max_length",
        max_length=64) -> model(**inputs).logits_per_image -> sigmoid.
        """
        torch = self.torch
        with torch.no_grad():
            inputs = self.processor(text=self.texts, images=crops, padding="max_length", max_length=64,
                                    return_tensors="pt").to(self.device)
            out = self.model(**inputs)
            logits = out.logits_per_image            # [n_crops, n_texts]
            self.last_logits = logits
            probs = torch.sigmoid(logits)
        res = torch.zeros(len(crops), len(self.labels))
        for i in range(len(self.labels)):
            res[:, i] = probs[:, self.owner == i].max(dim=1).values
        return res.cpu().numpy()

    def diagnostics(self) -> str:
        """Numbers that show whether the model is wired up correctly."""
        m = self.model
        scale = float(m.logit_scale.detach().exp()) if hasattr(m, "logit_scale") else float("nan")
        bias = float(m.logit_bias.detach()) if hasattr(m, "logit_bias") else float("nan")
        lg = self.last_logits
        rng = f"{float(lg.min()):.1f} .. {float(lg.max()):.1f}" if lg is not None else "n/a"
        import transformers
        return (f"model {type(m).__name__}, transformers {transformers.__version__}, torch {self.torch.__version__}; "
                f"logit scale {scale:.1f}, bias {bias:.1f}; raw logits {rng}")


def strip_bounds(width: int, n: int) -> List[Tuple[int, int]]:
    """Column ranges of 2n-1 strips, each 1/n of the width, overlapping by half."""
    sw = width / n
    return [(int(k * sw / 2), int(min(width, k * sw / 2 + sw))) for k in range(2 * n - 1)]


def strip_windows(width: int, height: int, n: int, v_centre: float = 0.6) -> List[Tuple[int, int, int, int, float]]:
    """Square crop windows (x0, y0, x1, y1, centre_column), one per strip.

    SigLIP2 resizes every input to a 224x224 square. A tall, narrow strip
    (about 1:4.5) would be squashed sideways and filled with ceiling and floor,
    so the objects stop looking like themselves. Each window is therefore a
    square around the strip's centre column, at least half the image height,
    placed slightly below the image centre where floor objects usually are.
    """
    side = int(min(height, width, max(width / n, 0.5 * height)))
    cy = int(min(max(v_centre * height, side / 2), height - side / 2))
    out = []
    for x0, x1 in strip_bounds(width, n):
        cx = (x0 + x1) / 2
        wx0 = int(min(max(cx - side / 2, 0), width - side))
        out.append((wx0, cy - side // 2, wx0 + side, cy - side // 2 + side, cx))
    return out


def split_strips(image, n: int):
    """PIL image -> list of (square crop, centre_column), one per overlapping strip."""
    w, h = image.size
    return [(image.crop((x0, y0, x1, y1)), cx) for x0, y0, x1, y1, cx in strip_windows(w, h, n)]


def strip_labels(scores: np.ndarray, labels: List[str], cfg: SemanticConfig) -> List[Optional[Tuple[str, float]]]:
    """Per strip: (label, score) of the winning object class, or None."""
    bg_idx = [labels.index(b) for b in cfg.background if b in labels]
    obj_idx = [i for i, l in enumerate(labels) if l not in cfg.background]
    res = []
    for row in scores:
        ranked = sorted(obj_idx, key=lambda j: row[j], reverse=True)
        i = ranked[0]
        second = row[ranked[1]] if len(ranked) > 1 else 0.0
        bg = max((row[j] for j in bg_idx), default=0.0)
        ok = (row[i] >= cfg.threshold and row[i] >= bg + cfg.margin
              and row[i] >= getattr(cfg, "ratio", 1.0) * second)
        res.append((labels[i], float(row[i])) if ok else None)
    return res


# ============================================================================ geometry
def pixel_bearing(u: float, width: int, hfov: float, cam_yaw: float = 0.0) -> float:
    """Image column -> bearing in the robot frame (left = positive, ROS convention)."""
    fx = (width / 2) / math.tan(hfov / 2)
    return cam_yaw - math.atan((u - width / 2) / fx)


def lidar_range(ranges: Sequence[float], angle_min: float, angle_inc: float, bearing: float,
                half_width: float = math.radians(3), max_range: float = 3.5) -> Optional[float]:
    """Median valid LiDAR range in a small sector around `bearing` (robot frame)."""
    vals = []
    n = len(ranges)
    k0 = int(round((bearing - half_width - angle_min) / angle_inc))
    k1 = int(round((bearing + half_width - angle_min) / angle_inc))
    for k in range(k0, k1 + 1):
        r = ranges[k % n]
        if 0.05 < r < max_range and math.isfinite(r):
            vals.append(r)
    return float(np.median(vals)) if vals else None


# ============================================================================ map
@dataclass
class Landmark:
    id: str
    label: str
    x: float
    y: float
    n: int = 1
    score: float = 0.0
    feat: Optional[list] = None   # running mean of the SigLIP2 score vector over all classes (for the tensor)

    def update(self, x: float, y: float, s: float, feat=None):
        self.n += 1
        a = 1.0 / self.n
        self.x += a * (x - self.x)
        self.y += a * (y - self.y)
        self.score += a * (s - self.score)
        if feat is not None:
            f = np.asarray(feat, float)
            self.feat = (f if self.feat is None else np.asarray(self.feat) + a * (f - np.asarray(self.feat))).tolist()


class SemanticMap:
    def __init__(self, cfg: SemanticConfig, rooms: Dict[str, dict]):
        """rooms: name -> {"pose": [x, y, yaw], "region": [[x, y], ...] (optional), "restricted": bool}"""
        self.cfg, self.rooms = cfg, rooms
        self.landmarks: List[Landmark] = []
        self._count: Dict[str, int] = {}
        self.observations = 0

    # ---- integration of one frame ---------------------------------------
    def add_detection(self, label: str, x: float, y: float, score: float, feat=None) -> Landmark:
        x, y, score = float(x), float(y), float(score)
        self.observations += 1
        best, best_d = None, self.cfg.assoc_dist
        for lm in self.landmarks:
            if lm.label == label:
                d = math.hypot(lm.x - x, lm.y - y)
                if d < best_d:
                    best, best_d = lm, d
        if best is not None:
            best.update(x, y, score, feat)
            return best
        k = self._count.get(label, 0)
        self._count[label] = k + 1
        lm = Landmark(f"{label}_{k}", label, x, y, 1, score,
                      None if feat is None else np.asarray(feat, float).tolist())
        self.landmarks.append(lm)
        return lm

    def integrate(self, pose: Tuple[float, float, float], strips: List[Optional[Tuple[str, float]]],
                  bearings: List[float], range_at, feats=None) -> int:
        """pose = robot (x, y, yaw) in map; range_at(bearing) -> metres or None.
        feats (optional): per-strip SigLIP2 score vectors, averaged into Landmark.feat."""
        x, y, th = pose
        added = 0
        feats = list(feats) if feats is not None else [None] * len(strips)
        for det, b, f in zip(strips, bearings, feats):
            if det is None:
                continue
            r = range_at(b)
            if r is None or r > self.cfg.max_range:
                continue
            self.add_detection(det[0], x + r * math.cos(th + b), y + r * math.sin(th + b), det[1], f)
            added += 1
        return added

    def confirmed(self) -> List[Landmark]:
        return [l for l in self.landmarks if l.n >= self.cfg.min_observations]

    # ---- rooms and graph -------------------------------------------------
    def room_of(self, x: float, y: float) -> str:
        for name, r in self.rooms.items():
            reg = r.get("region")
            if reg and point_in_polygon(x, y, reg):
                return name
        return min(self.rooms, key=lambda n: math.hypot(self.rooms[n]["pose"][0] - x, self.rooms[n]["pose"][1] - y))

    def graph(self) -> dict:
        objs = self.confirmed()
        nodes = [{"id": n, "type": "room", "restricted": bool(r.get("restricted", False))}
                 for n, r in self.rooms.items()]
        edges = []
        for o in objs:
            room = self.room_of(o.x, o.y)
            nodes.append({"id": o.id, "type": "object", "label": o.label, "x": round(o.x, 2),
                          "y": round(o.y, 2), "room": room, "observations": o.n,
                          "score": round(o.score, 3)})
            edges.append({"from": o.id, "to": room, "rel": "в"})
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                d = math.hypot(a.x - b.x, a.y - b.y)
                if d < self.cfg.near_dist:
                    edges.append({"from": a.id, "to": b.id, "rel": "рядом с", "dist": round(d, 2)})
        for o in objs:
            if o.label != "door":
                continue
            near = sorted(self.rooms, key=lambda n: math.hypot(self.rooms[n]["pose"][0] - o.x,
                                                                self.rooms[n]["pose"][1] - o.y))[:2]
            if len(near) == 2:
                edges.append({"from": near[0], "to": near[1], "rel": "соединена через", "via": o.id})
        return {"nodes": nodes, "edges": edges}

    def describe(self) -> str:
        """Short text for the LLM planner (HW1 prompt)."""
        g = self.graph()
        by_room: Dict[str, List[str]] = {}
        for n in g["nodes"]:
            if n["type"] == "object":
                by_room.setdefault(n["room"], []).append(n["id"])
        lines = [f"{r}: {', '.join(v)}" for r, v in sorted(by_room.items())]
        near = [f"{e['from']} рядом с {e['to']}" for e in g["edges"] if e["rel"] == "рядом с"]
        return "Найденные объекты:\n" + "\n".join(lines) + ("\nСвязи: " + "; ".join(near) if near else "")

    # ---- export ----------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps({"landmarks": [l.__dict__ for l in self.landmarks], "graph": self.graph()},
                          ensure_ascii=False, indent=2)

    def to_world_yaml(self, start_room: str, speed: float = 0.18) -> str:
        """world.yaml for the HW1 planner with the cubes the robot actually found."""
        rooms = {n: {k: v for k, v in r.items() if k in ("pose", "restricted", "description", "region")}
                 for n, r in self.rooms.items()}
        objects = {}
        for o in self.confirmed():
            if o.label == "cube":
                room = self.room_of(o.x, o.y)
                objects[o.id] = {"room": room, "pose": approach_pose(o.x, o.y, self.rooms[room]["pose"])}
        return yaml.safe_dump({"start_room": start_room, "speed": speed, "rooms": rooms, "objects": objects},
                              allow_unicode=True, sort_keys=False)

    def to_dot(self) -> str:
        g = self.graph()
        lines = ["digraph semantic {", "  rankdir=LR;"]
        for n in g["nodes"]:
            shape = "box" if n["type"] == "room" else "ellipse"
            color = ', color="red"' if n.get("restricted") else ""
            lines.append(f'  "{n["id"]}" [shape={shape}{color}];')
        for e in g["edges"]:
            lines.append(f'  "{e["from"]}" -> "{e["to"]}" [label="{e["rel"]}"];')
        lines.append("}")
        return "\n".join(lines)


def approach_pose(x: float, y: float, room_pose, stand_off: float = 0.35):
    """Pose `stand_off` metres before the object, facing it, coming from the room centre."""
    ang = math.atan2(y - room_pose[1], x - room_pose[0])
    return [round(x - stand_off * math.cos(ang), 2), round(y - stand_off * math.sin(ang), 2), round(ang, 2)]


def point_in_polygon(x: float, y: float, poly) -> bool:
    inside, n = False, len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside
