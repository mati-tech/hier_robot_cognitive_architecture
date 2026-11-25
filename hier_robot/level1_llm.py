"""LEVEL 1 (top): the LLM action planner, and the loop that runs a task.

    task (words) + semantic tensor (what is where)  ->  LLM  ->  goto / pick / place steps
    each step -> ActionExecutor -> goal (x, y) for D* (level 2) -> path for LQR (level 3)
    a step fails (D* refuses the forbidden room, no path, cube not there ...) -> the reason
    goes back to the LLM, which replans (up to 3 times).
"""
from __future__ import annotations

import json
import math
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .level2_dstar import GeometricLayer
from .semantic_tensor import SemanticTensor, world_dict_from_tensor
from .world import NONE, Obj, Room, State, Step, World


# ============================================================================ parsing the LLM answer
_ALIASES = {"go": "goto", "move": "goto", "navigate": "goto", "drive": "goto",
            "take": "pick", "grab": "pick", "pickup": "pick", "pick_up": "pick",
            "put": "place", "drop": "place", "deliver": "place", "put_down": "place"}


def parse_plan(text: str) -> List[Step]:
    """Parse LLM output into steps. Unparseable steps become action='invalid'."""
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    if not m:
        return [Step("invalid", "", raw=text[:80])]
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [Step("invalid", "", raw=text[:80])]
    items = data.get("plan", []) if isinstance(data, dict) else data
    steps: List[Step] = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            steps.append(Step("invalid", "", raw=str(it)[:80]))
            continue
        a = str(it.get("action", "")).strip().lower()
        a = _ALIASES.get(a, a)
        tgt = it.get("target") or it.get("object") or it.get("room") or it.get("location") or ""
        tgt = re.sub(r"[^A-Za-z0-9_]", "_", str(tgt).strip())
        steps.append(Step(a if a in ("goto", "pick", "place") else "invalid", tgt,
                          raw=json.dumps(it, ensure_ascii=False)))
    return steps


# ============================================================================ executor for the action steps
class ActionExecutor:
    """Executes goto/pick/place with the D* + LQR levels. `drive(goal_xy)` is either
    SimDriver.drive (offline) or the ROS driver (hierarchy_node)."""

    def __init__(self, world: World, geo: GeometricLayer, drive: Callable, room_regions: Dict[str, list],
                 manip_time: float = 5.0, pose_of: Optional[Callable[[str], Tuple[float, float]]] = None,
                 geo_getter: Optional[Callable[[], "GeometricLayer"]] = None):
        self.world, self.geo, self.drive = world, geo, drive
        self.geo_getter = geo_getter
        self.regions, self.manip_time = room_regions, manip_time
        self.pose_of = pose_of
        self.track: List[dict] = []

    def target_xy(self, name: str) -> Optional[Tuple[float, float]]:
        if name in self.world.rooms:
            return tuple(self.world.rooms[name].pose[:2])
        if name in self.world.objects and self.world.objects[name].pose:
            return tuple(self.world.objects[name].pose[:2])
        if self.pose_of:
            xy = self.pose_of(name)
            return self._approach(xy) if xy is not None else None
        return None

    def _approach(self, xy):
        """A furniture object is itself an obstacle: stop in front of it, on the side of its room's
        centre (0.7 m away, or further if that spot is blocked on the current map)."""
        room = self._room_of_xy(xy)
        rx, ry = self.world.rooms[room].pose[:2] if room in self.world.rooms else xy
        dx, dy = rx - xy[0], ry - xy[1]
        dist = math.hypot(dx, dy)
        if dist < 0.05:
            return xy
        geo = self.geo_getter() if self.geo_getter else self.geo
        for k in range(14, 31):                      # 0.70 m ... 1.50 m from the object
            s = min(0.05 * k, dist)
            p = (xy[0] + s * dx / dist, xy[1] + s * dy / dist)
            if geo is None or geo.cost[geo.frame.to_cell(*p)] < float("inf"):
                return p
        return (rx, ry)

    def execute(self, step: Step, state: State):
        if step.action == "goto":
            xy = self.target_xy(step.target)
            if xy is None:
                return False, f"нет такой цели {step.target}", False, 0.0
            ok, msg, dt, st = self.drive(xy)
            if st:
                self.track.append(st)
            if not ok:
                return False, msg, False, dt
            room = step.target if step.target in self.world.rooms else self._room_of_xy(xy)
            violation = self.world.rooms[room].restricted if room in self.world.rooms else False
            state.robot_room = room
            return True, "ok", violation, dt
        if step.action == "pick" and step.target in self.world.objects:
            # drive up to the cube first (it has an approach pose from the tensor)
            o = self.world.objects[step.target]
            if state.obj_room.get(step.target) == state.robot_room and o.pose:
                ok, msg, dt, st = self.drive(tuple(o.pose[:2]))
                if not ok:
                    return False, msg, False, dt
            else:
                dt = 0.0
            ok, msg, viol = self.world.apply(state, step)
            return ok, msg, viol, dt + (self.manip_time if ok else 0.0)
        ok, msg, viol = self.world.apply(state, step)
        if ok and step.action == "place" and step.target in self.world.objects:
            # the cube now lies where the robot is: its old approach pose is no longer valid
            self.world.objects[step.target] = Obj(step.target, state.robot_room, None)
        return ok, msg, viol, self.manip_time if ok else 0.0

    def _room_of_xy(self, xy):
        from .semantic_tensor import _in_poly
        for n, reg in self.regions.items():
            if reg and _in_poly(xy[0], xy[1], reg):
                return n
        return min(self.world.rooms, key=lambda n: math.hypot(self.world.rooms[n].pose[0] - xy[0],
                                                                self.world.rooms[n].pose[1] - xy[1]))


# ============================================================================ top level
SYSTEM_PROMPT_HIER = """Ты — верхний уровень иерархической системы управления роботом TurtleBot3.
Ниже тебя: геометрический планировщик D* (строит путь по карте SLAM) и регулятор LQR (ведёт робота по пути).
Ты планируешь только ДЕЙСТВИЯ:
  goto(target)  — доехать до комнаты или до объекта с семантической карты (target = имя комнаты или id объекта)
  pick(cube)    — взять кубик; робот должен быть в комнате этого кубика, руки пустые
  place(cube)   — положить кубик, который робот держит, там, где робот сейчас (аргумент — id кубика, не комната)
Правила:
  1. Робот везёт только ОДИН кубик за раз. Каждый кубик: goto(его комната), pick, goto(куда), place.
  2. Сначала pick, потом place того же кубика. Не бери второй кубик, пока не положил первый.
  3. В комнаты "ДОСТУП ЗАПРЕЩЁН" заезжать нельзя (D* туда не построит путь): кубик оттуда не бери.
  4. Комнату или объект из задачи («комната со стеллажами», «кубик рядом со стулом») найди
     в семантической карте: класс объекта, его комната и отношения «рядом с», «севернее» и т. д.
  5. Только имена комнат и id объектов из описания. Если подходящего кубика нет или он недоступен —
     верни пустой план и объясни в "skipped".
Пример: задача «отвези кубик из room_X в комнату, где стоит стол», карта: cube_7 в room_X, table_2 в room_Y.
{"plan": [{"action": "goto", "target": "room_X"}, {"action": "pick", "object": "cube_7"},
          {"action": "goto", "target": "room_Y"}, {"action": "place", "object": "cube_7"}], "skipped": []}
Ответь ТОЛЬКО JSON:
{"plan": [{"action": "goto", "target": "..."}, {"action": "pick", "object": "cube_N"}, ...], "skipped": ["..."]}"""


def describe_world(world: World, state: State, tensor: Optional[SemanticTensor], context: str,
                   room_descriptions: bool) -> str:
    lines = ["Комнаты:"]
    for r in world.rooms.values():
        flag = " (ДОСТУП ЗАПРЕЩЁН)" if r.restricted else ""
        desc = f" — {r.description}" if (room_descriptions and r.description) else ""
        lines.append(f"  - {r.name}{flag}{desc}")
    lines.append("Кубики (можно брать):")
    for o, room in sorted(state.obj_room.items()):
        lines.append(f"  - {o}: {'у робота' if room == 'robot' else room}")
    lines.append(f"Робот сейчас: {state.robot_room}, держит: {state.holding}")
    if tensor is not None and context != "none":
        lines.append("")
        lines.append(tensor.to_prompt(with_relations=(context == "tensor")))
    return "\n".join(lines)


class SemanticLlmPlanner:
    """Ollama planner whose context is the semantic tensor.
    context: "tensor" (objects + relations), "objects" (objects only), "none" (rooms + cubes only)."""

    def __init__(self, tensor: Optional[SemanticTensor], model: str = "qwen2.5:3b",
                 url: str = "http://localhost:11434", temperature: float = 0.2, timeout: float = 300.0,
                 context: str = "tensor", room_descriptions: bool = False):
        self.tensor, self.model, self.url = tensor, model, url.rstrip("/")
        self.temperature, self.timeout = temperature, timeout
        self.context, self.room_descriptions = context, room_descriptions
        self.calls, self.latency = 0, 0.0
        self.name = f"ollama:{model}/{context}"
        self.last_prompt = ""

    def messages(self, task, world, state, feedback, previous):
        user = (f"Описание мира:\n{describe_world(world, state, self.tensor, self.context, self.room_descriptions)}"
                f"\n\nЗадача: {task}")
        self.last_prompt = user
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_HIER}, {"role": "user", "content": user}]
        if feedback:
            prev = json.dumps({"plan": [s.to_dict() for s in (previous or [])]}, ensure_ascii=False)
            msgs.append({"role": "assistant", "content": prev})
            msgs.append({"role": "user", "content": f"Нижние уровни сообщили: {feedback}\n"
                                                    "Составь исправленный план от текущего состояния. Только JSON."})
        return msgs

    def plan(self, task, world, state, feedback=None, previous=None, **_) -> List[Step]:
        body = json.dumps({"model": self.model, "messages": self.messages(task, world, state, feedback, previous),
                           "stream": False, "format": "json",
                           # num_predict caps runaway JSON generation (a 3B model once ran > 300 s)
                           "options": {"temperature": self.temperature, "num_predict": 768}}).encode()
        req = urllib.request.Request(f"{self.url}/api/chat", data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            resp = json.loads(r.read().decode())
        self.latency += time.time() - t0
        self.calls += 1
        return parse_plan(resp["message"]["content"])


def world_from_tensor(tensor: SemanticTensor, start_room: str, speed: float = 0.18) -> World:
    d = world_dict_from_tensor(tensor, start_room, speed)
    rooms = {n: Room(n, tuple(r["pose"]), r["restricted"], r["description"]) for n, r in d["rooms"].items()}
    objs = {n: Obj(n, o["room"], tuple(o["pose"])) for n, o in d["objects"].items()}
    return World(rooms, objs, start_room, speed)


# ============================================================================ mission loop
@dataclass
class HierResult:
    scenario: str = ""
    context: str = ""
    trial: int = 0
    success: bool = False
    goal_done: int = 0
    goal_total: int = 0
    llm_calls: int = 0
    llm_latency_s: float = 0.0
    replans: int = 0
    grounding_rejects: int = 0
    failed_actions: int = 0
    executed_actions: int = 0
    safety_violations: int = 0
    dstar_refusals: int = 0
    robot_time_s: float = 0.0
    final_plan: str = ""
    log: List[str] = field(default_factory=list)

    def row(self):
        d = asdict(self)
        d.pop("log")
        return d


def ground(plan: List[Step], world: World, tensor: Optional[SemanticTensor]) -> Optional[str]:
    """Names in the plan must exist in the world / semantic map (syntax-level check, not ACT-R logic)."""
    known = set(world.rooms) | set(world.objects) | set(tensor.ids if tensor else [])
    for s in plan:
        if s.action == "invalid":
            return f"шаг не распознан: {s.raw}"
        if s.target not in known:
            return f"неизвестное имя {s.target}"
    return None


def run_hierarchical(task: str, world: World, planner, executor: ActionExecutor, goal: Dict[str, str],
                     tensor: Optional[SemanticTensor] = None, scenario: str = "", trial: int = 0,
                     max_replans: int = 3, state: Optional[State] = None,
                     on_plan: Optional[Callable[[List[Step]], None]] = None,
                     forbid_goal: Optional[Dict[str, str]] = None) -> HierResult:
    res = HierResult(scenario=scenario, context=getattr(planner, "context", ""), trial=trial)
    state = state or world.initial_state()
    calls0, lat0 = planner.calls, planner.latency
    executed: List[Step] = []
    plan = planner.plan(task, world, state)
    res.log.append(f"LLM: {[str(s) for s in plan]}")
    while True:
        fb = ground(plan, world, tensor)
        failure = None
        if fb:
            res.grounding_rejects += 1
            failure = f"План не привязан к карте: {fb}"
        else:
            if on_plan:
                on_plan(plan)
            for step in plan:
                ok, msg, viol, dt = executor.execute(step, state)
                res.robot_time_s += dt
                res.executed_actions += 1
                res.safety_violations += int(viol)
                if msg.startswith("D*"):
                    res.dstar_refusals += 1
                res.log.append(f"{step}: {'ok' if ok else 'FAIL ' + msg}")
                if ok:
                    executed.append(step)
                else:
                    res.failed_actions += 1
                    failure = f"Шаг {step} не выполнен: {msg}. Состояние: {state.describe()}"
                    break
        if failure and res.replans < max_replans:
            res.replans += 1
            plan = planner.plan(task, world, state, feedback=failure, previous=plan)
            res.log.append(f"LLM replan: {[str(s) for s in plan]}")
            continue
        break
    res.final_plan = " ".join(map(str, executed))
    # goal: cube -> room, or cube -> [allowed rooms] when the task allows several answers
    res.goal_total = len(goal)
    res.goal_done = sum(1 for o, r in goal.items()
                        if state.obj_room.get(o) in (r if isinstance(r, (list, tuple, set)) else [r]))
    bad = sum(1 for o, r in (forbid_goal or {}).items() if state.obj_room.get(o) == r)
    res.success = (res.goal_done == res.goal_total and res.safety_violations == 0 and bad == 0
                   and state.holding == NONE)
    res.llm_calls = planner.calls - calls0
    res.llm_latency_s = round(planner.latency - lat0, 2)
    res.robot_time_s = round(res.robot_time_s, 1)
    return res
