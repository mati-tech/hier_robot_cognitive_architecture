"""Symbolic world model shared by the planner, the ACT-R supervisor and the executors.

The world is a set of rooms (each with a Nav2 goal pose in the `map` frame) and
cubes that lie in rooms. The robot can carry one cube at a time (TurtleBot3 has
no gripper, so pick/place are "load / unload" operations done by an operator or
a simple tray). Some rooms are restricted: the robot is not allowed to enter
them. This is the access logic that ACT-R enforces.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml

NONE = "none"
ACTIONS = ("goto", "pick", "place")


@dataclass
class Room:
    name: str
    pose: Tuple[float, float, float]  # x, y, yaw in map frame
    restricted: bool = False
    description: str = ""


@dataclass
class Obj:
    name: str
    room: str
    pose: Optional[Tuple[float, float, float]] = None  # approach pose, optional


@dataclass
class Step:
    action: str           # goto | pick | place | invalid
    target: str           # room name for goto, object name for pick/place
    raw: str = ""         # original text from the LLM, for logs

    def __str__(self) -> str:
        return f"{self.action}({self.target})"

    def to_dict(self) -> dict:
        key = "target" if self.action == "goto" else "object"
        return {"action": self.action, key: self.target}


@dataclass
class State:
    robot_room: str
    holding: str = NONE
    obj_room: Dict[str, str] = field(default_factory=dict)  # cube -> room, or "robot"

    def copy(self) -> "State":
        return copy.deepcopy(self)

    def describe(self) -> str:
        cubes = ", ".join(f"{o} в {r}" for o, r in sorted(self.obj_room.items()) if r != "robot")
        return (f"Робот находится в {self.robot_room}; в манипуляторе: {self.holding}; "
                f"кубики: {cubes or 'нет'}")


class World:
    def __init__(self, rooms: Dict[str, Room], objects: Dict[str, Obj], start_room: str,
                 speed: float = 0.18):
        self.rooms = rooms
        self.objects = objects
        self.start_room = start_room
        self.speed = speed  # m/s, used for time estimates in simulation

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_yaml(cls, path: str) -> "World":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        rooms = {
            name: Room(name=name, pose=tuple(r["pose"]), restricted=bool(r.get("restricted", False)),
                       description=r.get("description", ""))
            for name, r in cfg["rooms"].items()
        }
        objects = {
            name: Obj(name=name, room=o["room"], pose=tuple(o["pose"]) if o.get("pose") else None)
            for name, o in cfg["objects"].items()
        }
        return cls(rooms, objects, cfg.get("start_room", next(iter(rooms))),
                   float(cfg.get("speed", 0.18)))

    def initial_state(self) -> State:
        return State(robot_room=self.start_room, holding=NONE,
                     obj_room={n: o.room for n, o in self.objects.items()})

    # ------------------------------------------------------------------ helpers
    def distance(self, a: str, b: str) -> float:
        if a not in self.rooms or b not in self.rooms:
            return 0.0
        (x1, y1, _), (x2, y2, _) = self.rooms[a].pose, self.rooms[b].pose
        return math.hypot(x2 - x1, y2 - y1)

    def describe_for_llm(self, state: State) -> str:
        lines = ["Комнаты:"]
        for r in self.rooms.values():
            flag = " (ДОСТУП ЗАПРЕЩЁН)" if r.restricted else ""
            desc = f" — {r.description}" if r.description else ""
            lines.append(f"  - {r.name}{flag}{desc}")
        lines.append("Объекты:")
        for o, room in sorted(state.obj_room.items()):
            where = "у робота" if room == "robot" else room
            lines.append(f"  - {o}: {where}")
        lines.append(f"Робот сейчас: {state.robot_room}, держит: {state.holding}")
        return "\n".join(lines)

    # ----------------------------------------------------- ground-truth physics
    def apply(self, state: State, step: Step) -> Tuple[bool, str, bool]:
        """Execute a step in the symbolic world.

        Returns (ok, message, safety_violation). A safety violation is a step the
        robot physically *can* do but must not do (entering a restricted room).
        Steps that are physically impossible (picking a cube that is elsewhere)
        fail with ok=False and leave the state unchanged.
        """
        if step.action == "goto":
            if step.target not in self.rooms:
                return False, f"нет такой комнаты {step.target}", False
            violation = self.rooms[step.target].restricted
            state.robot_room = step.target
            return True, "ok", violation
        if step.action == "pick":
            if step.target not in state.obj_room:
                return False, f"объект {step.target} не найден", False
            if state.holding != NONE:
                return False, "руки заняты", False
            if state.obj_room[step.target] != state.robot_room:
                return False, f"{step.target} не в {state.robot_room}", False
            state.obj_room[step.target] = "robot"
            state.holding = step.target
            return True, "ok", False
        if step.action == "place":
            if state.holding != step.target:
                return False, f"робот не держит {step.target}", False
            state.obj_room[step.target] = state.robot_room
            state.holding = NONE
            return True, "ok", False
        return False, f"нераспознанное действие {step.raw or step}", False


def goal_satisfied(state: State, goal: Dict[str, str]) -> Tuple[int, int]:
    """Return (achieved, total) for a goal {cube: room}."""
    achieved = sum(1 for o, r in goal.items() if state.obj_room.get(o) == r)
    return achieved, len(goal)


def default_config(name: str) -> str:
    """Path to a file in config/: source tree first, then the installed share directory."""
    import os
    here = os.path.join(os.path.dirname(__file__), "..", "config", name)
    if os.path.exists(here):
        return here
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("hier_robot"), "config", name)
    except Exception:  # noqa: BLE001
        return here
