# How hier_robot works (for the defence)

## 1. The task from the teacher, sentence by sentence

| Teacher said | We built | Where |
|---|---|---|
| "an LLM (acting as an action planner) sits at the top level" | Ollama `qwen2.5:3b` turns the task into `goto / pick / place` steps and replans when a step fails | `level1_llm.py` |
| "alongside a SLAM system — e.g. RTAB-Map" | RTAB-Map in 2D-LiDAR mode: map `/map`, robot pose (TF map→odom) | `launch/slam.launch.py` |
| "integrate SigLIP-2 with the SLAM system to generate a tensor-based representation of the semantic relationships between objects" | SigLIP2 says *what* the camera sees, LiDAR says *how far*, RTAB-Map says *where the robot is* → objects on the map → tensor of relations | `perception.py`, `mapper_node.py`, `semantic_tensor.py` |
| "essential for planning the agent's behaviour" | the tensor is the LLM's only knowledge of the building (room descriptions are hidden) | `SemanticTensor.to_prompt()` |
| "at the intermediate level a geometric planner, e.g. D*" | D* Lite on RTAB-Map's grid; forbidden room = no-go zone; repairs the path when the map changes | `level2_dstar.py` |
| "augment it with LQR" | LQR follows the D* path; a Kalman filter (EKF) estimates the pose | `level3_lqr.py` |

Simple picture: **boss** (LLM) → **navigator** (D*) → **driver** (LQR), with **memory of the building** (RTAB-Map)
and **eyes with notes** (SigLIP2 + tensor) feeding them. Orders go down, problems go up.

## 2. What happens when you type a task

Example: `Отвези кубик, который лежит ближе всех к стулу, в комнату со стеллажами`

1. **Tensor → text.** `robot_node.setup_mission()` loads the tensor and `to_prompt()` writes it for the LLM:
   ```
   [2] cube_3: класс cube, комната room_D, x=5.5 y=1.6
   [11] chair_D1: класс chair, комната room_D, x=5.0 y=2.5
   [3] shelf_A1: класс shelf, комната room_A ...
   Отношения: chair_D1 севернее cube_3 (1.00), cube_3 рядом с chair_D1 (0.23) ...
   ```
2. **Level 1 — LLM.** `SemanticLlmPlanner.plan()` sends rules + this text + the task to Ollama and gets JSON:
   `goto(room_D) → pick(cube_3) → goto(room_A) → place(cube_3)`. Shelves are in room A, so "the room with shelves" = room_A.
3. **Action → goal.** `ActionExecutor.execute()` turns `goto(room_A)` into a point (x, y)
   (a room centre, or the spot in front of an object such as `table_D`).
4. **Level 2 — D\*.** `GeometricLayer.plan()` builds a costmap from RTAB-Map's grid
   (walls inflated by the robot radius, forbidden room C = infinite cost, unknown cells = expensive)
   and runs D* Lite → a path, smoothed. Shown in RViz on `/hierarchy/path`.
5. **Level 3 — LQR.** `robot_node.drive()` 20 times per second: EKF pose (odometry + RTAB-Map pose) →
   `LqrTracker.command()` → `/cmd_vel`. LiDAR stops the robot if something is closer than 0.2 m.
6. **Map changes.** Every 2 s, if RTAB-Map published a new grid, D* *repairs* the path
   (`update_map`: only the changed cells are re-planned) instead of planning from scratch.
7. **Failures go up.** If D* says "цель в запретной зоне" (goal in the forbidden zone) or "путь не найден",
   the step fails; `run_hierarchical()` sends the reason and the current state back to the LLM, which replans (max 3).
   This replaces ACT-R: safety is guaranteed by geometry, the LLM cannot drive the robot into room C.

## 3. The semantic tensor

| Tensor | Size | Meaning |
|---|---|---|
| X | N × C | for each object, SigLIP2 score per class (cube, door, chair, table, shelf, trash_can) |
| P | N × 2 | object position in the RTAB-Map map (m) |
| **T** | **N × N × 7** | **relations**: same_room, near (< 1.5 m), north_of, south_of, east_of, west_of, door_between |
| M | N × R | which room each object is in |
| A | R × R | which rooms are connected by doors |
| r | R | forbidden rooms (used by D* as no-go) |

N = objects, C = classes, R = rooms. Built in `SemanticTensor.build()`, saved as `semantic_tensor.npz`.

## 4. D* Lite in two sentences

D* Lite searches **backwards from the goal**, so when the robot moves only an offset `km` changes, and when some
cells change (`update_costs`) only the affected part is recomputed. Measured on our map: same optimal path as
planning from scratch, 2× fewer cell expansions when an obstacle appears on the path, ~37× fewer when the change is
off the path (the usual case while RTAB-Map grows the map).

## 5. LQR in two sentences

The error between robot and path is (along, across, heading); for the linearised unicycle the discrete Riccati
equation gives the gain K and the command is u = −K·e (plus the path curvature as feed-forward).
Q = diag(2, 12, 4) punishes the sideways error most, R = diag(1, 0.6) the speeds; measured tracking error 1–2.5 cm.

## 6. Two ways to show it

- `./run.sh start` + `./run.sh orders`: you give orders → see the three levels working. Furniture and cube positions
  come from `config/building.yaml` ("known in advance").
- `./run.sh explore`: the robot builds the map and the tensor itself with SigLIP2 (slow on CPU; plain Gazebo
  shapes are recognised poorly — mainly cubes). Then `./run.sh orders-seen`.

## 7. Questions the teacher may ask

- **Why RTAB-Map without the camera?** Waffle Pi has no depth camera; RTAB-Map uses the LiDAR (ICP) + odometry.
  The camera is used by SigLIP2. (`use_camera:=true` in `slam.launch.py` also feeds it to RTAB-Map for loop closure.)
- **Why D* and not A\*?** The map changes while the robot drives (SLAM); D* Lite repairs the path instead of replanning.
- **Where is the "tensor"?** `semantic_tensor.py`, array `T[N, N, 7]`; open `~/hier_results/semantic/semantic_tensor.npz`
  with numpy, or read `tensor_prompt.txt`.
- **What if the LLM makes a mistake?** Unknown names are rejected before execution; impossible or forbidden steps fail
  at D* or at the pick/place rules, and the LLM replans with the reason. The robot never enters room C.
- **Why is it slow?** CPU only: SigLIP2 ~7 s per frame, the LLM 1–2 min per plan. With a GPU both are seconds.
- **What are the limits?** SigLIP2-base on untextured Gazebo models; a small 3B LLM sometimes orders steps wrongly
  (the replanning loop fixes most); object positions are not corrected after RTAB-Map loop closures.
