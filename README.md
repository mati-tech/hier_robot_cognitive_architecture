# hier_robot — hierarchical robot: LLM → D* → LQR on RTAB-Map + SigLIP2

A TurtleBot3 in Gazebo gets a task in plain words ("bring the cube that is next to the chair
to the room with shelves") and does it by itself:

| Level | What it does | File |
|---|---|---|
| **1. LLM** (top) | reads the task + the semantic tensor, plans `goto / pick / place`, replans when a step fails | `hier_robot/level1_llm.py` |
| **SLAM** | RTAB-Map builds the map from the LiDAR and tracks where the robot is | `launch/slam.launch.py` |
| **Semantic tensor** | SigLIP2 (camera) + LiDAR + RTAB-Map pose → objects and their relations (near, same room, north of …) | `hier_robot/perception.py`, `hier_robot/semantic_tensor.py`, `hier_robot/mapper_node.py` |
| **2. D\*** (middle) | D* Lite finds a path on the RTAB-Map grid, forbidden room = no-go, repairs the path when the map changes | `hier_robot/level2_dstar.py` |
| **3. LQR** (bottom) | LQR + Kalman filter steer the wheels along the path | `hier_robot/level3_lqr.py` |
| robot program | connects all levels in ROS 2 | `hier_robot/robot_node.py` |

How it works and what to say at the defence: **HOW_IT_WORKS.md**.

## 1. Install (once)

Needs: ROS 2 Humble, TurtleBot3 + Gazebo (already in `~/turtlebot3_ws`), RTAB-Map, Ollama with `qwen2.5:3b`,
Python packages torch + transformers (only for `explore`).

```bash
sudo apt install ros-humble-rtabmap-ros          # if not installed yet
mkdir -p ~/hier_ws/src
cd ~/hier_ws/src && unzip ~/hier_robot.zip        # creates ~/hier_ws/src/hier_robot
cd ~/hier_ws && colcon build && source install/setup.bash
cd ~/hier_ws/src/hier_robot && chmod +x run.sh && ./run.sh test    # 7 tests must pass
```
This is a separate workspace (`~/hier_ws`). It does not use or change the homework folder `~/ros2_ws`.

## 2. Run — the easy way

```bash
cd ~/hier_ws/src/hier_robot
./run.sh start      # Gazebo + RTAB-Map + RViz open (~2 min)
./run.sh orders     # you type tasks, the robot does them
./run.sh selftest   # the robot runs 5 test tasks by itself and checks the results -> ~/hier_results/selftest.md
./run.sh stop       # close everything
```

In `orders` the terminal asks `Задача>`. Examples:

```
Отвези кубик из room_B в комнату со стеллажами
Отвези кубик, который лежит ближе всех к стулу, к столу
Отвези кубик из серверной комнаты в room_A          <- forbidden room C: D* refuses, the LLM replans
```

For every task you see the three levels in the log:

```
[1 LLM] Задача: ...
[1 LLM] План: goto(room_B) -> pick(cube_0) -> goto(room_A) -> place(cube_0)
[2 D*] путь 9.8 м до (-3.5, -2.4), раскрыто 6979 клеток
[3 LQR] прибыл, RMS отклонения 1.5 см, перепланирований D* 0
[1 LLM] шаг goto(room_B): выполнен
...
```
In RViz: the RTAB-Map grid, the LiDAR, the current D* path (blue line) and the robot moving.
Between tasks the robot remembers where the cubes are. Empty line or `q` ends.
The LLM needs 1–2 min per plan on this laptop (CPU only).

### Perception (SigLIP2 builds the tensor by itself)

```bash
./run.sh start
./run.sh explore        # ~20 min: tour through all rooms, SigLIP2 recognises objects
./run.sh orders-seen    # orders using the objects the robot found
```
SigLIP2 is slow on CPU (~7 s per frame) and recognises plain Gazebo shapes poorly; `orders` therefore
uses the known furniture and cube positions from `config/building.yaml` (marked "[известен заранее]" for the LLM).

## 3. Run — by hand, three terminals (to see every part)

```bash
# terminal 1: the world
source ~/hier_ws/install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=~/hier_ws/src/hier_robot/config/fastdds_udp.xml
ros2 launch hier_robot sim.launch.py

# terminal 2: SLAM (RTAB-Map) + RViz          (same two first lines)
ros2 launch hier_robot slam.launch.py use_sim_time:=true

# terminal 3: the robot program (LLM -> D* -> LQR)   (same two first lines)
ros2 run hier_robot robot_node --ros-args -p use_sim_time:=true -p use_prior:=true -p prior_cubes:=true
```
Ctrl+C in each terminal to stop.

## 4. Files

```
hier_robot/
  run.sh                 start / orders / selftest / explore / orders-seen / stop / test
  TEST_PLAN.md           instructions for an AI agent that tests and fixes the project
  hier_robot/
    level1_llm.py        LEVEL 1: LLM planner, task loop with replanning, action -> goal
    level2_dstar.py      LEVEL 2: D* Lite + costmap from the RTAB-Map grid
    level3_lqr.py        LEVEL 3: LQR tracker + EKF
    semantic_tensor.py   tensor X[N,C], P[N,2], T[N,N,7], M[N,R], A[R,R], r[R] + text for the LLM
    perception.py        SigLIP2 scoring, object landmarks on the map
    mapper_node.py       ROS node: camera + LiDAR + RTAB-Map pose -> tensor (semantic_tensor.npz)
    robot_node.py        ROS node: the whole hierarchy (modes orders / mission / explore)
    world.py             rooms, cubes, pick/place rules
    offline_sim.py       robot model for the tests (no ROS)
  launch/  sim.launch.py (Gazebo), slam.launch.py (RTAB-Map + RViz [+ mapper])
  config/  building.yaml (rooms, doors, furniture, cubes), selftest.yaml (test tasks), siglip_labels.yaml, view.rviz, fastdds_udp.xml
  worlds/office.world    maps/office.*  (map only for the tests; the robot uses RTAB-Map's map)
  test/test_levels.py
```
Results and logs: `~/hier_results/` (`missions.csv`, `logs/`, `semantic/`).

## 5. If something goes wrong

- `!! not started` → run `./run.sh start` first.
- Gazebo opens but no robot → `./run.sh stop`, then `./run.sh start` again (it spawns by hand after 90 s).
- Robot node waits for RTAB-Map forever → `ros2 topic list | grep map`; check `~/hier_results/logs/slam.log`.
- LLM is slow or times out → `ollama ps`; close other programs.
