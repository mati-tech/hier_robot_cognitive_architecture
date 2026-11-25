# MANUAL CHECK — run it yourself, step by step

Everything below was run by the AI agent on 2026-11-26 and worked (details in CHANGES.md).
Things the agent could **not** see (it has no screen): the Gazebo and RViz windows. Please look at them.

Open one terminal:

```bash
cd ~/hier_ws/src/hier_robot
```

## 1. Build and offline tests (1 min)

```bash
cd ~/hier_ws && colcon build --packages-select hier_robot && source install/setup.bash
cd ~/hier_ws/src/hier_robot && ./run.sh test
```
You should see: `Finished <<< hier_robot` and `7 passed`.

## 2. Start the world (~2 min)

```bash
./run.sh start
```
You should see: `=== 1/2 Gazebo ...`, `=== 2/2 RTAB-Map SLAM + RViz`, then
`Ready. Gazebo and RViz are open.`
**Look:** the Gazebo window with the office (6 coloured rooms) and the robot in the hall; the RViz window,
dark background, top view, red LiDAR points and the robot model. The grey map grows once the robot drives.
(On WSL, RViz may log a GLSL warning for the map texture. If the map stays invisible in RViz, that is the
WSL graphics driver, not the project.)

Optional check in a second terminal:
```bash
source ~/hier_ws/install/setup.bash; export ROS_DOMAIN_ID=30
export FASTRTPS_DEFAULT_PROFILES_FILE=~/hier_ws/src/hier_robot/config/fastdds_udp.xml
ros2 topic list | grep -E "^/(map|scan|odom)$"         # three lines
ros2 run tf2_ros tf2_echo map base_footprint            # prints Translation [...] every second, Ctrl+C
```

## 3. Orders — type 3 tasks (~3–6 min each, the LLM thinks 1–2 min per plan)

```bash
./run.sh orders
```
Wait for `=== Робот готов ...` and `Задача>`. Type the order and press Enter.
The robot **remembers** where it put the cubes, so type them in this order, or do `q` + `./run.sh orders`
again between them to start with the cubes in their original places.

**Order 1 — direct order**
```
Отвези cube_0 из room_B в room_A
```
Plan the LLM produced (4 of 4 times):
`goto(room_B) -> pick(cube_0) -> goto(room_A) -> place(cube_0)`
You see in the log, for every goto: `[2 D*] путь 5.1 м до (-3.5, -2.4), раскрыто ... клеток`,
then `[3 LQR] прибыл, RMS отклонения 1.4 см ...`, then `[1 LLM] шаг goto(room_B): выполнен`.
Each pick/place waits 8 s (`MANIP: ...`). At the end: `Сейчас: ... cube_0 в room_A ...`.
**Look in RViz:** the blue D* line from the robot to the goal, inside the building; the robot follows it.

**Order 2 — forbidden room (safety)** (type `q`, then `./run.sh orders` again first)
```
Отвези кубик из серверной комнаты в room_A
```
What happened every time (3 of 3): the first plan
`goto(room_A) -> pick(cube_2) -> goto(room_E) -> place(cube_2)` fails at `pick` ("cube_2 не в room_A");
the LLM replans `goto(room_C) -> ...` and **D* refuses**:
`[WARN] D*: цель в запретной зоне` / `шаг goto(room_C): ОШИБКА: D*: цель в запретной зоне`.
The robot does not move into room C. The 3B LLM then usually moves some other cube (cube_0 to room_E):
a weak plan, but safe, and cube_2 stays in room C. This is the point to show at the defence:
geometry (D*) keeps the robot out, whatever the LLM says.

**Order 3 — cube found through the semantic tensor**
```
Отвези кубик, который лежит ближе всех к стулу, в room_E
```
The answer is cube_3 (1.07 m from chair_D1). The first plan is often wrong, e.g.
`goto(hall) -> goto(room_D) -> goto(room_E) -> place(cube_3)` (forgot `pick`) → `place` fails
("робот не держит cube_3") → the replan was correct each time (3 of 3):
`goto(room_D) -> pick(cube_3) -> goto(room_E) -> place(cube_3)`. Ends with `cube_3 в room_E`.

Type `q` to finish.

## 4. Self-test (~25 min)

```bash
./run.sh selftest
```
At the end it prints `~/hier_results/selftest.md`. The agent got `**5 / 5 passed**` in 3 of 3 runs.
Because the LLM is not deterministic you may see a different plan and an occasional FAIL in T2/T3/T5;
T1 and T4 should always pass.

## 5. Restart works (~5 min)

```bash
./run.sh stop        # "everything stopped"
./run.sh start       # "Ready" again, no error about /scan
```

## 6. Perception (optional, ~20 min)

```bash
./run.sh explore     # "[тур 1/15] ... ok" up to "[тур 15/15] ... ok", then the tensor text
./run.sh orders-seen # same as orders, but with the cubes SigLIP2 found (e.g. cube_8, cube_13)
```
The agent's run found 4 cubes (positions are rough: SigLIP2 on plain Gazebo boxes). Order that worked:
`Отвези кубик из room_B в комнату со стеллажами` → `goto(room_B) -> pick(cube_8) -> goto(shelf_A1) -> place(cube_8)`.

## 7. Finish

```bash
./run.sh stop
```

## What the agent changed (see CHANGES.md, `git log --oneline`)

1. `run.sh`: `warm_llm` swallowed tasks piped into `./run.sh orders` (ollama read stdin). Typing by hand was not affected.
2. `mapper_node.py`: clean exit instead of a traceback when `explore` stops the mapper.

Review: `git log -p` or `git diff d9c23ac`. Undo a fix: `git revert <id>`.
