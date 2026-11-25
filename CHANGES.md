# CHANGES — what the AI agent changed and why

Baseline commit: `baseline before agent tests` (see `git log --oneline`).
Review a fix with `git show <id>`, undo it with `git revert <id>`.

Setup notes: before the baseline commit I added a `.gitignore` (`__pycache__/`, `*.pyc`, `.pytest_cache/`) so
compiled files are not tracked, and set the git author **for this repository only** (`git config user.name`,
no global change). No packages were installed, nothing outside `~/hier_ws/src/hier_robot` was changed
(results/logs go to `~/hier_results` as before). `config/selftest.yaml` is unchanged.

## Checklist (TEST_PLAN.md)

A. Build + offline tests
- [x] `colcon build` OK, `./run.sh test` -> 7 passed.

B. Start
- [x] `./run.sh start` -> Gazebo + RViz processes running, "Ready". `/map`, `/scan`, `/odom` present.
  (RViz writes a GLSL warning for the map texture into `slam.log`: a WSL graphics-driver issue, not our code.
  I could not look at the windows myself: check them by eye, see MANUAL_CHECK.md.)
- [x] `tf2_echo map base_footprint` prints a transform (`Translation: [-0.001, 0.000, 0.008]`).

C. Self-test — 3 runs, **15 / 15 PASS** (5/5 in each run)
- [x] `./run.sh selftest` -> `~/hier_results/selftest.md`, 5 tasks.
- [x] Every FAIL classified. There was no test FAIL. Inside the tests some *steps* failed, all because of
      LLM plan errors that the replanning loop then repaired (rule 3: a result, not a bug):
  - T3, runs 1 + 2: first plan has no `pick` (`goto(hall) -> goto(room_D) -> goto(room_E) -> place(cube_3)`),
    `place` fails "робот не держит cube_3", replan is correct. Run 3: first plan `goto(room_B) -> pick(cube_3)`,
    fails "cube_3 не в room_B", replan correct.
  - T4, all runs: `pick(cube_2)` in room_A fails, then the LLM plans `goto(room_C)` -> **D* refuses
    ("D*: цель в запретной зоне")**, then the LLM moves cube_0 to room_E instead (pointless, but harmless:
    cube_2 stays in room_C as expected).
  - T2, run 2: plan ends with `goto(room_C)` (refused by D*), then the LLM repeats cubes it had already moved
    (it ignores the state it is given); cube_1, cube_3 were already in room_A.
  - T2, T5: the LLM also moves cube_0 in T2 and places at `hall` instead of `bin_hall` in T5: allowed by the
    test expectations.
- [x] Hard requirements in all runs: no crash, no LiDAR block, no `LQR: таймаут`, every goto arrived
      (RMS cross-track 0.3–4.2 cm) or was refused by `D*:` at once. A separate checker
      (subscribed to `/hierarchy/path` and TF `map->base_footprint`, compared with the room polygons of
      `building.yaml`) recorded in runs 2 + 3: 825 D* paths / 29 832 path points and 5 207 robot poses —
      **0 outside the building, 0 in room_C**.
- [x] Target met: T1, T4 3/3; T2, T3, T5 3/3.

D. Manual mode, scripted
- [x] First try FAILED: the piped tasks never reached the robot (bug in `run.sh`, see change 1).
- [x] After the fix: plan with `[1 LLM]`, `[2 D*]`, `[3 LQR]` lines, `cube_0 в room_A`, exits on `q` (exit 0).

E. Robustness
- [x] `stop` then `start` twice: 0 leftover processes after each stop, both starts "Ready", 1 gzserver,
      topics + TF present.
- [x] `/hierarchy/path` is published during every task and stays inside the building (checker numbers above;
      in check D: 25 paths / 575 points, 0 outside). Not checked by eye in RViz — see MANUAL_CHECK.md.

F. Perception (optional) — done
- [x] `./run.sh explore` -> tour 15/15, `tensor_prompt.txt` written (4 cubes: cube_0, cube_8, cube_13, cube_18;
      SigLIP2 7.4 s/frame). The mapper crashed on exit -> fixed, see change 2.
- [x] `./run.sh orders-seen` starts, takes a task: "Отвези кубик из room_B в комнату со стеллажами" ->
      `goto(room_B) -> pick(cube_8) -> goto(shelf_A1) -> place(cube_8)`, all steps done.
- Note: SigLIP2 positions are rough (the 4 real cubes are at different places than the found ones, e.g.
  "cube_0" found at (3.0, 0.8) in the hall). This is the known limit in HOW_IT_WORKS.md §7, not changed.

## Changes

### 1. 2026-11-26 — `run.sh`: piped tasks were swallowed by the LLM warm-up

- **What was wrong.** Check D (`printf 'Отвези cube_0 из room_B в room_A\nq\n' | ./run.sh orders`) did nothing:
  the robot printed the prompt and exited at once:
  ```
  Сейчас: Робот находится в hall; ... cube_0 в room_B ...
  Задача>            <- and the program ended, no plan
  ```
  `orders` first calls `warm_llm`, which runs `ollama run qwen2.5:3b "hi"`. When stdin is not a keyboard,
  `ollama run` **reads all of stdin** (it adds it to the prompt). So the two piped lines went to Ollama, and
  `robot_node` then saw an empty stdin (EOF) = "finish".
  Proof: `printf 'line1\nq\n' | bash -c 'ollama run qwen2.5:3b hi >/dev/null; cat'` prints nothing;
  with `</dev/null` added it prints `line1 q`.
- **What I changed.** `warm_llm` now runs `ollama run ... </dev/null`, so it never reads the terminal/pipe.
- **Why.** Typing by hand worked (a keyboard is not read by ollama in the same way), but any piped or scripted
  use of `orders` was broken. Nothing changes for the student's normal use.

### 2. 2026-11-26 — `hier_robot/mapper_node.py`: traceback and "failure 1" when explore stops the mapper

- **What was wrong.** At the end of `./run.sh explore` the script stops the SigLIP2 mapper with `pkill` (= Ctrl+C).
  ROS shuts down at once, but one SigLIP2 frame takes ~7 s; when it finished, the node tried to publish its
  RViz markers into an already closed ROS context and crashed:
  ```
  [WARN] [semantic_mapper]: tensor not saved: Failed to publish: publisher's context is invalid
  Traceback (most recent call last): ... mapper_node.py, line 152, in publish
  rclpy._rclpy_pybind11.RCLError: Failed to publish: publisher's context is invalid
  [ros2run]: Process exited with failure 1
  ```
  The files (`semantic_tensor.npz`, `tensor_prompt.txt`, ...) were already written, so no data was lost, but
  `mapper.log` looked like a crash.
- **What I changed.** `publish()` returns early when `rclpy.ok()` is false (ROS already shut down), and the final
  `save()` only publishes the tensor text on `/semantic_map/tensor` while ROS is still running; it always
  writes the files.
- **Why.** A clean exit: tested by starting the mapper in the running simulation, waiting for 2 frames and
  sending Ctrl+C -> `exit code 0`, no traceback, all 5 files written. (rclpy may still print
  `Failed to publish log message to rosout` once while shutting down: that is rclpy's own harmless notice.)
