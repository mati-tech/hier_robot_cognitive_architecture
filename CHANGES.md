# CHANGES — what the AI agent changed and why

Baseline commit: `baseline before agent tests` (see `git log --oneline`).
Review a fix with `git show <id>`, undo it with `git revert <id>`.

## Checklist (TEST_PLAN.md)

A. Build + offline tests
- [x] `colcon build` OK, `./run.sh test` -> 7 passed.

B. Start
- [x] `./run.sh start` -> Gazebo + RViz open, "Ready". `/map`, `/scan`, `/odom` present.
- [x] `tf2_echo map base_footprint` prints a transform.

C. Self-test
- [ ] (filled in below when all runs are done)

D. Manual mode, scripted
- [ ] (filled in below)

E. Robustness
- [ ] (filled in below)

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
