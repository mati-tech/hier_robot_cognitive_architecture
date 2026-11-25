# TEST PLAN for the AI agent — hier_robot

Goal: make sure everything works **before the student tests it by hand**. You may fix bugs, but every change
must be visible and explained, because the student has to understand and defend this code.

Project: hierarchical robot LLM (level 1) -> D* Lite (level 2) -> LQR + EKF (level 3) on RTAB-Map SLAM with a
SigLIP2 semantic tensor. Read README.md and HOW_IT_WORKS.md first. Workspace: `~/hier_ws` (separate from `~/ros2_ws`;
never touch `~/ros2_ws`).

## Rules (important)

1. **Git first.** Before any change:
   `cd ~/hier_ws/src/hier_robot && git init -q 2>/dev/null; git add -A && git commit -qm "baseline before agent tests" || true`
   Commit **each fix separately** with a clear message (`git commit -am "fix: ..."`). Never squash, never rewrite history.
2. **Log every change in CHANGES.md** (create it): date, file, what was wrong (with the log line), what you changed, why.
   Plain language — the student must be able to explain it.
3. **Do not cheat the tests.** Never change `config/selftest.yaml` tasks or expectations, never special-case a test
   task in the code, never hard-code answers. The LLM is not deterministic: if a test fails because the LLM made a
   bad plan, re-run it up to 2 more times before treating it as a code bug. A bad LLM plan is a result, not a bug.
4. Keep the student's commands working exactly as in README.md (`./run.sh start | orders | selftest | explore | stop`).
   If you must change how something is run, update README.md and say so in CHANGES.md.
5. Ask the student before installing anything. Do not change files outside `~/hier_ws/src/hier_robot`.
6. After every code change: `cd ~/hier_ws && colcon build --packages-select hier_robot && source install/setup.bash`
   and `cd ~/hier_ws/src/hier_robot && ./run.sh test` (must stay green).

## Checks (do them in this order, tick in CHANGES.md)

A. Build + offline tests
- [ ] `colcon build` OK, `./run.sh test` -> 7 passed.

B. Start
- [ ] `./run.sh start` -> Gazebo + RViz open, ends with "Ready". `ros2 topic list` has `/map`, `/scan`, `/odom`.
- [ ] `ros2 run tf2_ros tf2_echo map base_footprint` prints a transform.

C. Self-test (main check)
- [ ] `./run.sh selftest` -> `~/hier_results/selftest.md`. 5 tasks:
  T1 direct order, T2 room found by its furniture, T3 cube nearest to a chair, T4 forbidden room (must NOT be entered),
  T5 deliver to an object (bin_hall).
- [ ] For every FAIL decide: LLM plan error (see rule 3) or code bug (wrong level behaviour: D* path outside the
      building, robot stuck/oscillating, wrong goal point, crash, node not reacting). Fix code bugs only.
- [ ] Hard requirements, must hold in every run: no crash, robot never enters room_C, never leaves the building,
      every `goto` either arrives or fails with a clear `D*:` / `LQR:` / `LiDAR` reason within ~5 min.
- [ ] Target: T1, T4 always PASS; T2, T3, T5 PASS in at least one of 3 runs (3B LLM).

D. Manual mode, scripted (simulates the student typing)
- [ ] `printf 'Отвези cube_0 из room_B в room_A\nq\n' | ./run.sh orders` -> plan printed with [1 LLM], [2 D*], [3 LQR]
      lines, cube_0 ends in room_A, program exits on `q`.

E. Robustness
- [ ] Run `./run.sh stop` then `./run.sh start` twice in a row: second start must work (no leftover gzserver).
- [ ] While a task drives, the RViz D* path (`/hierarchy/path`) is visible and stays inside the building.

F. Perception (optional, slow ~20 min)
- [ ] `./run.sh explore` -> tour 15/15 hops, `~/hier_results/semantic/tensor_prompt.txt` written (any number of objects).
- [ ] `./run.sh orders-seen` starts and accepts a task.

## At the end

1. `./run.sh stop`.
2. Write **MANUAL_CHECK.md** for the student: the exact commands to run by hand (in order), what he should see after
   each one, and 3 orders to type that worked for you, with the plan the LLM produced.
3. Summary to the student (short): what passed, what you fixed (one line each, see CHANGES.md and `git log --oneline`),
   what still fails and why. He reviews with `git log -p` or `git diff <baseline>`; any fix can be undone with `git revert <id>`.
