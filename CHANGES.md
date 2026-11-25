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
