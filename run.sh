#!/usr/bin/env bash
# hier_robot — hierarchical robot: LLM -> D* Lite -> LQR, on RTAB-Map SLAM + SigLIP2 semantic tensor.
#
#   ./run.sh start     open Gazebo (office world) + RTAB-Map + RViz            (~2 min)
#   ./run.sh orders    YOU type tasks, the robot does them                      (after start)
#   ./run.sh explore   the robot drives through all rooms; SigLIP2 recognises objects
#                      and builds the semantic tensor                            (after start, ~20 min)
#   ./run.sh orders-seen   like orders, but with the objects SigLIP2 found in explore
#   ./run.sh selftest  the robot runs 5 test tasks by itself and checks the results  (after start)
#   ./run.sh stop      close everything
#   ./run.sh test      run the offline tests (no Gazebo)
#
# Results and logs: ~/hier_results/
HERE="$(cd "$(dirname "$0")" && pwd)"
set +u   # ROS setup files read unset variables
for f in /opt/ros/humble/setup.bash ~/turtlebot3_ws/install/setup.bash ~/hier_ws/install/setup.bash \
         /usr/share/gazebo/setup.sh; do
  # shellcheck disable=SC1090
  [ -f "$f" ] && source "$f"
done
set -u
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-waffle_pi}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"
# UDP-only DDS: on WSL2 the shared-memory transport silently stopped delivering messages
export FASTRTPS_DEFAULT_PROFILES_FILE="$HERE/config/fastdds_udp.xml"
export RMW_FASTRTPS_USE_QOS_FROM_XML=0

OUT=~/hier_results
LOG=$OUT/logs
mkdir -p "$LOG" "$OUT/semantic"

say() { echo; echo "=== $* ==="; }
wait_for() {  # wait_for <seconds> <description> <command...>
  local t=$1 what=$2; shift 2
  for ((i = 0; i < t; i += 3)); do
    if "$@" >/dev/null 2>&1; then return 0; fi
    sleep 3
  done
  echo "!! timed out waiting for: $what (logs: $LOG)"; return 1
}
topic_up() { ros2 topic list 2>/dev/null | grep -qx "$1"; }
need_start() { topic_up /map || { echo "!! not started: run ./run.sh start first"; exit 1; }; }
warm_llm() { ollama run "$OLLAMA_MODEL" "hi" >/dev/null 2>&1; }

stop_all() {
  # patterns chosen so they never match this script itself
  pkill -f "lib/hier_robot/"; pkill -f "launch hier_robot"; pkill -f "lib/rtabmap_slam/"; pkill -f rviz2
  pkill -f gzserver; pkill -f gzclient; pkill -f robot_state_publisher
  sleep 4
  # a gzserver that is still alive makes the next start fail (no /scan)
  for _ in $(seq 10); do pgrep -f "gzserver|lib/rtabmap_slam/|lib/hier_robot/" >/dev/null || break; sleep 2; done
  pkill -9 -f "gzserver|gzclient|lib/rtabmap_slam/|lib/hier_robot/" 2>/dev/null
  echo "everything stopped"
}

start() {
  stop_all
  say "1/2 Gazebo: office world, robot in the hall at (0, 0)"
  nohup ros2 launch hier_robot sim.launch.py >"$LOG/gazebo.log" 2>&1 &
  if ! wait_for 90 "robot in Gazebo (/scan)" topic_up /scan; then
    echo "   spawning the robot manually (WSL sometimes stalls the spawn service)"
    ros2 run gazebo_ros spawn_entity.py -entity waffle_pi -x 0.0 -y 0.0 -z 0.01 \
      -file "$(ros2 pkg prefix turtlebot3_gazebo)/share/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf" \
      >>"$LOG/gazebo.log" 2>&1
    wait_for 60 "robot in Gazebo (/scan)" topic_up /scan || exit 1
  fi
  sleep 5
  say "2/2 RTAB-Map SLAM + RViz"
  ros2 pkg prefix rtabmap_slam >/dev/null 2>&1 || {
    echo "!! RTAB-Map is not installed: sudo apt install ros-humble-rtabmap-ros"; exit 1; }
  nohup ros2 launch hier_robot slam.launch.py use_sim_time:=true >"$LOG/slam.log" 2>&1 &
  wait_for 120 "RTAB-Map grid (/map)" topic_up /map || exit 1
  sleep 5
  echo
  echo "Ready. Gazebo and RViz are open. Next:  ./run.sh orders   (or ./run.sh explore)"
}

orders() {   # $1 = known | seen
  need_start
  warm_llm
  local args=(-p use_sim_time:=true -p mode:=orders -p use_prior:=true)
  if [ "${1:-known}" = seen ]; then
    args+=(-p tensor_file:="$OUT/semantic/semantic_tensor.npz")
    say "ORDERS: objects found by SigLIP2 (explore) + known furniture"
  else
    args+=(-p prior_cubes:=true -p tensor_file:=none)
    say "ORDERS: known furniture + known cube positions"
  fi
  echo "Type a task and press Enter. Examples:"
  echo "   Отвези кубик из room_B в комнату со стеллажами"
  echo "   Отвези кубик, который лежит ближе всех к стулу, к столу"
  echo "   Отвези кубик из серверной комнаты в room_A      (forbidden room: D* refuses)"
  echo "Empty line or q = finish."
  # the task prompt stays on the terminal; the log lines are also saved
  ros2 run hier_robot robot_node --ros-args "${args[@]}" 2> >(tee "$LOG/orders_$(date +%H%M%S).log" >&2)
}

selftest() {
  need_start
  warm_llm
  say "SELFTEST: 5 tasks from config/selftest.yaml, known furniture + cube positions (~20-40 min)"
  ros2 run hier_robot robot_node --ros-args -p use_sim_time:=true -p mode:=selftest -p use_prior:=true \
    -p prior_cubes:=true -p tensor_file:=none 2> >(tee "$LOG/selftest_$(date +%H%M%S).log" >&2)
  echo; cat "$OUT/selftest.md"
}

explore() {
  need_start
  say "SigLIP2 mapper (recognition) — first load takes ~30 s"
  # cap torch threads: unbounded SigLIP2 took all cores and starved the ROS nodes
  OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 nohup ros2 run hier_robot mapper_node --ros-args -p use_sim_time:=true \
    -p world_file:="$(ros2 pkg prefix hier_robot)/share/hier_robot/config/building.yaml" \
    -p labels_file:="$(ros2 pkg prefix hier_robot)/share/hier_robot/config/siglip_labels.yaml" \
    -p rate_hz:=0.5 -p out_dir:="$OUT/semantic" >"$LOG/mapper.log" 2>&1 &
  wait_for 180 "SigLIP2 model loaded" grep -q "SigLIP2 ready" "$LOG/mapper.log" || exit 1
  say "EXPLORE: tour through all rooms (D* plans each hop, LQR drives; room C is never entered)"
  ros2 run hier_robot robot_node --ros-args -p use_sim_time:=true -p mode:=explore \
    2> >(tee "$LOG/explore.log" >&2)
  sleep 12   # let the mapper save once more
  pkill -f "lib/hier_robot/mapper_node"
  say "semantic tensor (what the LLM will see)"
  cat "$OUT/semantic/tensor_prompt.txt" 2>/dev/null || echo "!! no tensor: see $LOG/mapper.log"
  echo; echo "Next:  ./run.sh orders-seen"
}

case "${1:-}" in
  start) start ;;
  orders) orders known ;;
  orders-seen) orders seen ;;
  explore) explore ;;
  selftest) selftest ;;
  stop) stop_all ;;
  test) cd "$HERE" && python3 -m pytest -q test ;;
  *) sed -n '2,13p' "$0" ;;
esac
