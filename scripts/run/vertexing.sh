#!/usr/bin/env bash
# VertexHead evaluation with ACTS, from setup to the comparison against truth. From anywhere:
#   scripts/run/vertexing.sh setup [--rebuild-base]   ACTS image (if missing) and module; the site detector file
#   scripts/run/vertexing.sh eval [options]           eval_vertex_reconstruction.py with the run config
#   scripts/run/vertexing.sh compare [options]        compare_vertexing.py on the run's newest per-point CSV
# Options pass through to the Python script. RUN_CONFIG picks the run (default
# scripts/configs/vertex_m5_gpuserver0.yaml); its detector: key names the site detector file, which setup creates
# from scripts/acts/detectors/sphenix/sphenix.yaml if missing.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUN_CONFIG=$(realpath "${RUN_CONFIG:-$REPO/scripts/configs/vertex_m5_gpuserver0.yaml}")
COMMAND=${1:-}
shift || true

# the run config's detector file and run folder (paths relative to the config, ~ expanded)
{ read -r DETECTOR; read -r RUN_DIR; } < <(python - "$RUN_CONFIG" <<'EOF' | tr -d '\r'
import os, sys, yaml
path = sys.argv[1]
run = yaml.safe_load(open(path))
where = lambda p: os.path.join(os.path.dirname(path), os.path.expanduser(p))  # noqa: E731
print(where(run["detector"]))
print(os.path.join(where(run["root_dir"]), str(run["config"]), str(run["run_num"])))
EOF
)

case "$COMMAND" in
    setup)
        "$REPO/scripts/acts/build_images.sh" "$@"
        if [[ ! -f "$DETECTOR" ]]; then
            image=$(realpath --relative-to="$(dirname "$DETECTOR")" "$REPO/scripts/acts/acts-base-48.2.0.sif")
            sed "s|^  image: null .*|  image: $image|" "$REPO/scripts/acts/detectors/sphenix/sphenix.yaml" > "$DETECTOR"
            echo "wrote $DETECTOR: fill in acts.geometry, geometry_config and material_map (README: sPHENIX inputs)"
        fi
        ;;
    eval)
        cd "$REPO/train/downstream"
        python eval_vertex_reconstruction.py --run_config "$RUN_CONFIG" "$@"
        ;;
    compare)
        csv=$(ls -t "$RUN_DIR"/*_per_point_data.csv 2> /dev/null | head -1)
        [[ -n "$csv" ]] || { echo "no per-point CSV in $RUN_DIR: run 'eval' first" >&2; exit 1; }
        python "$REPO/scripts/acts/compare_vertexing.py" --csv "$csv" --detector "$DETECTOR" --tracks both "$@"
        ;;
    *)
        sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac
