#!/usr/bin/env bash
# VertexHead evaluation with ACTS, from setup to the comparison against truth. From anywhere:
#   scripts/run/vertexing.sh setup [options]      ACTS image (if missing) and module; writes or updates the site
#                                                 detector file
#   scripts/run/vertexing.sh eval [options]       eval_vertex_reconstruction.py with the run config
#   scripts/run/vertexing.sh compare [options]    VertexHead vs ACTS's vertex finder on the run's newest CSV
# setup options: --geometry, --geometry_config, --material_map, --field_map <file> (ACTS inputs, must exist);
# --set <section.key>=<value> for any other detector-file setting (e.g. --set acts.energy_loss=true,
# --set vertex_head.track_error_scale=2.5); --rebuild-base. Other commands pass options to the Python script.
# RUN_CONFIG picks the run (default scripts/configs/vertex_m5_gpuserver0.yaml); its detector: key names the site
# detector file, which setup creates from scripts/acts/detectors/sphenix/sphenix.yaml if missing.
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
        build_args=()
        settings=()
        while (( $# )); do
            case "$1" in
                --geometry|--geometry_config|--material_map|--field_map)
                    [[ -f "${2:-}" ]] || { echo "$1: no such file: ${2:-}" >&2; exit 1; }
                    key=${1#--}
                    [[ $key == field_map ]] && key=field.map || key=acts.$key
                    settings+=("$key=$(realpath "$2")"); shift 2 ;;
                --set) settings+=("${2:?--set needs section.key=value}"); shift 2 ;;
                *) build_args+=("$1"); shift ;;
            esac
        done
        if [[ ! -f "$DETECTOR" ]]; then
            image=$(realpath --relative-to="$(dirname "$DETECTOR")" "$REPO/scripts/acts/acts-base-48.2.0.sif")
            sed "s|^  image: null .*|  image: $image|" "$REPO/scripts/acts/detectors/sphenix/sphenix.yaml" > "$DETECTOR"
            echo "wrote $DETECTOR"
        fi
        # each section.key=value replaces that line's value (its comment kept) or is added to its section
        python - "$DETECTOR" ${settings[@]+"${settings[@]}"} <<'EOF'
import re, sys
path, settings = sys.argv[1], sys.argv[2:]
lines = open(path).read().splitlines()
for setting in settings:
    name, sep, value = setting.partition("=")
    section, _, key = name.rpartition(".")
    if not sep or not key:
        sys.exit(f"--set {setting}: expected section.key=value")
    start = next((i for i, l in enumerate(lines) if re.match(rf"{re.escape(section)}:(\s|$)", l)), None) if section else -1
    if start is None:
        sys.exit(f"--set {setting}: no section {section!r} in {path}")
    lines[start] = re.sub(r":\s*\{\}", ":", lines[start]) if section else lines[start]
    end = next((i for i in range(start + 1, len(lines)) if lines[i] and not lines[i][0].isspace() and not lines[i].startswith("#")),
               len(lines))
    indent = "  " if section else ""
    hit = next((i for i in range(start + 1, end) if re.match(rf"{indent}{re.escape(key)}:(\s|$)", lines[i])), None)
    if hit is None:
        last = max((i for i in range(start, end) if lines[i].startswith(indent) and lines[i].strip()), default=start)
        lines.insert(last + 1, f"{indent}{key}: {value}")
    else:
        old = lines[hit]
        comment = old[old.index("#"):] if "#" in old else ""
        new = f"{indent}{key}: {value}"
        lines[hit] = f"{new:{max(len(old) - len(comment), len(new) + 1)}s}{comment}" if comment else new
    print(f"set {name}: {value}")
open(path, "w").write("\n".join(lines) + "\n")
EOF
        "$REPO/scripts/acts/build_images.sh" ${build_args[@]+"${build_args[@]}"}
        if grep -qE "^  (geometry|geometry_config|material_map): null" "$DETECTOR"; then
            echo "$DETECTOR: no ACTS geometry yet; rerun setup with --geometry <tgeo.root> --geometry_config <json>" \
                 "--material_map <json> (README: sPHENIX inputs)"
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
        sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac
