#!/usr/bin/env bash
# VertexHead evaluation with ACTS, from setup to the comparison against truth. From anywhere:
#   scripts/run/vertexing.sh setup [options]      ACTS image (if missing) and module; writes or updates the site
#                                                 detector file
#   scripts/run/vertexing.sh eval [options]       eval_vertex_reconstruction.py with the run config
#   scripts/run/vertexing.sh compare [options]    VertexHead vs ACTS's vertex finder on the newest CSV in root_dir
# setup options: --geometry, --geometry_config, --material_map <file> (ACTS inputs, must exist);
# --field_map <file> (a ROOT map, converted into the detector folder and then used by every site file, or an ACTS
# text map; with --field_kind xyz|rz, --field_rescale <factor>); --set <section.key>=<value> for any other detector-file setting (e.g.
# --set acts.energy_loss=true, --set vertex_head.track_error_scale=2.5); --rebuild-base. Other commands pass
# options to the Python script.
# RUN_CONFIG picks the run (default scripts/configs/vertex_m5_gpuserver0.yaml); its detector: key names the site
# detector file, which setup writes from scripts/acts/detectors/sphenix/sphenix.yaml, keeping the site's own values.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUN_CONFIG=$(realpath "${RUN_CONFIG:-$REPO/scripts/configs/vertex_m5_gpuserver0.yaml}")
COMMAND=${1:-}
shift || true

# the run config's detector file and output folder (paths relative to the config, ~ expanded)
{ read -r DETECTOR; read -r ROOT_DIR; } < <(python - "$RUN_CONFIG" <<'EOF' | tr -d '\r'
import os, sys, yaml
path = sys.argv[1]
run = yaml.safe_load(open(path))
where = lambda p: os.path.join(os.path.dirname(path), os.path.expanduser(p))  # noqa: E731
print(where(run["detector"]))
print(where(run["root_dir"]))
EOF
)

case "$COMMAND" in
    setup)
        build_args=()
        settings=()
        convert_args=()
        field_map="" field_kind=xyz
        while (( $# )); do
            case "$1" in
                --geometry|--geometry_config|--material_map|--field_map)
                    [[ -f "${2:-}" ]] || { echo "$1: no such file: ${2:-}" >&2; exit 1; }
                    if [[ $1 == --field_map ]]; then field_map=$(realpath "$2")
                    else settings+=("acts.${1#--}=$(realpath "$2")"); fi
                    shift 2 ;;
                --field_kind) field_kind=${2:?--field_kind needs xyz or rz}; shift 2 ;;
                --field_rescale) convert_args+=(--rescale "${2:?--field_rescale needs a factor}"); shift 2 ;;
                --set) settings+=("${2:?--set needs section.key=value}"); shift 2 ;;
                *) build_args+=("$1"); shift ;;
            esac
        done
        # the field map belongs to the detector: a ROOT map (sPHENIX's conditions database) becomes ACTS's text map
        # in the detector folder, which every site file then uses
        detector_dir="$REPO/scripts/acts/detectors/sphenix"
        if [[ $field_map == *.root ]]; then
            python "$detector_dir/convert_field_map.py" "$field_map" "$detector_dir/field_$field_kind.txt" \
                --kind "$field_kind" ${convert_args[@]+"${convert_args[@]}"}
            field_map=$detector_dir/field_$field_kind.txt
        fi
        # a map converted earlier is used by a site file that has none yet
        if [[ -z $field_map && -f $detector_dir/field_$field_kind.txt ]] &&
                { [[ ! -f "$DETECTOR" ]] || grep -qE "^  map: *(null)? *(#|$)" "$DETECTOR"; }; then
            field_map=$detector_dir/field_$field_kind.txt
        fi
        [[ -z $field_map ]] || settings+=("field.map=$field_map" "field.map_kind=$field_kind")
        image=$(realpath --relative-to="$(dirname "$DETECTOR")" "$REPO/scripts/acts/acts-base-48.2.0.sif")
        # the site file is rewritten from the template each time (new keys and comments), keeping the site's own
        # values (any section.key differing from the template); then each section.key=value replaces that line's
        # value (its comment kept) or is added to its section
        python - "$DETECTOR" "$detector_dir/sphenix.yaml" "acts.image=$image" ${settings[@]+"${settings[@]}"} <<'EOF'
import os, re, sys
path, template, settings = sys.argv[1], sys.argv[2], sys.argv[3:]


def entries(lines):
    """section.key -> value of each two-level setting."""
    found, section = {}, None
    for line in lines:
        if re.match(r"\w+:", line):
            section = line.split(":")[0]
        elif (m := re.match(r"  (\w+):\s*(.*?)\s*(#.*)?$", line)) and section:
            found[f"{section}.{m[1]}"] = m[2]
    return found


lines = open(template).read().splitlines()
if os.path.exists(path):
    defaults = entries(lines)
    kept = [f"{k}={v}" for k, v in entries(open(path).read().splitlines()).items() if v and v != defaults.get(k)]
    settings = settings[:1] + kept + settings[1:]
else:
    print(f"writing {path}")
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
        # the newest eval output under root_dir (<backbone>/<track head>/e<events>/); --csv <file> overrides it
        csv=$(find "$ROOT_DIR" -name '*_per_point_data.csv' -printf '%T@ %p\n' 2> /dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        [[ -n "$csv" ]] || { echo "no per-point CSV under $ROOT_DIR: run 'eval' first" >&2; exit 1; }
        echo "comparing on $csv"
        python "$REPO/scripts/acts/compare_vertexing.py" --csv "$csv" --detector "$DETECTOR" --tracks both "$@"
        ;;
    *)
        sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac
