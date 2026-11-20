#!/usr/bin/env bash
# Triggered via git pre-commit hook or maintenance workflow to synchronize
# `mips/<mip_id>/leaderboards/tables_3_to_14.json` and regenerate manuscript/portal
# figures in `mips/<mip_id>/figures/` directly from `<leaderboard_id>.json`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${1:-}" == "--install-hook" ]]; then
    HOOK_PATH="${REPO_ROOT}/.git/hooks/pre-commit"
    cat > "${HOOK_PATH}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
if git diff --cached --name-only | grep -Eq '^(mips/[^/]+/(leaderboards|protocol\.json)|leaderboards/)'; then
    "${REPO_ROOT}/leaderboards/on_community_commit.sh"
    for mip_proto in "${REPO_ROOT}"/mips/*/protocol.json; do
        mip_dir="$(dirname "${mip_proto}")"
        if [[ -d "${mip_dir}/leaderboards" ]]; then
            git add "${mip_dir}/leaderboards/tables_3_to_14.json"
        fi
        if [[ -d "${mip_dir}/figures" ]]; then
            git add "${mip_dir}/figures/"
        fi
    done
fi
EOF
    chmod +x "${HOOK_PATH}"
    echo "Installed multi-MIP pre-commit hook at ${HOOK_PATH}"
    exit 0
fi

# Interpreter selection: honour an explicit PYTHON_BIN, otherwise use the first python3 on PATH.
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "ERROR: no 'python3' found on PATH and PYTHON_BIN is not set." >&2
    echo "       Set PYTHON_BIN=/path/to/python (an interpreter with this project's dependencies installed)." >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: PYTHON_BIN='${PYTHON_BIN}' is not an executable interpreter." >&2
    exit 1
fi
if ! import_error="$("${PYTHON_BIN}" -c 'import hydromip, leaderboards' 2>&1)"; then
    echo "ERROR: '${PYTHON_BIN}' cannot import 'hydromip'/'leaderboards' from ${REPO_ROOT}:" >&2
    echo "${import_error}" >&2
    echo "       Install the project and its dependencies into that interpreter" >&2
    echo "       (e.g. '${PYTHON_BIN} -m pip install -e \"${REPO_ROOT}\"') or point PYTHON_BIN at one that has them." >&2
    exit 1
fi

if [[ -n "${MIP_ID:-}" ]]; then
    MIP_IDS=("${MIP_ID}")
else
    mapfile -t MIP_IDS < <(
        "${PYTHON_BIN}" -c 'from hydromip.protocol import discover_mips; print("\n".join(sorted(discover_mips().keys())))'
    )
fi

for mip in "${MIP_IDS[@]}"; do
    "${PYTHON_BIN}" - "${mip}" <<'PY'
import sys
from hydromip.paths import resolve_mip_paths
from leaderboards.update_leaderboard import sync_master_tables_from_per_leaderboard_files

mip_id = sys.argv[1]
lb_dir = resolve_mip_paths(mip_id).leaderboards_dir
if lb_dir.is_dir():
    sync_master_tables_from_per_leaderboard_files(lb_dir)
PY
    if [[ -d "${REPO_ROOT}/mips/${mip}/leaderboards" ]]; then
        "${PYTHON_BIN}" -m leaderboards.generate_figures --mip "${mip}"
    fi
done
