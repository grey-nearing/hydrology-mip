#!/usr/bin/env bash
# Triggered on each new community commit (via git pre-commit hook or CI workflow)
# to synchronize `leaderboards/tables_3_to_14.json` and regenerate all manuscript
# figures in `figures/` directly from `leaderboards/*.json`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${1:-}" == "--install-hook" ]]; then
    HOOK_PATH="${REPO_ROOT}/.git/hooks/pre-commit"
    cat > "${HOOK_PATH}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
if git diff --cached --name-only | grep -Eq '^(leaderboards|submissions)/'; then
    "${REPO_ROOT}/leaderboards/on_community_commit.sh"
    git add "${REPO_ROOT}/leaderboards/tables_3_to_14.json" "${REPO_ROOT}/figures/"
fi
EOF
    chmod +x "${HOOK_PATH}"
    echo "Installed pre-commit hook at ${HOOK_PATH}"
    exit 0
fi

# Interpreter selection: honour an explicit PYTHON_BIN, otherwise use the first python3 on PATH.
# No machine-specific interpreter paths are hard-coded here.
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
if ! import_error="$("${PYTHON_BIN}" -c 'import leaderboards' 2>&1)"; then
    echo "ERROR: '${PYTHON_BIN}' cannot import the 'leaderboards' package from ${REPO_ROOT}:" >&2
    echo "${import_error}" >&2
    echo "       Install the project and its dependencies into that interpreter" >&2
    echo "       (e.g. '${PYTHON_BIN} -m pip install -e \"${REPO_ROOT}\"') or point PYTHON_BIN at one that has them." >&2
    exit 1
fi

"${PYTHON_BIN}" -c "from leaderboards.update_leaderboard import sync_master_tables_from_per_leaderboard_files; sync_master_tables_from_per_leaderboard_files()"
"${PYTHON_BIN}" -m leaderboards.generate_figures
