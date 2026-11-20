#!/usr/bin/env bash
# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Turnkey post-cleanup script to activate owner-approval (@grey-nearing) + required
# `CI Gate` status checks on `main` once `grey-nearing/camels-us-mip` goes public
# (or if the owner account is upgraded to GitHub Pro).
#
# Usage:
#   ./hydromip/tools/enable_public_launch_guardrails.sh [--enforce-admins true|false] [--repo owner/name]

set -euo pipefail

REPO="grey-nearing/camels-us-mip"
ENFORCE_ADMINS="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --enforce-admins)
      if [[ $# -lt 2 || ( "$2" != "true" && "$2" != "false" ) ]]; then
        echo "ERROR: --enforce-admins requires 'true' or 'false'." >&2
        exit 2
      fi
      ENFORCE_ADMINS="$2"
      shift 2
      ;;
    --repo)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "ERROR: --repo requires 'owner/name'." >&2
        exit 2
      fi
      REPO="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument '$1'." >&2
      exit 2
      ;;
  esac
done

VISIBILITY="$(gh api -X GET "repos/${REPO}" --jq '.visibility')"
MAIN_SHA="$(gh api -X GET "repos/${REPO}/branches/main" --jq '.commit.sha')"
echo "Configuring guardrails on ${REPO} (visibility=${VISIBILITY}, main=${MAIN_SHA:0:7}, enforce_admins=${ENFORCE_ADMINS})..."

# Verify `CI Gate` has completed successfully on `main` so branch protection never locks on a missing check name.
CI_GATE_CONCLUSIONS="$(gh api -X GET "repos/${REPO}/commits/${MAIN_SHA}/check-runs" --jq '[.check_runs[] | select(.name == "CI Gate") | .conclusion] | join(",")')"
if [[ "${CI_GATE_CONCLUSIONS}" != *"success"* ]]; then
  echo "ERROR: Required check 'CI Gate' has not yet succeeded on main (${MAIN_SHA:0:7}; found='${CI_GATE_CONCLUSIONS}')." >&2
  echo "       Push the final cleanup commit to main and wait for the 'CI' workflow to turn green before locking main." >&2
  exit 1
fi

# 1. Enforce read-only default GITHUB_TOKEN permissions and forbid Actions from approving PR reviews.
gh api -X PUT "repos/${REPO}/actions/permissions/workflow" \
  -H "Accept: application/vnd.github+json" \
  --input - <<'JSON'
{
  "default_workflow_permissions": "read",
  "can_approve_pull_request_reviews": false
}
JSON

# 2. Configure fork PR workflow approval policy (public repositories only).
if [[ "${VISIBILITY}" == "public" ]]; then
  gh api -X PUT "repos/${REPO}/actions/permissions/fork-pr-contributor-approval" \
    -H "Accept: application/vnd.github+json" \
    -f approval_policy="first_time_contributors"
fi

# 3. Activate `main` branch protection requiring `CI Gate` + `@grey-nearing` Code Owner approval.
gh api -X PUT "repos/${REPO}/branches/main/protection" \
  -H "Accept: application/vnd.github+json" \
  --input - <<JSON
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "CI Gate"
    ]
  },
  "enforce_admins": ${ENFORCE_ADMINS},
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "required_approving_review_count": 1,
    "require_last_push_approval": true
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": true,
  "lock_branch": false,
  "allow_fork_syncing": true
}
JSON

echo "=== VERIFIED LIVE BRANCH PROTECTION (READ-ONLY GET) ==="
gh api -X GET "repos/${REPO}/branches/main/protection" \
  --jq '{
    enforce_admins: .enforce_admins.enabled,
    strict_status_checks: .required_status_checks.strict,
    required_contexts: .required_status_checks.contexts,
    require_code_owner_reviews: .required_pull_request_reviews.require_code_owner_reviews,
    required_approving_review_count: .required_pull_request_reviews.required_approving_review_count,
    dismiss_stale_reviews: .required_pull_request_reviews.dismiss_stale_reviews,
    require_last_push_approval: .required_pull_request_reviews.require_last_push_approval,
    required_conversation_resolution: .required_conversation_resolution.enabled,
    allow_force_pushes: .allow_force_pushes.enabled,
    allow_deletions: .allow_deletions.enabled
  }'
