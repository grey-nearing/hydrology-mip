#!/usr/bin/env bash
# Build and deploy the stateless HydrologyMIP preview scorer service to Cloud Run.
# Usage:
#   PROJECT_ID="my-gcp-project" REGION="us-central1" QOBS_NPZ_PATH="./web/scorer_api/data/qobs_531.npz" ./web/scorer_api/deploy_cloud_run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to target GCP project}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-hydromip-scorer}"
QOBS_NPZ_PATH="${QOBS_NPZ_PATH:-${SCRIPT_DIR}/data/qobs_531.npz}"

if [[ ! -f "${QOBS_NPZ_PATH}" ]]; then
  echo "ERROR: Missing canonical reference ${QOBS_NPZ_PATH}. Stage qobs_531.npz (or set QOBS_NPZ_PATH) before deploying." >&2
  exit 1
fi

BUILD_CTX="$(mktemp -d)"
cleanup() {
  rm -rf "${BUILD_CTX}"
}
trap cleanup EXIT

cp "${SCRIPT_DIR}/Dockerfile" "${BUILD_CTX}/Dockerfile"
cp "${SCRIPT_DIR}/.dockerignore" "${BUILD_CTX}/.dockerignore"
mkdir -p "${BUILD_CTX}/web/scorer_api" "${BUILD_CTX}/data" "${BUILD_CTX}/mips/camels_us_daily"

cp -R "${REPO_ROOT}/hydromip" "${BUILD_CTX}/hydromip"
cp -R "${REPO_ROOT}/metrics" "${BUILD_CTX}/metrics"
cp "${REPO_ROOT}/web/__init__.py" "${BUILD_CTX}/web/__init__.py"
cp "${SCRIPT_DIR}/__init__.py" "${BUILD_CTX}/web/scorer_api/__init__.py"
cp "${SCRIPT_DIR}/app.py" "${BUILD_CTX}/web/scorer_api/app.py"
cp "${SCRIPT_DIR}/requirements.txt" "${BUILD_CTX}/web/scorer_api/requirements.txt"

cp -R "${REPO_ROOT}/mips/camels_us_daily/leaderboards" "${BUILD_CTX}/mips/camels_us_daily/leaderboards"
cp -R "${REPO_ROOT}/mips/camels_us_daily/splits" "${BUILD_CTX}/mips/camels_us_daily/splits"
cp "${REPO_ROOT}/mips/camels_us_daily/protocol.json" "${BUILD_CTX}/mips/camels_us_daily/protocol.json"
cp "${QOBS_NPZ_PATH}" "${BUILD_CTX}/data/qobs_531.npz"

AR_REPO="${AR_REPO:-hydrologymip}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${SERVICE_NAME}:latest"

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
RUNTIME_SERVICE_ACCOUNT="${RUNTIME_SERVICE_ACCOUNT:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"

# Idempotently grant the Cloud Run runtime service account Vertex AI inference access.
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${RUNTIME_SERVICE_ACCOUNT}" \
  --role="roles/aiplatform.user" \
  --condition=None \
  >/dev/null

if ! gcloud artifacts repositories describe "${AR_REPO}" \
  --project="${PROJECT_ID}" --location="${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${AR_REPO}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --repository-format=docker \
    --description="HydrologyMIP stateless scorer container images"
fi

gcloud builds submit "${BUILD_CTX}" \
  --project="${PROJECT_ID}" \
  --tag="${IMAGE_URI}"

VERTEX_AI_LOCATION="${VERTEX_AI_LOCATION:-${REGION}}"
HYDROMIP_LLM_MODEL="${HYDROMIP_LLM_MODEL:-gemini-2.5-flash}"
HYDROMIP_MAX_LLM_CALLS_PER_DAY="${HYDROMIP_MAX_LLM_CALLS_PER_DAY:-100}"
HYDROMIP_MAX_LLM_CALLS_PER_IP_HOUR="${HYDROMIP_MAX_LLM_CALLS_PER_IP_HOUR:-3}"

ENV_VARS="VERTEX_AI_PROJECT=${PROJECT_ID},VERTEX_AI_LOCATION=${VERTEX_AI_LOCATION},HYDROMIP_LLM_MODEL=${HYDROMIP_LLM_MODEL},HYDROMIP_MAX_LLM_CALLS_PER_DAY=${HYDROMIP_MAX_LLM_CALLS_PER_DAY},HYDROMIP_MAX_LLM_CALLS_PER_IP_HOUR=${HYDROMIP_MAX_LLM_CALLS_PER_IP_HOUR}"
if [[ -n "${GEMINI_API_KEY:-}" ]]; then
  ENV_VARS="${ENV_VARS},GEMINI_API_KEY=${GEMINI_API_KEY}"
fi

DEPLOY_ARGS=(
  run deploy "${SERVICE_NAME}"
  --project="${PROJECT_ID}"
  --region="${REGION}"
  --image="${IMAGE_URI}"
  --service-account="${RUNTIME_SERVICE_ACCOUNT}"
  --platform=managed
  --allow-unauthenticated
  --cpu=2
  --memory=4Gi
  --concurrency=8
  --timeout=300
  --session-affinity
  --min-instances=0
  --max-instances=3
  --set-env-vars="${ENV_VARS}"
)

gcloud "${DEPLOY_ARGS[@]}"

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --project="${PROJECT_ID}" --region="${REGION}" --format='value(status.url)')"
echo "Deployed ${SERVICE_NAME}: ${SERVICE_URL}"
echo "Configure Worker env/secret: SCORER_API_URL=${SERVICE_URL}"
