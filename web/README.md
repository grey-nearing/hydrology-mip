# HydrologyMIP Web Platform (`web/`)

The unified web portal, stateless preview scoring API, and Cloudflare Workers deployer for
<https://hydrologymip.org>.

| Folder / File | Purpose | Docs / Commands |
| :--- | :--- | :--- |
| `web/portal/` | Static HTML/CSS/JS frontend (14 Gauch metrics + 4-step Contribute wizard) | Deployed directly as Workers Static Assets |
| `web/scorer_api/` | Stateless FastAPI `/api/preview/{inspect,score}` + multi-MIP `/api/{mips,splits,figures}` | `uvicorn web.scorer_api.app:app --port 8080` |
| `web/worker/` | Standalone Cloudflare Worker bundler + deployer (`deploy_cf_worker.py`) | `python -m web.worker.deploy_cf_worker --dry-run` |
| `web/tests/` | Unit & integration tests for ingestion, scorer API, CI processor, and Worker bundle | `pytest web/tests` |

## Prerequisites & Commands (Run from Repository Root)

1. Install portal & test dependencies (`pip install -e ".[dev]"`).
2. Stage `web/scorer_api/data/qobs_531.npz` (never committed to Git) or set `HYDROMIP_QOBS_PATH`.
3. Build & validate the static Cloudflare bundle locally without uploading:

```bash
python -m web.worker.deploy_cf_worker --dry-run
```

4. **Maintainers only:** Deploy to Cloudflare Workers with credentials in `CLOUDFLARE_ACCOUNT_ID` /
   `CLOUDFLARE_API_TOKEN` (or `~/.config/hydrologymip/cloudflare.env`):

```bash
HYDROLOGYMIP_PREVIEW_GATE=true python -m web.worker.deploy_cf_worker \
  --explorer-store /path/to/hydrograph_explorer_store.npz
```

Set `HYDROLOGYMIP_PREVIEW_GATE=false` for unauthenticated public launch and
`HYDROLOGYMIP_SCORER_API_URL=https://<cloud-run-service>` to enable live browser preview scoring.

## Public Archive & Telemetry Endpoints

- Ongoing public hydrograph & target Zarr mirror (R2):
  <https://data.hydrologymip.org/camels_us_daily/> (alongside Zenodo v1.0 release
  [10.5281/zenodo.24383448](https://doi.org/10.5281/zenodo.24383448)).
- Scorer readiness & LLM backend telemetry: `GET /api/preview/health` (proxied from the Worker to
  Cloud Run, forwarding `GAESA` sticky cookies while stripping preview gate cookies).
