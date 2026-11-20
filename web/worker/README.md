# Cloudflare Worker Bundle & Deployment (`web/worker/`)

`deploy_cf_worker.py` compiles static assets directly from `mips/camels_us_daily/` and
`web/portal/` and deploys the `hydrologymip-router` Cloudflare Worker (`https://hydrologymip.org`).

## Environment Variables

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `CLOUDFLARE_ACCOUNT_ID` | from `~/.config/hydrologymip/cloudflare.env` | Target Cloudflare account ID |
| `CLOUDFLARE_API_TOKEN` | from `~/.config/hydrologymip/cloudflare.env` | Workers write API token |
| `HYDROLOGYMIP_PREVIEW_GATE` | `"true"` | Set `"false"` when lifting the pre-launch password gate |
| `HYDROLOGYMIP_PREVIEW_ACCESS_KEY` | `"camels-mip-2026"` | Pre-launch gate password |
| `HYDROLOGYMIP_SCORER_API_URL` | `""` | Cloud Run scorer URL for `/api/preview/*` reverse proxy |
| `HYDROLOGYMIP_SCRUB_PRIVATE_URLS` | `"true"` | Strip pre-release private repo/bucket URLs from public JSONs |
| `HYDROMIP_EXPLORER_STORE` | unset | Path to `hydrograph_explorer_store.npz` (or `--explorer-store`) |

## Dry-Run & Live Deploy (**Maintainers Only**)

```bash
python -m web.worker.deploy_cf_worker --dry-run
python -m web.worker.deploy_cf_worker --explorer-store /path/to/hydrograph_explorer_store.npz
```
