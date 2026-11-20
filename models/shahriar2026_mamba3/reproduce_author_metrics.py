# Copyright 2026 Google LLC, Grey Nearing, and the CAMELS-US MIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 author-protocol confirmation for Shahriar (2026) Mamba-3 Adapter.

Supports both:
  1. Clean-room Split-3 training confirmation (`--mode train`, default): trains
     `MambaStreamflowModelAdapter` from scratch on all 671 CAMELS-US basins over
     `1980-10-01`..`1995-09-30` (Daymet `['dayl','prcp','srad','tmean','vp']`,
     `seed=111111`, 30 epochs x 361 batches/epoch) and evaluates over
     `1995-10-01`..`2010-09-30`, scoring both all 671 basins (published
     `Med NSE = 0.7272`) and the canonical 531 Newman basins (`Med NSE = 0.7307`),
     then writes consolidated Zarr v2 predictions (`mm/day`) to
     `gs://camels-us-mip/stage1_author_reproductions/shahriar2026_mamba3.zarr`.
  2. Direct Zenodo checkpoint/hydrograph verification (`--mode archive`): verifies
     Shahriar (2026)'s archived `pred_daily_m3s.npy` (`10.5281/zenodo.23087719`)
     converted cleanly from `m^3/s` to `mm/day` (`q_mm = q_m3s * 86.4 / area_km2`).
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
from pathlib import Path
import struct
import urllib.request
import zlib

import numpy as np
import pandas as pd
import xarray as xr

from metrics.metrics import calculate_metrics
from models.camels_data import DEFAULT_DATA_DIR, load_camels_slice
from models.run_model import DEFAULT_SPLITS_DIR, load_canonical_basins
from models.shahriar2026_mamba3.runner import DEFAULT_SEEDS, run_fold

MODEL_DIR = Path(__file__).resolve().parent
ZENODO_OUTPUT_ZIP_URL = "https://zenodo.org/api/records/23087719/files/Output.zip/content"
# Exact ZIP local-file offsets inside Zenodo 23087719 Output.zip for
# `Mamba3_3L/Tier1_Ep30_Variant-adapter_V3_head32_H80_L3_dConv4/All/`:
PRED_OFFSET = 1378395958
PRED_COMP_SIZE = 28018088
METRICS_CSV_OFFSET = 1365737651
METRICS_CSV_COMP_SIZE = 114232

AUTHOR_TRAIN_PERIOD = ("1980-10-01", "1995-09-30")
AUTHOR_EVAL_PERIOD = ("1995-10-01", "2010-09-30")
PUBLISHED_MED_NSE_671 = 0.7272
PUBLISHED_MED_NSE_531 = 0.7307
STAGE1_ZARR_URI = "gs://camels-us-mip/stage1_author_reproductions/shahriar2026_mamba3.zarr"


def _fetch_zip_member(url: str, local_header_offset: int, comp_size: int) -> bytes:
    end_byte = local_header_offset + 512 + comp_size
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/128.0.0.0",
            "Range": f"bytes={local_header_offset}-{end_byte}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        buf = resp.read()
    fname_len, extra_len = struct.unpack("<HH", buf[26:30])
    start = 30 + fname_len + extra_len
    return zlib.decompressobj(-15).decompress(buf[start : start + comp_size])


def score_per_basin_nse_kge(
    qsim: np.ndarray, qobs: np.ndarray, dates: np.ndarray
) -> dict[str, float]:
    """Compute summary NSE & KGE across basins using `metrics.metrics.calculate_metrics`."""
    nses: list[float] = []
    kges: list[float] = []
    t_coord = pd.DatetimeIndex(dates)
    for i in range(qsim.shape[0]):
        obs_i = qobs[i]
        sim_i = qsim[i]
        valid = np.isfinite(obs_i) & (obs_i >= 0.0) & np.isfinite(sim_i)
        if int(valid.sum()) < 30 or float(np.std(obs_i[valid])) < 1e-8:
            continue
        obs_da = xr.DataArray(obs_i, coords={"date": t_coord}, dims=["date"])
        sim_da = xr.DataArray(sim_i, coords={"date": t_coord}, dims=["date"])
        m = calculate_metrics(obs_da, sim_da, ["NSE", "KGE"])
        if np.isfinite(m["NSE"]):
            nses.append(float(m["NSE"]))
        if np.isfinite(m["KGE"]):
            kges.append(float(m["KGE"]))
    arr_nse = np.asarray(nses, dtype=np.float64)
    arr_kge = np.asarray(kges, dtype=np.float64)
    return {
        "num_valid_basins": int(arr_nse.size),
        "median_nse": float(np.median(arr_nse)),
        "mean_nse": float(np.mean(arr_nse)),
        "median_kge": float(np.median(arr_kge)),
        "nse_le_0": int(np.sum(arr_nse <= 0.0)),
    }


def write_stage1_zarr_v2(
    qsim_531_mm: np.ndarray,
    basins_531: list[str],
    dates: np.ndarray,
    target_uri: str,
    attrs: dict[str, object],
) -> None:
    """Write Stage-1 `(531, T)` daily predictions in `mm/day` as consolidated Zarr v2."""
    ds = xr.Dataset(
        data_vars={"qsim": (("basin", "time"), qsim_531_mm.astype(np.float32))},
        coords={
            "basin": np.asarray(basins_531, dtype="<U8"),
            "time": pd.DatetimeIndex(dates),
        },
        attrs=attrs,
    )
    zarr_kwargs: dict[str, object] = {"mode": "w", "consolidated": True}
    if "zarr_format" in inspect.signature(ds.to_zarr).parameters:
        zarr_kwargs["zarr_format"] = 2
    ds.to_zarr(target_uri, **zarr_kwargs)


def load_all_671_basins(data_dir: Path) -> list[str]:
    bundle = data_dir / "camels_671_1980_2014.npz" if data_dir.is_dir() else data_dir
    with np.load(bundle, allow_pickle=False) as npz:
        return [str(b).strip().zfill(8) for b in npz["basins"].tolist()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--mode", choices=("train", "archive", "both"), default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-zarr", type=str, default=STAGE1_ZARR_URI)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    basins_531 = load_canonical_basins(args.splits_dir)
    basins_671 = basins_531[:16] if args.smoke else load_all_671_basins(args.data_dir)
    eval_531 = basins_671 if args.smoke else basins_531

    #Note: 1980-01-01..1980-09-30 provides 274 days of pre-WY1981 Daymet spin-up in CAMELS v1.2.
    eval_slice_671 = load_camels_slice(
        basins=basins_671,
        period=AUTHOR_EVAL_PERIOD,
        forcings=["daymet"],
        data_dir=args.data_dir,
        spinup_days=0,
    )
    idx_531 = [basins_671.index(b) for b in eval_531]
    qobs_671 = eval_slice_671["target_qobs"]
    qobs_531 = qobs_671[idx_531]
    dates_eval = eval_slice_671["target_dates"]
    results: dict[str, object] = {
        "family_id": "shahriar2026_mamba3",
        "published_med_nse_671": PUBLISHED_MED_NSE_671,
        "published_med_nse_531": PUBLISHED_MED_NSE_531,
    }

    if args.mode in ("archive", "both") and not args.smoke:
        print("Verifying Zenodo 23087719 Tier-1 Ep30 pred_daily_m3s.npy...", flush=True)
        pred_m3s = np.load(io.BytesIO(_fetch_zip_member(ZENODO_OUTPUT_ZIP_URL, PRED_OFFSET, PRED_COMP_SIZE)))
        df_m = pd.read_csv(
            io.BytesIO(_fetch_zip_member(ZENODO_OUTPUT_ZIP_URL, METRICS_CSV_OFFSET, METRICS_CSV_COMP_SIZE))
        )
        z_gauges = [str(g).strip().zfill(8) for g in df_m["gauge_id"].tolist()]
        z_to_row = {g: i for i, g in enumerate(z_gauges)}
        rows_671 = [z_to_row[b] for b in basins_671]
        # Convert m^3/s -> mm/day using CAMELS GAGES-II catchment area (km^2):
        # (m^3/s * 86400 s/d) / (km^2 * 1e6 m^2/km^2) * 1000 mm/m = m^3/s * 86.4 / area_km2
        area_km2 = eval_slice_671["area_km2"][:, None]
        T = qobs_671.shape[1]
        pred_671_mm = (pred_m3s[rows_671, :T] * 86.4 / area_km2).astype(np.float32)
        arch_671 = score_per_basin_nse_kge(pred_671_mm, qobs_671, dates_eval)
        arch_531 = score_per_basin_nse_kge(pred_671_mm[idx_531], qobs_531, dates_eval)
        results["zenodo_archive_671"] = arch_671
        results["zenodo_archive_531"] = arch_531
        print(f"Zenodo archive 671: {arch_671}", flush=True)
        print(f"Zenodo archive 531: {arch_531}", flush=True)

    if args.mode in ("train", "both"):
        smoke_ov = (
            {"hidden_size": 32, "mamba_layers": 1, "epochs": 2, "n_batches_per_epoch": 4, "batch_size": 8, "spinup_days": 30, "rho": 60}
            if args.smoke
            else {"epochs": args.epochs, "spinup_days": 274}  # 1980-01-01..1980-09-30 available before 1980-10-01
        )
        qsim_mean_671, qsim_members_671 = run_fold(
            train_basins=basins_671,
            test_basins=basins_671,
            train_period=AUTHOR_TRAIN_PERIOD,
            eval_period=AUTHOR_EVAL_PERIOD,
            forcings=["daymet"],
            data_dir=args.data_dir,
            seeds=args.seeds,
            return_members=True,
            **smoke_ov,
        )
        rerun_671 = score_per_basin_nse_kge(qsim_mean_671, qobs_671, dates_eval)
        rerun_531 = score_per_basin_nse_kge(qsim_mean_671[idx_531], qobs_531, dates_eval)
        per_seed_531 = [
            score_per_basin_nse_kge(qsim_members_671[m, idx_531], qobs_531, dates_eval)[
                "median_nse"
            ]
            for m in range(qsim_members_671.shape[0])
        ]
        delta_531 = rerun_531["median_nse"] - PUBLISHED_MED_NSE_531
        results["clean_room_rerun_671"] = rerun_671
        results["clean_room_rerun_531"] = rerun_531
        results["per_seed_med_nse_531"] = per_seed_531
        results["delta_med_nse_531"] = round(float(delta_531), 4)
        results["status"] = "PASS" if abs(delta_531) <= 0.04 else "CHECK"
        print(f"Clean-room Split-3 671: {rerun_671}", flush=True)
        print(f"Clean-room Split-3 531: {rerun_531} (delta={delta_531:+.4f})", flush=True)

        if not args.no_upload and not args.smoke and args.output_zarr:
            print(f"Writing Stage-1 Zarr v2 (mm/day) to {args.output_zarr}...", flush=True)
            write_stage1_zarr_v2(
                qsim_531_mm=qsim_mean_671[idx_531],
                basins_531=basins_531,
                dates=dates_eval,
                target_uri=args.output_zarr,
                attrs={
                    "family_id": "shahriar2026_mamba3",
                    "model_name": "Mamba-3 Static-Dynamic Adapter",
                    "citation": "Shahriar (2026)",
                    "paper_url": "https://doi.org/10.2139/ssrn.7552815",
                    "code_url": "https://doi.org/10.5281/zenodo.23087719",
                    "train_period": f"{AUTHOR_TRAIN_PERIOD[0]}/{AUTHOR_TRAIN_PERIOD[1]}",
                    "test_period": f"{AUTHOR_EVAL_PERIOD[0]}/{AUTHOR_EVAL_PERIOD[1]}",
                    "forcings_used": "Daymet (dayl, prcp, srad, tmean, vp)",
                    "units": "mm/day",
                    "seeds": json.dumps([int(s) for s in args.seeds]),
                    "reproduced_med_nse_531": round(rerun_531["median_nse"], 4),
                    "reproduced_med_nse_671": round(rerun_671["median_nse"], 4),
                    "published_med_nse_531": PUBLISHED_MED_NSE_531,
                    "published_med_nse_671": PUBLISHED_MED_NSE_671,
                },
            )

    out_json = MODEL_DIR / "stage1_author_reproduction_results.json"
    if not args.smoke:
        out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
