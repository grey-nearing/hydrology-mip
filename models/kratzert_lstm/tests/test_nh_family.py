"""Tests for the NeuralHydrology-family trainers and runners (CudaLSTM, EA-LSTM, MC-LSTM, TFT).

Hermetic: a tiny synthetic CAMELS bundle (canonical schema) is written to `tmp_path`; nothing is
read from outside the repository.  The evaluation decade of the synthetic discharge is poisoned
with a huge constant so that any leakage of spin-up/evaluation discharge into training statistics
is detected immediately.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from models.camels_data import (
    BUNDLE_SCHEMA_VERSION,
    FORCING_VARIABLES,
    STATIC_ATTRIBUTE_NAMES_27,
    load_camels_slice,
    write_camels_bundle,
)
from models.hoedt_mclstm.model import MCLSTM, MCLSTMTrainer
from models.hoedt_mclstm.runner import precipitation_column_indices
from models.kratzert_lstm.model import CudaLSTM, KratzertLSTMTrainer
from models.kratzert_lstm.windows import (
    finite_window_mask,
    lr_for_epoch,
    mask_spinup_discharge,
    normalize_lr_schedule,
    nse_star_weights,
    resolve_runner_kwargs,
    target_stats,
    training_pairs,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_KEYS = ("kratzert_lstm", "kratzert_ealstm", "hoedt_mclstm", "rasiya_koya_tft")
BASINS = ("01013500", "01022500", "01030500", "01031500", "01047000", "01052500")
TRAIN_PERIOD = ("2000-10-01", "2004-09-30")
EVAL_PERIOD = ("1995-10-01", "1998-09-30")
# The whole Newman evaluation decade (which contains EVAL_PERIOD and the spin-up window of the
# author-protocol training period used by the reproduction scripts) is poisoned.
POISONED_PERIOD = ("1989-10-01", "1999-09-30")
POISON = 1.0e6


def _write_synthetic_bundle(root: Path) -> Path:
    dates = pd.date_range("1980-01-01", "2014-12-31", freq="D")
    n_b, n_t = len(BASINS), len(dates)
    rng = np.random.RandomState(0)

    def product() -> np.ndarray:
        arr = np.zeros((n_b, n_t, 5), dtype=np.float32)
        arr[:, :, 0] = rng.gamma(0.6, 4.0, size=(n_b, n_t))
        arr[:, :, 1] = rng.uniform(50, 400, size=(n_b, n_t))
        arr[:, :, 2] = rng.uniform(-5, 35, size=(n_b, n_t))
        arr[:, :, 3] = rng.uniform(-15, 20, size=(n_b, n_t))
        arr[:, :, 4] = rng.uniform(100, 3000, size=(n_b, n_t))
        return arr

    daymet = product()
    qobs = np.stack(
        [np.convolve(daymet[b, :, 0], np.ones(10) / 10.0, mode="same") * (0.5 + 0.1 * b) for b in range(n_b)]
    ).astype(np.float32)
    qobs[0, 7700:7800] = np.nan
    qobs[1, 8000:8010] = -999.0
    poisoned = (dates >= POISONED_PERIOD[0]) & (dates <= POISONED_PERIOD[1])
    qobs[:, poisoned] = POISON  # any leak into training statistics becomes glaring

    bundle = {
        "schema_version": np.asarray(BUNDLE_SCHEMA_VERSION),
        "basins": np.asarray(BASINS, dtype="<U8"),
        "dates": np.asarray([d.strftime("%Y-%m-%d") for d in dates], dtype="<U10"),
        "forcing_variable_names": np.asarray(FORCING_VARIABLES, dtype=str),
        "forcing_daymet": daymet,
        "forcing_maurer": product(),
        "forcing_nldas": product(),
        "qobs": qobs,
        "static_attributes": rng.normal(size=(n_b, 27)).astype(np.float32),
        "static_attribute_names": np.asarray(STATIC_ATTRIBUTE_NAMES_27, dtype=str),
        "gauge_lat": np.full(n_b, 45.0, dtype=np.float32),
        "gauge_lon": np.full(n_b, -70.0, dtype=np.float32),
        "elev_mean": np.full(n_b, 300.0, dtype=np.float32),
        "area_km2": np.full(n_b, 500.0, dtype=np.float64),
        "huc_02": np.asarray(["01"] * n_b, dtype="<U2"),
        "provenance": np.asarray("synthetic test bundle"),
    }
    return write_camels_bundle(bundle, root / "camels_671_1980_2014.npz", compress=False)


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("nh_family_bundle")
    _write_synthetic_bundle(root)
    return root


def _n_eval_days() -> int:
    return len(pd.date_range(EVAL_PERIOD[0], EVAL_PERIOD[1], freq="D"))


def _run_fold(model_key: str):
    module = __import__(f"models.{model_key}.runner", fromlist=["run_fold"])
    return module.run_fold


SMOKE_KWARGS = {
    "kratzert_lstm": {"hidden_size": 8, "seq_len": 30},
    "kratzert_ealstm": {"hidden_size": 8, "seq_len": 30},
    "hoedt_mclstm": {"hidden_size": 4, "seq_len": 30},
    "rasiya_koya_tft": {"hidden_size": 8, "seq_len": 20},
}


# --------------------------------------------------------------------------------------------
# windows.py unit tests
# --------------------------------------------------------------------------------------------
def test_lr_schedule_steps_follow_neuralhydrology_convention() -> None:
    sched = normalize_lr_schedule(None, 1e-3, {0: 1.0, 10: 0.5, 25: 0.1})
    assert sched == {0: 1e-3, 10: 5e-4, 25: 1e-4}
    assert [lr_for_epoch(sched, e) for e in (0, 9, 10, 24, 25, 29)] == [1e-3, 1e-3, 5e-4, 5e-4, 1e-4, 1e-4]
    explicit = normalize_lr_schedule({"0": 1e-2, "20": 5e-3, "25": 1e-3}, 1e-9, {0: 1.0})
    assert lr_for_epoch(explicit, 19) == 1e-2 and lr_for_epoch(explicit, 20) == 5e-3
    with pytest.raises(ValueError):
        normalize_lr_schedule({5: 1e-3}, 1e-3, {0: 1.0})


def test_mask_spinup_discharge_removes_sentinels_and_spinup_window() -> None:
    q = np.array([[-1.0, 2.0, 3.0, 4.0], [1.0, -999.0, 2.0, 0.0]], dtype=np.float32)
    masked = mask_spinup_discharge(q, 2)
    assert np.isnan(masked[:, :2]).all()
    assert masked[0, 2] == 3.0 and masked[1, 3] == 0.0
    assert np.isnan(mask_spinup_discharge(q, 0)[1, 1])
    with pytest.raises(ValueError):
        mask_spinup_discharge(q, 4)


def test_training_pairs_never_start_before_spinup_or_window_end() -> None:
    q = np.ones((1, 12), dtype=np.float32)
    ok = finite_window_mask(np.ones((1, 12), dtype=bool), seq_len=4)
    pairs = training_pairs(q, ok, seq_len=4, spinup_days=7)
    assert pairs[:, 1].min() == 7  # spin-up dominates seq_len - 1 = 3
    pairs = training_pairs(q, ok, seq_len=4, spinup_days=0)
    assert pairs[:, 1].min() == 3  # window length dominates
    # NaN-weight basins (fewer than two valid targets) are dropped entirely
    weights = np.array([np.nan], dtype=np.float32)
    assert training_pairs(q, ok, 4, 0, basin_weights=weights).shape[0] == 0


def test_finite_window_mask_requires_entire_window_finite() -> None:
    finite = np.array([[True, True, False, True, True, True]])
    assert finite_window_mask(finite, 2).tolist() == [[False, True, False, False, True, True]]
    assert finite_window_mask(finite, 1).tolist() == finite.tolist()


def test_target_stats_and_nse_star_weights_use_raw_mm_units() -> None:
    q = np.array([[1.0, 3.0, 5.0], [np.nan, 2.0, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    stats = target_stats(q)
    assert stats["y_mean"] == pytest.approx(np.mean([1.0, 3.0, 5.0, 2.0]))
    assert stats["basin_std_mm"][0] == pytest.approx(np.std([1.0, 3.0, 5.0]))
    assert np.isnan(stats["basin_std_mm"][1]) and np.isnan(stats["basin_std_mm"][2])
    w = nse_star_weights(stats["basin_std_mm"], eps=0.1)
    assert w[0] == pytest.approx(1.0 / (np.std([1.0, 3.0, 5.0]) + 0.1) ** 2)
    with pytest.raises(ValueError):
        target_stats(np.full((2, 3), np.nan, dtype=np.float32))


def test_resolve_runner_kwargs_rejects_unknown_names_but_allows_launcher_passthrough() -> None:
    defaults = {"epochs": 30, "seq_len": 365}
    resolved = resolve_runner_kwargs({"epochs": 1, "fold_id": "gauged", "fold_index": 0}, defaults)
    assert resolved == {"epochs": 1, "seq_len": 365}
    with pytest.raises(ValueError, match="n_epochs"):
        resolve_runner_kwargs({"n_epochs": 0}, defaults)


# --------------------------------------------------------------------------------------------
# architectures
# --------------------------------------------------------------------------------------------
def test_forget_gate_biases_are_initialized() -> None:
    lstm = CudaLSTM(5, 27, hidden_size=16, initial_forget_bias=3.0)
    assert torch.allclose(lstm.lstm.bias_hh_l0[16:32], torch.full((16,), 3.0))
    mclstm = MCLSTM(aux_dyn_dim=4, stat_in_dim=27, hidden_size=8, initial_forget_bias=-2.0)
    assert torch.allclose(mclstm.output_gate.fc.bias, torch.full((8,), -2.0))


def test_mclstm_precipitation_columns_are_selected_by_name() -> None:
    names = ["prcp(mm/day)_daymet", "srad(W/m2)_daymet", "tmax(C)_daymet", "tmin(C)_daymet", "vp(Pa)_daymet",
             "prcp(mm/day)_nldas", "srad(W/m2)_nldas", "tmax(C)_nldas", "tmin(C)_nldas", "vp(Pa)_nldas"]
    assert precipitation_column_indices(names) == [0, 5]
    with pytest.raises(ValueError):
        precipitation_column_indices(["srad(W/m2)_daymet"])


def test_mclstm_trainer_rejects_negative_mass_input_columns() -> None:
    trainer = MCLSTMTrainer(seq_len=5, epochs=0, hidden_size=4)
    x = np.random.RandomState(0).uniform(-5.0, 5.0, size=(2, 40, 5)).astype(np.float32)
    x[:, :, 0] = np.abs(x[:, :, 0])
    q = np.ones((2, 40), dtype=np.float32)
    s = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="negative"):
        trainer.fit_and_predict(x, s, q, x, s, n_target_days=30, spinup_days_train=4, prcp_col_indices=[2])


# --------------------------------------------------------------------------------------------
# trainer protocol guarantees
# --------------------------------------------------------------------------------------------
def test_trainer_requires_enough_evaluation_spinup() -> None:
    trainer = KratzertLSTMTrainer(seq_len=10, epochs=0, hidden_size=4)
    x_tr = np.zeros((2, 50, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="warm-up"):
        trainer.fit_and_predict(
            x_tr, np.zeros((2, 2), np.float32), np.ones((2, 50), np.float32),
            np.zeros((2, 20, 3), np.float32), np.zeros((2, 2), np.float32),
            n_target_days=15, spinup_days_train=9,
        )


def test_training_statistics_never_see_spinup_or_evaluation_discharge(
    bundle_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import models.hoedt_mclstm.model as mclstm_module
    import models.kratzert_lstm.model as lstm_module

    captured: list[float] = []
    original = lstm_module.target_stats

    def spy(q_window: np.ndarray):
        captured.append(float(np.nanmax(q_window)))
        return original(q_window)

    monkeypatch.setattr(lstm_module, "target_stats", spy)
    monkeypatch.setattr(mclstm_module, "target_stats", spy)

    for key in MODEL_KEYS:
        pred = _run_fold(key)(
            train_basins=list(BASINS), test_basins=list(BASINS[:2]), train_period=TRAIN_PERIOD,
            eval_period=EVAL_PERIOD, forcings=["daymet"], data_dir=bundle_dir, seeds=(1,),
            epochs=1, n_batches_per_epoch=2, **SMOKE_KWARGS[key],
        )
        assert pred.shape == (2, _n_eval_days())
    assert len(captured) == len(MODEL_KEYS)
    assert max(captured) < POISON / 1000.0, "evaluation-decade discharge leaked into target statistics"


def test_training_targets_lie_strictly_inside_train_period(bundle_dir: Path) -> None:
    seq_len, spinup = 30, 29
    train_slice = load_camels_slice(list(BASINS), TRAIN_PERIOD, ["daymet"], bundle_dir, spinup_days=spinup)
    q = mask_spinup_discharge(train_slice["qobs"], spinup)
    ok = finite_window_mask(np.all(np.isfinite(train_slice["x_dyn"]), axis=2), seq_len)
    pairs = training_pairs(q, ok, seq_len, spinup)
    dates = pd.DatetimeIndex(pd.to_datetime(train_slice["dates"]))
    target_dates = dates[np.unique(pairs[:, 1])]
    assert target_dates.min() == pd.Timestamp(TRAIN_PERIOD[0])
    assert target_dates.max() == pd.Timestamp(TRAIN_PERIOD[1])


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_windowed_inference_yields_time_varying_predictions_for_every_evaluation_day(
    model_key: str, bundle_dir: Path
) -> None:
    mean_pred, members = _run_fold(model_key)(
        train_basins=list(BASINS), test_basins=list(BASINS[:3]), train_period=TRAIN_PERIOD,
        eval_period=EVAL_PERIOD, forcings=["daymet", "maurer", "nldas"], data_dir=bundle_dir, seeds=(1, 2),
        epochs=1, n_batches_per_epoch=2, return_members=True, fold_id="gauged", fold_index=0,
        **SMOKE_KWARGS[model_key],
    )
    n_days = _n_eval_days()
    assert mean_pred.shape == (3, n_days) and members.shape == (2, 3, n_days)
    assert mean_pred.dtype == np.float32
    assert np.all(np.isfinite(mean_pred)) and np.all(mean_pred >= 0.0)
    assert np.allclose(mean_pred, members.mean(axis=0), atol=1e-5)
    # every evaluation day carries a genuine prediction
    for b in range(3):
        assert len(np.unique(np.round(mean_pred[b], 6))) > n_days // 2
    assert np.std(mean_pred[0, : n_days // 2]) > 0.0 and np.std(mean_pred[0, n_days // 2 :]) > 0.0


def test_single_seed_call_returns_that_member_without_averaging(bundle_dir: Path) -> None:
    run_fold = _run_fold("kratzert_lstm")
    common = dict(
        train_basins=list(BASINS), test_basins=list(BASINS[:2]), train_period=TRAIN_PERIOD,
        eval_period=EVAL_PERIOD, forcings=["daymet"], data_dir=bundle_dir, epochs=1, n_batches_per_epoch=2,
        **SMOKE_KWARGS["kratzert_lstm"],
    )
    single = run_fold(seeds=(5,), **common)
    again = run_fold(seeds=(5,), **common)
    assert np.array_equal(single, again)


def test_runners_reject_unknown_keyword_arguments(bundle_dir: Path) -> None:
    for key in MODEL_KEYS:
        with pytest.raises(ValueError, match="Unknown run_fold keyword"):
            _run_fold(key)(
                train_basins=list(BASINS), test_basins=list(BASINS), train_period=TRAIN_PERIOD,
                eval_period=EVAL_PERIOD, forcings=["daymet"], data_dir=bundle_dir, seeds=(1,), n_epochs=0,
            )


# --------------------------------------------------------------------------------------------
# reproduction scripts: hermetic and executable
# --------------------------------------------------------------------------------------------
FORBIDDEN_SOURCE_TOKENS = ("camels_leaderboard", "watershed_theory_discovery", "ensemble_smtslstm", "CLOUDSDK", "/tmp/")


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_reproduce_script_is_hermetic_and_trains_in_process(model_key: str, bundle_dir: Path) -> None:
    script = REPO_ROOT / "models" / model_key / "reproduce_author_metrics.py"
    source = script.read_text(encoding="utf-8")
    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in source, f"{script} references forbidden token {token!r}"
    assert "run_fold" in source and "from metrics.metrics import calculate_all_metrics" in source

    out_dir = bundle_dir / f"smoke_{model_key}"
    proc = subprocess.run(
        [sys.executable, "-m", f"models.{model_key}.reproduce_author_metrics", "--smoke", "--smoke-basins", "3",
         "--data-dir", str(bundle_dir), "--seeds", "1", "--output-dir", str(out_dir)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((out_dir / "stage1_smoke_results.json").read_text(encoding="utf-8"))
    assert payload["pred_nan_when_obs_valid"] == 0
    assert payload["ours_n_scored_basins"] == 3
    assert payload["protocol"]["runner_kwargs"]["epochs"] == 1
    assert "no precomputed hydrographs" in payload["provenance"]
    assert payload["status"] in {"PASS", "FAIL", "COMPLETED_NO_PUBLISHED_REFERENCE"}
    assert (out_dir / "stage1_author_reproduction_hydrograph.npz").is_file()


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_stage1_results_json_is_honest(model_key: str) -> None:
    payload = json.loads(
        (REPO_ROOT / "models" / model_key / "stage1_author_reproduction_results.json").read_text(encoding="utf-8")
    )
    assert payload["status"] in {"PASS", "FAIL", "NOT_RUN", "COMPLETED_NO_PUBLISHED_REFERENCE"}
    assert payload["model_key"] == model_key
    assert "provenance" in payload and "protocol" in payload
    if payload["status"] in {"PASS", "FAIL"}:
        assert "ours_median_nse" in payload and payload["pred_nan_when_obs_valid"] == 0
