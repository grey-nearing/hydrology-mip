# Copyright 2025-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
#
# Portions of this file are derived from neuralhydrology/evaluation/metrics.py
# (https://github.com/neuralhydrology/neuralhydrology), Copyright (c) 2021,
# NeuralHydrology, and are used under the BSD 3-Clause License. See
# THIRD_PARTY_NOTICES.md for the full license text.

"""Evaluation metrics for observed and simulated discharge time series.

Implements the exact 14 hydrological evaluation metrics from Gauch et al. (2023)
("In Defense of Metrics: Metrics Sufficiently Encode Typical Human Preferences
Regarding Hydrological Model Performance", Water Resources Research) and
NeuralHydrology (Kratzert et al., 2022).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from scipy import signal, stats
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

LOGGER = logging.getLogger(__name__)

# Pearson's r (and therefore the KGE) is undefined for fewer than two samples.
_MIN_SAMPLES_FOR_CORRELATION = 2
# The KGE weights one term each for the correlation, alpha and beta.
_NUM_KGE_WEIGHTS = 3
# The FLV needs at least two low-flow values; a single value sums to zero.
_MIN_LOW_FLOW_SAMPLES = 2

# Pandas 2.2+ uses 'YE', 'ME', 'QE' for year/month/quarter-end offset aliases.
_PANDAS_MAJOR_MINOR = tuple(
    int(part) for part in pd.__version__.split('.')[:2] if part.isdigit()
)
_YE_FREQ = 'YE' if _PANDAS_MAJOR_MINOR >= (2, 2) else 'Y'
_ME_FREQ = 'ME' if _PANDAS_MAJOR_MINOR >= (2, 2) else 'M'
_QE_FREQ = 'QE' if _PANDAS_MAJOR_MINOR >= (2, 2) else 'Q'
_TIMEDELTA_COMPATIBLE_UNITS = frozenset(
    {'W', 'D', 'h', 'H', 'min', 'T', 's', 'S', 'ms', 'L', 'us', 'U', 'ns', 'N'}
)


class AllNaNError(Exception):
    """Raised by `calculate_(all_)metrics` if all observations or all simulations are NaN."""


def infer_datetime_coord(xr_obj: DataArray | Dataset) -> str:
    """Checks for coordinate with 'date' in its name and returns the name.

    Args:
        xr_obj: DataArray or Dataset to infer coordinate name of.

    Returns:
        Name of datetime coordinate.

    Raises:
        RuntimeError: If none or multiple coordinates with 'date' in its name are found.
    """
    candidates = [c for c in list(xr_obj.coords) if 'date' in c]
    if len(candidates) > 1:
        raise RuntimeError(
            "Found multiple coordinates with 'date' in its name."
        )
    if not candidates:
        raise RuntimeError(
            "Did not find any coordinate with 'date' in its name"
        )

    return candidates[0]


def get_frequency_factor(freq_one: str, freq_two: str) -> float:
    """Get relative factor between the two frequencies.

    Args:
        freq_one: String representation of the first frequency.
        freq_two: String representation of the second frequency.

    Returns:
        Ratio of `freq_one` to `freq_two`.

    Raises:
        NotImplementedError: If either frequency is negative.
        ValueError: If the frequency factor cannot be determined because the
            frequencies do not represent comparable time deltas.
    """
    if freq_one == freq_two:
        return 1

    offset_one = to_offset(freq_one)
    offset_two = to_offset(freq_two)
    if offset_one.n < 0 or offset_two.n < 0:
        raise NotImplementedError('Cannot compare negative frequencies.')
    if offset_one.n == offset_two.n == 0:
        return 1
    if offset_two.n == 0:
        return np.inf
    if offset_one.name == offset_two.name:
        return offset_one.n / offset_two.n

    factor = None
    regex_month_or_day = (
        '-(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|'
        'MON|TUE|WED|THU|FRI|SAT|SUN)$'
    )
    for i, (one, two) in enumerate(
        [(offset_one, offset_two), (offset_two, offset_one)]
    ):
        name_one = re.sub(regex_month_or_day, '', one.name)
        name_two = re.sub(regex_month_or_day, '', two.name)
        if (name_one in ['A', _YE_FREQ] and name_two == _ME_FREQ) or (
            name_one in ['AS', 'YS'] and name_two == 'MS'
        ):
            factor = 12 * one.n / two.n
        if (name_one in ['A', _YE_FREQ] and name_two == _QE_FREQ) or (
            name_one in ['AS', 'YS'] and name_two == 'QS'
        ):
            factor = 4 * one.n / two.n
        if (name_one == _QE_FREQ and name_two == _ME_FREQ) or (
            name_one == 'QS' and name_two == 'MS'
        ):
            factor = 3 * one.n / two.n
        if name_one == 'W' and name_two == 'D':
            factor = 7 * one.n / two.n

        if factor is not None:
            if i == 1:
                return 1 / factor
            return factor

    base_one = re.sub(regex_month_or_day, '', offset_one.name)
    base_two = re.sub(regex_month_or_day, '', offset_two.name)
    if (
        base_one in ['M', 'Y', 'A', 'y', 'ME', 'YE']
        or base_two in ['M', 'Y', 'A', 'y', 'ME', 'YE']
        or base_one not in _TIMEDELTA_COMPATIBLE_UNITS
        or base_two not in _TIMEDELTA_COMPATIBLE_UNITS
    ):
        raise ValueError(
            f'Frequencies {freq_one} and/or {freq_two} are not comparable.'
        )

    return float(pd.to_timedelta(freq_one) / pd.to_timedelta(freq_two))


def get_available_metrics() -> list[str]:
    """Get list of available metrics.

    Returns:
        List of implemented metric names.
    """
    return list(_METRIC_FUNCTIONS)


def _validate_inputs(obs: DataArray, sim: DataArray) -> None:
    if obs.shape != sim.shape:
        raise RuntimeError('Shapes of observations and simulations must match')

    if (len(obs.shape) > 1) and (obs.shape[1] > 1):
        raise RuntimeError(
            'Metrics only defined for time series '
            '(1d or 2d with second dimension 1)'
        )


def _mask_valid(obs: DataArray, sim: DataArray) -> tuple[DataArray, DataArray]:
    # mask of invalid entries. NaNs in simulations can happen during
    # validation/testing
    obs_null = np.isnan(np.asarray(obs.values))
    sim_null = np.isnan(np.asarray(sim.values))
    if not (obs_null.any() or sim_null.any()):
        return obs, sim
    idx = (~sim.isnull()) & (~obs.isnull())

    obs = obs[idx]
    sim = sim[idx]

    return obs, sim


def _mask_valid_arrays(
    obs: DataArray, sim: DataArray
) -> tuple[np.ndarray, np.ndarray]:
    obs_vals = np.asarray(obs.values)
    sim_vals = np.asarray(sim.values)
    idx = (~np.isnan(sim_vals)) & (~np.isnan(obs_vals))
    return obs_vals[idx], sim_vals[idx]


def _get_fdc_from_array(vals: np.ndarray) -> np.ndarray:
    nan_mask = np.isnan(vals)
    if nan_mask.any():
        valid = np.sort(vals[~nan_mask])[::-1]
        nans = np.full(int(nan_mask.sum()), np.nan, dtype=vals.dtype)
        return np.concatenate([valid, nans])
    return np.sort(vals)[::-1]


def _get_fdc(da: DataArray) -> np.ndarray:
    return _get_fdc_from_array(np.asarray(da.values))


def nse(obs: DataArray, sim: DataArray) -> float:
    r"""Calculate Nash-Sutcliffe Efficiency [#]_.

    Nash-Sutcliffe Efficiency is the R-square between observed and simulated
    discharge.

    .. math::
        \text{NSE} = 1 - \frac{\sum_{t=1}^{T}(Q_m^t - Q_o^t)^2}
        {\sum_{t=1}^T(Q_o^t - \overline{Q}_o)^2},

    where :math:`Q_m` are the simulations (here, `sim`) and :math:`Q_o` are
    observations (here, `obs`).

    Args:
        obs: Observed time series.
        sim: Simulated time series.

    Returns:
        Nash-Sutcliffe Efficiency.

    References:
        .. [#] Nash, J. E.; Sutcliffe, J. V. (1970). "River flow forecasting
            through conceptual models part I - A discussion of principles".
            Journal of Hydrology. 10 (3): 282-290.
            doi:10.1016/0022-1694(70)90255-6.
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs_v, sim_v = _mask_valid_arrays(obs, sim)
    if obs_v.size == 0:
        return np.nan

    denominator = float(np.sum((obs_v - np.mean(obs_v)) ** 2))
    if denominator == 0.0:
        return np.nan
    numerator = float(np.sum((sim_v - obs_v) ** 2))

    value = 1.0 - numerator / denominator

    return float(value)


def mse(obs: DataArray, sim: DataArray) -> float:
    r"""Calculate mean squared error.

    .. math:: \text{MSE} = \frac{1}{T}\sum_{t=1}^T (\widehat{y}_t - y_t)^2,

    where :math:`\widehat{y}` are the simulations (here, `sim`) and :math:`y`
    are observations (here, `obs`).

    Args:
        obs: Observed time series.
        sim: Simulated time series.

    Returns:
        Mean squared error.
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs_v, sim_v = _mask_valid_arrays(obs, sim)
    if obs_v.size == 0:
        return np.nan

    return float(np.mean((sim_v - obs_v) ** 2))


def rmse(obs: DataArray, sim: DataArray) -> float:
    r"""Calculate root mean squared error.

    .. math::
        \text{RMSE} = \sqrt{\frac{1}{T}\sum_{t=1}^T (\widehat{y}_t - y_t)^2},

    where :math:`\widehat{y}` are the simulations (here, `sim`) and :math:`y`
    are observations (here, `obs`).

    Args:
        obs: Observed time series.
        sim: Simulated time series.

    Returns:
        Root mean squared error.
    """
    return float(np.sqrt(mse(obs, sim)))


def alpha_nse(obs: DataArray, sim: DataArray) -> float:
    r"""Calculate the alpha NSE decomposition [#]_.

    The alpha NSE decomposition is the fraction of the standard deviations of
    simulations and observations.

    .. math:: \alpha = \frac{\sigma_s}{\sigma_o},

    where :math:`\sigma_s` is the standard deviation of the simulations (here,
    `sim`) and :math:`\sigma_o` is the standard deviation of the observations
    (here, `obs`).

    Args:
        obs: Observed time series.
        sim: Simulated time series.

    Returns:
        Alpha NSE decomposition.

    References:
        .. [#] Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F.
            (2009). Decomposition of the mean squared error and NSE
            performance criteria: Implications for improving hydrological
            modelling. Journal of hydrology, 377(1-2), 80-91.
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs_v, sim_v = _mask_valid_arrays(obs, sim)
    if obs_v.size == 0:
        return np.nan
    obs_std = float(np.std(obs_v))
    if obs_std == 0.0:
        return np.nan

    return float(np.std(sim_v) / obs_std)


def beta_nse(obs: DataArray, sim: DataArray) -> float:
    r"""Calculate the beta NSE decomposition [#]_.

    The beta NSE decomposition is the difference of the mean simulation and
    mean observation divided by the standard deviation of the observations.

    .. math:: \beta = \frac{\mu_s - \mu_o}{\sigma_o},

    where :math:`\mu_s` is the mean of the simulations (here, `sim`),
    :math:`\mu_o` is the mean of the observations (here, `obs`) and
    :math:`\sigma_o` the standard deviation of the observations.

    Args:
        obs: Observed time series.
        sim: Simulated time series.

    Returns:
        Beta NSE decomposition.

    References:
        .. [#] Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F.
            (2009). Decomposition of the mean squared error and NSE
            performance criteria: Implications for improving hydrological
            modelling. Journal of hydrology, 377(1-2), 80-91.
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs_v, sim_v = _mask_valid_arrays(obs, sim)
    if obs_v.size == 0:
        return np.nan
    obs_std = float(np.std(obs_v))
    if obs_std == 0.0:
        return np.nan

    return float((np.mean(sim_v) - np.mean(obs_v)) / obs_std)


def beta_kge(obs: DataArray, sim: DataArray) -> float:
    r"""Calculate the beta KGE term [#]_.

    The beta term of the Kling-Gupta Efficiency is defined as the fraction of
    the means.

    .. math:: \beta_{\text{KGE}} = \frac{\mu_s}{\mu_o},

    where :math:`\mu_s` is the mean of the simulations (here, `sim`) and
    :math:`\mu_o` is the mean of the observations (here, `obs`).

    Args:
        obs: Observed time series.
        sim: Simulated time series.

    Returns:
        Beta KGE term.

    References:
        .. [#] Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F.
            (2009). Decomposition of the mean squared error and NSE
            performance criteria: Implications for improving hydrological
            modelling. Journal of hydrology, 377(1-2), 80-91.
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs_v, sim_v = _mask_valid_arrays(obs, sim)
    if obs_v.size == 0:
        return np.nan
    obs_mean = float(np.mean(obs_v))
    if obs_mean == 0.0:
        return np.nan

    return float(np.mean(sim_v) / obs_mean)


def kge(
    obs: DataArray,
    sim: DataArray,
    weights: list[float] | None = None,
) -> float:
    r"""Calculate the Kling-Gupta Efficiency [#]_.

    .. math::
        \text{KGE} = 1 - \sqrt{[ s_r (r - 1)]^2 + [s_\alpha ( \alpha - 1)]^2 +
            [s_\beta(\beta_{\text{KGE}} - 1)]^2},

    where :math:`r` is the correlation coefficient, :math:`\alpha` the
    :math:`\alpha`-NSE decomposition, :math:`\beta_{\text{KGE}}` the fraction
    of the means and :math:`s_r, s_\alpha, s_\beta` the corresponding weights
    (here the three float values in the `weights` parameter).

    Args:
        obs: Observed time series.
        sim: Simulated time series.
        weights: Weighting factors of the 3 KGE parts, by default each part
            has a weight of 1.

    Returns:
        Kling-Gupta Efficiency.

    References:
        .. [#] Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F.
            (2009). Decomposition of the mean squared error and NSE
            performance criteria: Implications for improving hydrological
            modelling. Journal of hydrology, 377(1-2), 80-91.
    """
    if weights is None:
        weights = [1.0, 1.0, 1.0]
    if len(weights) != _NUM_KGE_WEIGHTS:
        raise ValueError('Weights of the KGE must be a list of three values')

    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs_v, sim_v = _mask_valid_arrays(obs, sim)

    if len(obs_v) < _MIN_SAMPLES_FOR_CORRELATION:
        return np.nan

    obs_vals = np.asarray(obs_v, dtype=np.float64)
    sim_vals = np.asarray(sim_v, dtype=np.float64)
    obs_std = float(np.std(obs_vals))
    sim_std = float(np.std(sim_vals))
    obs_mean = float(np.mean(obs_vals))
    if obs_std == 0.0 or sim_std == 0.0 or obs_mean == 0.0:
        return np.nan

    r, _ = stats.pearsonr(obs_vals, sim_vals)

    alpha = sim_std / obs_std
    beta = float(np.mean(sim_vals)) / obs_mean

    value = (
        weights[0] * (r - 1) ** 2
        + weights[1] * (alpha - 1) ** 2
        + weights[2] * (beta - 1) ** 2
    )

    return float(1 - np.sqrt(float(value)))


def pearsonr(obs: DataArray, sim: DataArray) -> float:
    """Calculate pearson correlation coefficient (using scipy.stats.pearsonr).

    Args:
        obs: Observed time series.
        sim: Simulated time series.

    Returns:
        Pearson correlation coefficient.
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs_v, sim_v = _mask_valid_arrays(obs, sim)

    if len(obs_v) < _MIN_SAMPLES_FOR_CORRELATION:
        return np.nan

    obs_vals = np.asarray(obs_v, dtype=np.float64)
    sim_vals = np.asarray(sim_v, dtype=np.float64)
    if np.std(obs_vals) == 0.0 or np.std(sim_vals) == 0.0:
        return np.nan

    r, _ = stats.pearsonr(obs_vals, sim_vals)

    return float(r)


def fdc_fms(
    obs: DataArray, sim: DataArray, lower: float = 0.2, upper: float = 0.7
) -> float:
    r"""Calculate the middle-section slope of the flow duration curve [#]_.

    .. math::
        \%\text{BiasFMS} = \frac{
        \left | \log(Q_{s,\text{lower}}) - \log(Q_{s,\text{upper}}) \right | -
        \left | \log(Q_{o,\text{lower}}) - \log(Q_{o,\text{upper}}) \right |}
        {\left | \log(Q_{s,\text{lower}}) - \log(Q_{s,\text{upper}}) \right |}
        \times 100,

    where :math:`Q_{s,\text{lower/upper}}` corresponds to the FDC of the
    simulations (here, `sim`) at the `lower` and `upper` bound of the middle
    section and :math:`Q_{o,\text{lower/upper}}` similarly for the
    observations (here, `obs`).

    Args:
        obs: Observed time series.
        sim: Simulated time series.
        lower: Lower bound of the middle section in range ]0,1[, by default
            0.2.
        upper: Upper bound of the middle section in range ]0,1[, by default
            0.7.

    Returns:
        Slope of the middle section of the flow duration curve. Returns NaN
        if the rounded bounds do not select two distinct, valid ranks.

    References:
        .. [#] Yilmaz, K. K., Gupta, H. V., and Wagener, T. ( 2008), A
            process-based diagnostic approach to model evaluation: Application
            to the NWS distributed hydrologic model, Water Resour. Res., 44,
            W09417, doi:10.1029/2007WR006716.
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs_v, sim_v = _mask_valid_arrays(obs, sim)

    if len(obs_v) < 1:
        return np.nan

    if any((x <= 0) or (x >= 1) for x in [upper, lower]):
        raise ValueError('upper and lower have to be in range ]0,1[')

    if lower >= upper:
        raise ValueError(
            'The lower threshold has to be smaller than the upper.'
        )

    lower_index = int(np.round(lower * len(obs_v)))
    upper_index = int(np.round(upper * len(obs_v)))
    if lower_index == upper_index or upper_index >= len(obs_v):
        return np.nan

    # get arrays of sorted (descending) discharges
    obs_fdc = _get_fdc_from_array(obs_v)
    sim_fdc = _get_fdc_from_array(sim_v)

    # for numerical reasons change 0s to 1e-6. Simulations can still contain
    # negatives, so also reset those.
    sim_fdc[sim_fdc <= 0] = 1e-6
    obs_fdc[obs_fdc == 0] = 1e-6

    # calculate fms part by part
    qsm_lower = np.log(sim_fdc[lower_index])
    qsm_upper = np.log(sim_fdc[upper_index])
    qom_lower = np.log(obs_fdc[lower_index])
    qom_upper = np.log(obs_fdc[upper_index])

    fms = ((qsm_lower - qsm_upper) - (qom_lower - qom_upper)) / (
        qom_lower - qom_upper + 1e-6
    )

    return float(fms * 100)


def fdc_fhv(obs: DataArray, sim: DataArray, h: float = 0.02) -> float:
    r"""Calculate the peak flow bias of the flow duration curve [#]_.

    .. math::
        \%\text{BiasFHV} = \frac{\sum_{h=1}^{H}(Q_{s,h} - Q_{o,h})}
        {\sum_{h=1}^{H}Q_{o,h}} \times 100,

    where :math:`Q_s` are the simulations (here, `sim`), :math:`Q_o` the
    observations (here, `obs`) and `H` is the upper fraction of flows of the
    FDC (here, `h`).

    Args:
        obs: Observed time series.
        sim: Simulated time series.
        h: Fraction of upper flows to consider as peak flows of range ]0,1[,
            by default 0.02.

    Returns:
        Peak flow bias. Returns NaN if the selected high-flow tail is empty.

    References:
        .. [#] Yilmaz, K. K., Gupta, H. V., and Wagener, T. ( 2008), A
            process-based diagnostic approach to model evaluation: Application
            to the NWS distributed hydrologic model, Water Resour. Res., 44,
            W09417, doi:10.1029/2007WR006716.
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs_v, sim_v = _mask_valid_arrays(obs, sim)

    if len(obs_v) < 1:
        return np.nan

    if (h <= 0) or (h >= 1):
        raise ValueError(
            'h has to be in range ]0,1[. '
            'Consider small values, e.g. 0.02 for 2% peak flows'
        )

    n_high = int(np.round(h * len(obs_v)))
    if n_high == 0:
        return np.nan

    # get arrays of sorted (descending) discharges
    obs_fdc = _get_fdc_from_array(obs_v)
    sim_fdc = _get_fdc_from_array(sim_v)

    # subset data to only top h flow values
    obs_top = obs_fdc[:n_high]
    sim_top = sim_fdc[:n_high]

    obs_top_sum = float(np.sum(obs_top))
    if obs_top_sum == 0.0:
        # The relative peak-flow bias is undefined when all observed peak flows are zero.
        return np.nan

    fhv = np.sum(sim_top - obs_top) / obs_top_sum

    return float(fhv * 100)


def fdc_flv(obs: DataArray, sim: DataArray, l: float = 0.3) -> float:
    r"""Calculate the low flow bias of the flow duration curve [#]_.

    .. math::
        \%\text{BiasFLV} = -1 \times \frac{\sum_{l=1}^{L}[\log(Q_{s,l}) -
        \log(Q_{s,L})] - \sum_{l=1}^{L}[\log(Q_{o,l}) - \log(Q_{o,L})]}
        {\sum_{l=1}^{L}[\log(Q_{o,l}) - \log(Q_{o,L})]} \times 100,

    where :math:`Q_s` are the simulations (here, `sim`), :math:`Q_o` the
    observations (here, `obs`) and `L` is the lower fraction of flows of the
    FDC (here, `l`).

    Args:
        obs: Observed time series.
        sim: Simulated time series.
        l: Fraction of lower flows to consider as low flows of range ]0,1[,
            by default 0.3.

    Returns:
        Low flow bias. Returns NaN if fewer than two valid samples fall in
        the selected low-flow tail.

    References:
        .. [#] Yilmaz, K. K., Gupta, H. V., and Wagener, T. ( 2008), A
            process-based diagnostic approach to model evaluation: Application
            to the NWS distributed hydrologic model, Water Resour. Res., 44,
            W09417, doi:10.1029/2007WR006716.
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs_v, sim_v = _mask_valid_arrays(obs, sim)

    if len(obs_v) < 1:
        return np.nan

    if (l <= 0) or (l >= 1):
        raise ValueError(
            'l has to be in range ]0,1[. '
            'Consider small values, e.g. 0.3 for 30% low flows'
        )

    n_low = int(np.round(l * len(obs_v)))
    if n_low < _MIN_LOW_FLOW_SAMPLES:
        return np.nan

    # get arrays of sorted (descending) discharges
    obs_fdc = _get_fdc_from_array(obs_v)
    sim_fdc = _get_fdc_from_array(sim_v)

    # for numerical reasons change 0s to 1e-6. Simulations can still contain
    # negatives, so also reset those.
    sim_fdc[sim_fdc <= 0] = 1e-6
    obs_fdc[obs_fdc == 0] = 1e-6

    obs_low = np.log(obs_fdc[-n_low:])
    sim_low = np.log(sim_fdc[-n_low:])

    # calculate flv part by part
    qsl = np.sum(sim_low - sim_low.min())
    qol = np.sum(obs_low - obs_low.min())

    flv = -1 * (qsl - qol) / (qol + 1e-6)

    return float(flv * 100)


def mean_peak_timing(
    obs: DataArray,
    sim: DataArray,
    window: int | None = None,
    resolution: str = '1D',
    datetime_coord: str | None = None,
) -> float:
    r"""Mean difference in peak flow timing.

    Uses scipy.find_peaks to find peaks in the observed time series. Starting
    with all observed peaks, those with a prominence of less than the standard
    deviation of the observed time series are discarded. Next, the lowest peaks
    are subsequently discarded until all remaining peaks have a distance of at
    least 100 steps. Finally, the corresponding peaks in the simulated time
    series are searched in a window of size `window` on either side of the
    observed peaks and the absolute time differences between observed and
    simulated peaks is calculated. The final metric is the mean absolute time
    difference across all peaks. For more details, see Appendix of [#]_.

    Args:
        obs: Observed time series.
        sim: Simulated time series.
        window: Size of window to consider on each side of the observed peak
            for finding the simulated peak. That is, the total window length
            to find the peak in the simulations is
            :math:`2 * \text{window} + 1` centered at the observed peak. The
            default depends on the temporal resolution, e.g. for a resolution
            of '1D', a window of 3 is used and for a resolution of '1h' the
            window size is 12.
        resolution: Temporal resolution of the time series in pandas format,
            e.g. '1D' for daily and '1h' for hourly.
        datetime_coord: Name of datetime coordinate. Tried to infer
            automatically if not specified.

    Returns:
        Mean peak time difference.

    References:
        .. [#] Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A
            note on leveraging synergy in multiple meteorological datasets with
            deep learning for rainfall-runoff modeling, Hydrol. Earth Syst.
            Sci. Discuss., https://doi.org/10.5194/hess-2020-221, in review,
            2020.
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # Handle pooled MultiIndex arrays (e.g. across lead times) cleanly
    if (
        obs.dims
        and obs.dims[0] in obs.indexes
        and isinstance(obs.indexes[obs.dims[0]], pd.MultiIndex)
    ):
        dim = obs.dims[0]
        obs_unstacked, sim_unstacked = obs.unstack(dim), sim.unstack(dim)
        group_dim = obs_unstacked.dims[0]
        errors = []
        for i in range(obs_unstacked.sizes[group_dim]):
            err = mean_peak_timing(
                obs_unstacked.isel({group_dim: i}),
                sim_unstacked.isel({group_dim: i}),
                window,
                resolution,
                datetime_coord,
            )
            if not np.isnan(err):
                errors.append(err)
        return float(np.mean(errors)) if errors else np.nan

    # get time series with only valid observations (scipy's find_peaks doesn't
    # guarantee correctness with NaNs)
    obs, sim = _mask_valid(obs, sim)

    obs_vals = np.asarray(obs.values)
    sim_vals = np.asarray(sim.values)
    if obs_vals.size == 0:
        return np.nan

    # heuristic to get indices of peaks and their corresponding height.
    peaks, _ = signal.find_peaks(
        obs_vals, distance=100, prominence=np.std(obs_vals)
    )

    # infer name of datetime index
    if datetime_coord is None:
        datetime_coord = infer_datetime_coord(obs)

    if window is None:
        # infer a reasonable window size
        window = max(int(get_frequency_factor('12h', resolution)), 3)

    dates_ns = np.asarray(obs.coords[datetime_coord].values, dtype='datetime64[ns]')
    res_td = pd.to_timedelta(resolution).to_numpy().astype('timedelta64[ns]')
    expected_span = (2 * window) * res_td

    # evaluate timing
    timing_errors = []
    for idx in peaks:
        # skip peaks at the start and end of the sequence and peaks around
        # missing observations (NaNs that were removed in obs & sim would
        # result in windows that span too much time).
        if (
            (idx - window < 0)
            or (idx + window >= len(obs_vals))
            or ((dates_ns[idx + window] - dates_ns[idx - window]) != expected_span)
        ):
            continue

        # check if the value at idx is a peak (both neighbors must be smaller)
        if (sim_vals[idx] > sim_vals[idx - 1]) and (sim_vals[idx] > sim_vals[idx + 1]):
            peak_sim_idx = idx
        else:
            # define peak around idx as the max value inside of the window
            values = sim_vals[idx - window : idx + window + 1]
            peak_sim_idx = (idx - window) + int(values.argmax())

        delta = dates_ns[idx] - dates_ns[peak_sim_idx]
        timing_error = float(np.abs(delta / res_td))
        timing_errors.append(timing_error)

    return float(np.mean(timing_errors)) if len(timing_errors) > 0 else np.nan


def missed_peaks(
    obs: DataArray,
    sim: DataArray,
    window: int | None = None,
    resolution: str = '1D',
    percentile: float = 80,
    datetime_coord: str | None = None,
) -> float:
    r"""Fraction of missed peaks.

    Uses scipy.find_peaks to find peaks in the observed and simulated time
    series above a certain percentile. Counts the number of peaks in obs that
    do not exist in sim within the specified window.

    Args:
        obs: Observed time series.
        sim: Simulated time series.
        window: Size of window to consider on each side of the observed peak
            for finding the simulated peak. That is, the total window length
            to find the peak in the simulations is
            :math:`2 * \text{window} + 1` centered at the observed peak. The
            default depends on the temporal resolution, e.g. for a resolution
            of '1D', a window of 1 is used and for a resolution of '1h' the
            window size is 12. Note that this is a different default window
            size than is used in the peak-timing metric for '1D'.
        resolution: Temporal resolution of the time series in pandas format,
            e.g. '1D' for daily and '1h' for hourly.
        percentile: Only consider peaks above this flow percentile (0, 100).
        datetime_coord: Name of datetime coordinate. Tried to infer
            automatically if not specified.

    Returns:
        Fraction of missed peaks.
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # Handle pooled MultiIndex arrays (e.g. across lead times) cleanly
    if (
        obs.dims
        and obs.dims[0] in obs.indexes
        and isinstance(obs.indexes[obs.dims[0]], pd.MultiIndex)
    ):
        dim = obs.dims[0]
        obs_unstacked, sim_unstacked = obs.unstack(dim), sim.unstack(dim)
        group_dim = obs_unstacked.dims[0]
        errors = []
        for i in range(obs_unstacked.sizes[group_dim]):
            err = missed_peaks(
                obs_unstacked.isel({group_dim: i}),
                sim_unstacked.isel({group_dim: i}),
                window,
                resolution,
                percentile,
                datetime_coord,
            )
            if not np.isnan(err):
                errors.append(err)
        return float(np.mean(errors)) if errors else np.nan

    # get time series with only valid observations (scipy's find_peaks doesn't
    # guarantee correctness with NaNs)
    obs, sim = _mask_valid(obs, sim)

    obs_vals = np.asarray(obs.values)
    sim_vals = np.asarray(sim.values)
    if obs_vals.size == 0 or sim_vals.size == 0:
        return np.nan

    # minimum height of a peak, as defined by percentile, which can be passed
    min_obs_height = np.percentile(obs_vals, percentile)
    min_sim_height = np.percentile(sim_vals, percentile)

    # get time indices of peaks in obs and sim.
    peaks_obs_times, _ = signal.find_peaks(
        obs_vals, distance=30, height=min_obs_height
    )
    peaks_sim_times, _ = signal.find_peaks(
        sim_vals, distance=30, height=min_sim_height
    )

    if len(peaks_obs_times) == 0:
        return 0.0

    # infer name of datetime index
    if datetime_coord is None:
        datetime_coord = infer_datetime_coord(obs)

    # infer a reasonable window size
    if window is None:
        window = max(int(get_frequency_factor('12h', resolution)), 1)

    dates_ns = np.asarray(obs.coords[datetime_coord].values, dtype='datetime64[ns]')
    res_td = pd.to_timedelta(resolution).to_numpy().astype('timedelta64[ns]')
    expected_span = (2 * window) * res_td

    # count missed peaks
    missed_events = 0
    valid_peaks = 0

    for idx in peaks_obs_times:
        # skip peaks at the start and end of the sequence and peaks around
        # missing observations (NaNs that were removed in obs & sim would
        # result in windows that span too much time).
        if (
            (idx - window < 0)
            or (idx + window >= len(obs_vals))
            or ((dates_ns[idx + window] - dates_ns[idx - window]) != expected_span)
        ):
            continue

        valid_peaks += 1
        nearby_peak_sim_index = np.where(
            np.abs(peaks_sim_times - idx) <= window
        )[0]
        if len(nearby_peak_sim_index) == 0:
            missed_events += 1

    return float(missed_events / valid_peaks) if valid_peaks > 0 else np.nan


def mean_absolute_percentage_peak_error(
    obs: DataArray, sim: DataArray
) -> float:
    r"""Calculate the mean absolute percentage error (MAPE) for peaks.

    .. math::
        \text{MAPE}_\text{peak} = \frac{1}{P}\sum_{p=1}^{P}
        \left |\frac{Q_{s,p} - Q_{o,p}}{Q_{o,p}} \right | \times 100,

    where :math:`Q_{s,p}` are the simulated peaks (here, `sim`),
    :math:`Q_{o,p}` the observed peaks (here, `obs`) and `P` is the number of
    peaks.

    Uses scipy.find_peaks to find peaks in the observed time series. The
    observed peaks indices are used to subset observed and simulated flows.
    Finally, the MAPE metric is calculated as the mean absolute percentage
    error of observed peak flows and corresponding simulated flows.

    Args:
        obs: Observed time series.
        sim: Simulated time series.

    Returns:
        Mean absolute percentage error (MAPE) for peaks.
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    # return np.nan if there are no valid observed or simulated values
    if obs.size == 0 or sim.size == 0:
        return np.nan

    obs_vals = np.asarray(obs.values)
    sim_vals = np.asarray(sim.values)

    # heuristic to get indices of peaks and their corresponding height.
    peaks, _ = signal.find_peaks(
        obs_vals, distance=100, prominence=np.std(obs_vals)
    )

    # check if any peaks exist, otherwise return np.nan
    if peaks.size == 0:
        return np.nan

    # subset data to only peak values
    obs_p = obs_vals[peaks]
    sim_p = sim_vals[peaks]

    # a relative error is undefined for zero-valued observed peaks; such peaks
    # cannot occur for prominence-based detection on non-negative discharge,
    # but guard explicitly so no Inf can leak into aggregate statistics.
    if np.any(obs_p == 0):
        return np.nan

    # calculate the mean absolute percentage peak error
    return float(np.sum(np.abs((sim_p - obs_p) / obs_p)) / peaks.size * 100)


# Keep discovery, explicit selection, and the all-metrics path in sync.
_METRIC_FUNCTIONS: dict[str, Callable[..., float]] = {
    'NSE': nse,
    'MSE': mse,
    'RMSE': rmse,
    'KGE': kge,
    'Alpha-NSE': alpha_nse,
    'Pearson-r': pearsonr,
    'Beta-KGE': beta_kge,
    'Beta-NSE': beta_nse,
    'FHV': fdc_fhv,
    'FMS': fdc_fms,
    'FLV': fdc_flv,
    'Peak-Timing': mean_peak_timing,
    'Missed-Peaks': missed_peaks,
    'Peak-MAPE': mean_absolute_percentage_peak_error,
}


def calculate_all_metrics(
    obs: DataArray,
    sim: DataArray,
    resolution: str = '1D',
    datetime_coord: str | None = None,
) -> dict[str, float]:
    """Calculate all metrics with default values.

    Args:
        obs: Observed time series.
        sim: Simulated time series.
        resolution: Temporal resolution of the time series in pandas format,
            e.g. '1D' for daily and '1h' for hourly.
        datetime_coord: Datetime coordinate in the passed DataArray. Tried to
            infer automatically if not specified.

    Returns:
        Dictionary with keys corresponding to metric name and values
        corresponding to metric values.

    Raises:
        AllNaNError: If all observations or all simulations are NaN.
    """
    return calculate_metrics(
        obs,
        sim,
        get_available_metrics(),
        resolution=resolution,
        datetime_coord=datetime_coord,
    )


def calculate_metrics(
    obs: DataArray,
    sim: DataArray,
    metrics: list[str],
    resolution: str = '1D',
    datetime_coord: str | None = None,
) -> dict[str, float]:
    """Calculate specific metrics with default values.

    Args:
        obs: Observed time series.
        sim: Simulated time series.
        metrics: List of metric names.
        resolution: Temporal resolution of the time series in pandas format,
            e.g. '1D' for daily and '1h' for hourly.
        datetime_coord: Datetime coordinate in the passed DataArray. Tried to
            infer automatically if not specified.

    Returns:
        Dictionary with keys corresponding to metric name and values
        corresponding to metric values.

    Raises:
        AllNaNError: If all observations or all simulations are NaN.
    """
    if 'all' in metrics:
        metrics = get_available_metrics()

    _check_all_nan(obs, sim)

    canonical_names = {name.lower(): name for name in _METRIC_FUNCTIONS}
    values = {}
    for metric in metrics:
        key_lower = metric.lower()
        if key_lower not in canonical_names:
            raise RuntimeError(f'Unknown metric {metric}')
        name = canonical_names[key_lower]
        function = _METRIC_FUNCTIONS[name]
        if function in (mean_peak_timing, missed_peaks):
            values[name] = function(
                obs, sim, resolution=resolution, datetime_coord=datetime_coord
            )
        else:
            values[name] = function(obs, sim)

    return values


def _check_all_nan(obs: DataArray, sim: DataArray) -> None:
    """Raise `AllNaNError` if all observations or all simulations are NaN.

    Raises:
        AllNaNError: If all observations or all simulations are NaN.
    """
    if bool(np.isnan(obs.values).all()):
        raise AllNaNError(
            'All observed values are NaN, thus metrics will be NaN, too.'
        )
    if bool(np.isnan(sim.values).all()):
        raise AllNaNError(
            'All simulated values are NaN, thus metrics will be NaN, too.'
        )
