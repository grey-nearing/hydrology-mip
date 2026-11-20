# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Schema-only Tier 2 LLM fallback using Vertex AI IAM or Gemini Developer API.

Only structural file metadata JSON (dimensions, variable names, shapes, dtypes,
and first 5 coordinate/column labels) is sent to Gemini Flash, which returns a
strictly validated Pydantic ``IngestionRecipe`` JSON object or sets ``cannot_map=True``.
No code generation, dynamic execution, or full-array inspection is performed.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

import google.auth
import google.auth.exceptions
import google.auth.transport.requests
from pydantic import BaseModel, Field, ValidationError

from hydromip.io.ingest import DEFAULT_PREFIX_STRIPS, IngestionRecipe

DEFAULT_GEMINI_MODEL: str = "gemini-3.5-flash-lite"
DEFAULT_VERTEX_LOCATION: str = "global"
CLOUD_PLATFORM_SCOPE: str = "https://www.googleapis.com/auth/cloud-platform"
GEMINI_TIMEOUT_SECONDS: int = 20

_SCHEMA_PROMPT_TEMPLATE = """You are a hydrological data schema mapper for HydrologyMIP (CAMELS-US Daily, 531 USGS basins, 1989-10-01 to 1999-09-30, units mm/day).

Inspect ONLY the structural file metadata JSON below and map the archive onto an IngestionRecipe JSON object.

RULES:
1. Respond with valid JSON ONLY.
2. Never invent basin IDs or dates if none are labeled in the metadata. If basin IDs or dates are completely absent or mapping is genuinely ambiguous, set "cannot_map": true and list concrete reasons in "missing_or_ambiguous".
3. Populate only field names that literally appear in the metadata below.
4. For 2D/3D gridded NetCDF/Zarr/NPZ files, fill "var", "basin_dim", "time_dim", and optional "member_dim".
5. For CSV/Parquet wide tables (rows=dates, cols=basins), set "basin_dim": "columns", "date_col", and "time_dim". For long/tidy CSV/Parquet tables, fill "basin_col", "date_col", "value_col", and optional "member_col".

Required JSON keys:
{
  "cannot_map": false,
  "missing_or_ambiguous": [],
  "format": "netcdf|zarr|zarr.zip|csv|parquet|npz",
  "var": "<string or null>",
  "basin_dim": "<string or null>",
  "time_dim": "<string or null>",
  "member_dim": "<string or null>",
  "basin_col": "<string or null>",
  "date_col": "<string or null>",
  "value_col": "<string or null>",
  "member_col": "<string or null>",
  "id_prefix_strip": ["usgs_", "camels_", "basin_", "gauge_"],
  "zero_pad_width": 8,
  "allow_positional_order": false,
  "unit_scale": 1.0,
  "confidence": 0.0,
  "notes": "<short explanation>"
}

UNRESOLVED_HINTS:
__HINTS_JSON__

STRUCTURAL_METADATA:
__METADATA_JSON__
"""


class _GeminiRecipeEnvelope(BaseModel):
    """Strict Pydantic validation for Gemini Flash Tier 2 JSON responses."""

    cannot_map: bool = False
    missing_or_ambiguous: list[str] = Field(default_factory=list)
    format: str | None = None
    var: str | None = None
    basin_dim: str | None = None
    time_dim: str | None = None
    member_dim: str | None = None
    basin_col: str | None = None
    date_col: str | None = None
    value_col: str | None = None
    member_col: str | None = None
    id_prefix_strip: list[str] = Field(default_factory=lambda: list(DEFAULT_PREFIX_STRIPS))
    zero_pad_width: int = 8
    allow_positional_order: bool = False
    unit_scale: float = 1.0
    confidence: float = 0.0
    notes: str = ""


def build_hydromip_package_snippet(
    filename: str = "your_predictions.nc",
    leaderboard_id: str = "gauged__single_camels",
) -> str:
    """Return a copyable CLI snippet for local packaging via ``hydromip package``."""
    safe_name = filename or "your_predictions.nc"
    return (
        "pip install git+https://github.com/grey-nearing/camels-us-mip.git\n"
        f"hydromip package --input {safe_name} --output ./camels_submission.zarr.zip \\\n"
        f"  --leaderboard {leaderboard_id}"
    )


def build_stuck_report(
    reason: str,
    missing_or_ambiguous: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
    leaderboard_id: str = "gauged__single_camels",
) -> dict[str, Any]:
    """Build a structured diagnostic report when automatic schema mapping cannot complete."""
    filename = str((metadata or {}).get("filename") or "your_predictions.nc")
    detected_vars = list((metadata or {}).get("vars") or [])[:12]
    detected_dims = list((metadata or {}).get("dims") or [])[:12]
    items = [str(x) for x in missing_or_ambiguous if str(x).strip()]
    if detected_vars:
        items.append(f"Detected variables/columns: {detected_vars}")
    if detected_dims:
        items.append(f"Detected dimensions: {detected_dims}")
    return {
        "reason": reason,
        "missing_or_ambiguous": items,
        "cli_snippet": build_hydromip_package_snippet(
            filename=filename,
            leaderboard_id=leaderboard_id,
        ),
    }


def _strip_json_fences(raw_text: str) -> str:
    cleaned = raw_text.strip()
    fence = re.match(r"^```(?:json)?\s*\n(.*?)\n```$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return cleaned


def _extract_gemini_text(response_payload: Mapping[str, Any]) -> str:
    candidates = response_payload.get("candidates")
    if isinstance(candidates, list) and candidates:
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        if parts and isinstance(parts[0], Mapping) and "text" in parts[0]:
            return str(parts[0]["text"])
    if "text" in response_payload:
        return str(response_payload["text"])
    raise ValueError("Gemini response JSON did not include candidates[0].content.parts[0].text.")


def get_llm_backend_mode() -> str:
    """Return active Tier 2 schema LLM backend: 'vertex_ai', 'gemini_api_key', or 'disabled'."""
    if os.environ.get("VERTEX_AI_PROJECT", "").strip():
        return "vertex_ai"
    if os.environ.get("GEMINI_API_KEY", "").strip():
        return "gemini_api_key"
    return "disabled"


def _call_gemini_rest(
    req: urllib.request.Request,
    default_format: str,
    timeout_seconds: int,
    *,
    use_vertex_adc: bool = False,
) -> tuple[_GeminiRecipeEnvelope | None, IngestionRecipe | None, str | None]:
    """Execute Vertex AI / Gemini REST HTTPS call and validate JSON schema response at the external boundary."""
    try:
        if use_vertex_adc:
            creds, _ = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
            creds.refresh(google.auth.transport.requests.Request())
            if not creds.token:
                raise ValueError("GCP Application Default Credentials returned an empty Bearer token.")
            req.add_header("Authorization", f"Bearer {creds.token}")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            status_code = getattr(resp, "status", None) or resp.getcode()
            if status_code is not None and int(status_code) >= 400:
                raise ValueError(f"Gemini HTTP {status_code}")
            raw_bytes = resp.read()
        outer_json = json.loads(raw_bytes.decode("utf-8"))
        text_payload = _extract_gemini_text(outer_json)
        parsed_obj = json.loads(_strip_json_fences(text_payload))
        envelope = _GeminiRecipeEnvelope.model_validate(parsed_obj)
        if envelope.cannot_map:
            return envelope, None, None
        fmt_resolved = envelope.format or default_format
        recipe = IngestionRecipe.model_validate(
            {
                "format": fmt_resolved,
                "var": envelope.var,
                "basin_dim": envelope.basin_dim,
                "time_dim": envelope.time_dim,
                "member_dim": envelope.member_dim,
                "basin_col": envelope.basin_col,
                "date_col": envelope.date_col,
                "value_col": envelope.value_col,
                "member_col": envelope.member_col,
                "id_prefix_strip": envelope.id_prefix_strip or list(DEFAULT_PREFIX_STRIPS),
                "zero_pad_width": envelope.zero_pad_width,
                "allow_positional_order": False,
                "unit_scale": envelope.unit_scale,
                "recipe_source": "llm_tier2",
                "confidence": envelope.confidence,
                "notes": envelope.notes,
            }
        )
        return envelope, recipe, None
    except (
        urllib.error.URLError,
        google.auth.exceptions.GoogleAuthError,
        TimeoutError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        ValidationError,
    ) as exc:
        return None, None, f"Tier 2 schema inference request failed ({type(exc).__name__}: {exc})."


def infer_recipe_from_metadata(
    metadata: Mapping[str, Any],
    unresolved_reasons: Sequence[str] | None = None,
    *,
    leaderboard_id: str = "gauged__single_camels",
    timeout_seconds: int = GEMINI_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Infer an ``IngestionRecipe`` from structural metadata using Vertex AI or Gemini Flash REST API."""
    hints = [str(r) for r in (unresolved_reasons or []) if str(r).strip()]
    backend_mode = get_llm_backend_mode()
    if backend_mode == "disabled":
        report = build_stuck_report(
            reason=(
                "Automatic schema fallback is unavailable because neither VERTEX_AI_PROJECT "
                "nor GEMINI_API_KEY is configured on this scorer instance. Use the manual "
                "dropdown mapping below or package locally with `hydromip package`."
            ),
            missing_or_ambiguous=hints or ["Could not deterministically match schema fields."],
            metadata=metadata,
            leaderboard_id=leaderboard_id,
        )
        return {"status": "stuck", "recipe": None, "stuck_report": report}

    model = os.environ.get("HYDROMIP_LLM_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    quoted_model = urllib.parse.quote(model, safe=".-")
    if backend_mode == "vertex_ai":
        project = os.environ.get("VERTEX_AI_PROJECT", "").strip()
        location = (
            os.environ.get("VERTEX_AI_LOCATION", DEFAULT_VERTEX_LOCATION).strip()
            or DEFAULT_VERTEX_LOCATION
        )
        quoted_project = urllib.parse.quote(project, safe=".-")
        quoted_location = urllib.parse.quote(location, safe=".-")
        host = (
            "aiplatform.googleapis.com"
            if location == "global"
            else f"{quoted_location}-aiplatform.googleapis.com"
        )
        endpoint = (
            f"https://{host}/v1/"
            f"projects/{quoted_project}/locations/{quoted_location}/"
            f"publishers/google/models/{quoted_model}:generateContent"
        )
    else:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quoted_model}:generateContent"
            f"?key={urllib.parse.quote(api_key, safe='')}"
        )

    prompt = (
        _SCHEMA_PROMPT_TEMPLATE.replace("__HINTS_JSON__", json.dumps(hints, indent=2)).replace(
            "__METADATA_JSON__", json.dumps(dict(metadata), indent=2, default=str)
        )
    )
    request_body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.0,
        },
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "HydrologyMIP/1.0"},
        method="POST",
    )

    envelope, recipe, err_reason = _call_gemini_rest(
        req,
        default_format=str(metadata.get("format") or ""),
        timeout_seconds=timeout_seconds,
        use_vertex_adc=(backend_mode == "vertex_ai"),
    )
    if err_reason is not None or envelope is None:
        report = build_stuck_report(
            reason=err_reason or "Tier 2 schema inference request failed.",
            missing_or_ambiguous=hints or ["Verify variable, basin, and date labels manually below."],
            metadata=metadata,
            leaderboard_id=leaderboard_id,
        )
        return {"status": "stuck", "recipe": None, "stuck_report": report}

    if envelope.cannot_map or recipe is None:
        report = build_stuck_report(
            reason=envelope.notes or "Schema adapter marked the uploaded archive as unmappable without manual confirmation.",
            missing_or_ambiguous=envelope.missing_or_ambiguous or hints,
            metadata=metadata,
            leaderboard_id=leaderboard_id,
        )
        return {"status": "stuck", "recipe": None, "stuck_report": report}

    return {"status": "ok", "recipe": recipe, "stuck_report": None}
