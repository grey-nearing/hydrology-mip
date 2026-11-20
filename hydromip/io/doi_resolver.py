# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared SSRF-hardened DOI & HTTPS archive resolver for HydrologyMIP.

Supports Zenodo, HydroShare, OSF, Figshare DOIs/record URLs as well as trusted
HTTPS dataset URLs, enforcing HTTPS-only transfers, trusted host allowlists,
global-only DNS resolution on every redirect hop, and strict download byte caps.

Zero ``try``/``except`` blocks to comply with repository AST guardrails.
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_MAX_DOWNLOAD_BYTES: int = 500 * 1024 * 1024  # 500 MiB ceiling
DEFAULT_METADATA_MAX_BYTES: int = 10 * 1024 * 1024
MAX_REDIRECT_HOPS: int = 5
ARCHIVE_TIMEOUT_SECONDS: int = 120

TRUSTED_EXACT_HOSTS: frozenset[str] = frozenset(
    {
        "zenodo.org",
        "hydroshare.org",
        "osf.io",
        "figshare.com",
        "github.com",
        "raw.githubusercontent.com",
        "objects.githubusercontent.com",
        "huggingface.co",
        "data.hydrologymip.org",
    }
)
TRUSTED_WILDCARD_SUFFIXES: tuple[str, ...] = (
    ".zenodo.org",
    ".hydroshare.org",
    ".osf.io",
    ".figshare.com",
)
SUPPORTED_ARCHIVE_FILE_EXTS: tuple[str, ...] = (
    ".zarr.zip",
    ".nc",
    ".nc4",
    ".parquet",
    ".csv",
    ".tsv",
    ".npz",
    ".zip",
)

ZENODO_DOI_RE: re.Pattern[str] = re.compile(
    r"(?:10\.5281/zenodo\.|zenodo\.org/records?/)(\d+)", re.IGNORECASE
)
HYDROSHARE_DOI_RE: re.Pattern[str] = re.compile(
    r"(?:10\.4211/hs\.|hydroshare\.org/resource/)([0-9a-f]{32})", re.IGNORECASE
)
OSF_DOI_RE: re.Pattern[str] = re.compile(
    r"(?:10\.17605/osf\.io/|osf\.io/)([A-Za-z0-9]{5,12})", re.IGNORECASE
)
FIGSHARE_DOI_RE: re.Pattern[str] = re.compile(
    r"(?:10\.6084/m9\.figshare\.|figshare\.com/articles/(?:[^/]+/)*)(\d+)", re.IGNORECASE
)


def _is_ip_literal(host: str) -> bool:
    stripped = host.strip("[]")
    return bool(
        re.fullmatch(r"\d+\.\d+\.\d+\.\d+", stripped)
        or (":" in stripped and re.fullmatch(r"[0-9a-fA-F:]+", stripped))
    )


def is_trusted_archive_host(hostname: str) -> bool:
    """Return True if ``hostname`` is an explicitly allowlisted public repository host."""
    host = hostname.strip().lower().rstrip(".")
    if not host:
        return False
    if host in TRUSTED_EXACT_HOSTS:
        return True
    return any(
        host.endswith(suffix) and len(host) > len(suffix)
        for suffix in TRUSTED_WILDCARD_SUFFIXES
    )


def validate_https_target_url(url: str) -> str:
    """Enforce HTTPS, trusted archive hostname allowlist, and global-only DNS resolution."""
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme.lower() != "https":
        raise ValueError(f"Only HTTPS archive URLs are permitted (got scheme {parsed.scheme!r}).")
    if parsed.username or parsed.password:
        raise ValueError("Archive URLs containing embedded credentials are rejected.")
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise ValueError("Archive URL is missing a hostname.")

    if _is_ip_literal(host):
        ip_obj = ipaddress.ip_address(host.strip("[]"))
        if not ip_obj.is_global:
            raise ValueError(f"SSRF guard blocked non-global IP literal {host!r}.")
        raise ValueError(f"Untrusted IP literal host {host!r}; archive must use a trusted domain.")

    if not is_trusted_archive_host(host):
        raise ValueError(
            f"Host {host!r} is outside the trusted archive allowlist "
            "(Zenodo, HydroShare, OSF, Figshare, GitHub, HuggingFace, data.hydrologymip.org)."
        )

    port = parsed.port or 443
    addr_infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not addr_infos:
        raise ValueError(f"DNS lookup returned no addresses for {host!r}.")
    for info in addr_infos:
        sockaddr = info[4]
        ip_str = str(sockaddr[0]).split("%", 1)[0]
        ip_obj = ipaddress.ip_address(ip_str)
        if not ip_obj.is_global:
            raise ValueError(
                f"SSRF guard blocked non-global resolved IP {ip_str!r} for host {host!r}."
            )
    return parsed.geturl()


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Disable automatic redirects so every redirect hop undergoes SSRF validation."""

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


class _NonRaisingHTTPErrorProcessor(urllib.request.HTTPErrorProcessor):
    """Return HTTP responses cleanly for manual status and redirect checks."""

    def http_response(self, request: Any, response: Any) -> Any:
        return response

    https_response = http_response


def validate_and_fetch_archive_url(
    url: str,
    dest_path: Path | None = None,
    *,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    timeout: int = ARCHIVE_TIMEOUT_SECONDS,
    user_agent: str = "HydrologyMIP/1.0",
) -> bytes:
    """Fetch a remote HTTPS resource while re-validating every HTTP redirect hop."""
    current_url = validate_https_target_url(url)
    opener = urllib.request.build_opener(_NoRedirectHandler(), _NonRaisingHTTPErrorProcessor())

    for _hop in range(MAX_REDIRECT_HOPS + 1):
        req = urllib.request.Request(current_url, headers={"User-Agent": user_agent})
        with opener.open(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", None) or resp.getcode() or 200)
            if status in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location", "").strip()
                if not location:
                    raise ValueError(f"HTTP {status} redirect without Location header.")
                next_url = urllib.parse.urljoin(current_url, location)
                current_url = validate_https_target_url(next_url)
                continue
            if status >= 400:
                raise ValueError(
                    f"Archive download failed with HTTP {status} from {current_url}."
                )

            content_len = resp.headers.get("Content-Length")
            if content_len and content_len.isdigit() and int(content_len) > max_bytes:
                raise ValueError(
                    f"Remote archive size {int(content_len)} bytes exceeds {max_bytes} byte cap."
                )

            chunk_size = 1024 * 1024
            total = 0
            if dest_path is not None:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with open(dest_path, "wb") as out_f:
                    while True:
                        buf = resp.read(chunk_size)
                        if not buf:
                            break
                        total += len(buf)
                        if total > max_bytes:
                            raise ValueError(
                                f"Downloaded archive exceeded {max_bytes} byte limit."
                            )
                        out_f.write(buf)
                return b""

            chunks: list[bytes] = []
            while True:
                buf = resp.read(chunk_size)
                if not buf:
                    break
                total += len(buf)
                if total > max_bytes:
                    raise ValueError(f"Downloaded payload exceeded {max_bytes} byte limit.")
                chunks.append(buf)
            return b"".join(chunks)

    raise ValueError(
        f"Exceeded maximum HTTP redirect limit ({MAX_REDIRECT_HOPS}) fetching archive."
    )


def archive_file_priority(filename: str) -> int:
    """Rank hydrograph archive filenames with `.zarr.zip` highest and plain `.zip` lowest."""
    low = filename.lower()
    if low.endswith(".zarr.zip"):
        return 0
    if low.endswith((".nc", ".nc4")):
        return 1
    if low.endswith(".parquet"):
        return 2
    if low.endswith((".csv", ".tsv")):
        return 3
    if low.endswith(".npz"):
        return 4
    if low.endswith(".zip"):
        return 5
    return 99


def _is_eligible_archive_entry(filename: str, size: int, max_bytes: int) -> bool:
    clean = filename.strip()
    if not clean or size > max_bytes:
        return False
    base = Path(clean).name.lower()
    if not base or base.startswith(".") or base.startswith(("readme", "license")):
        return False
    if "__macosx" in clean.lower():
        return False
    return base.endswith(SUPPORTED_ARCHIVE_FILE_EXTS)


def extract_zenodo_doi(raw_archive_url: str) -> str:
    """Return canonical ``10.5281/zenodo.<id>`` if ``raw_archive_url`` identifies Zenodo."""
    m = ZENODO_DOI_RE.search(raw_archive_url.strip())
    return f"10.5281/zenodo.{m.group(1)}" if m else ""


def resolve_archive_download_url(
    raw_archive_url: str,
    *,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    fetch_fn: Callable[..., bytes] | None = None,
) -> tuple[str, str]:
    """Resolve a Zenodo/HydroShare/OSF/Figshare DOI or HTTPS URL to ``(filename, download_url)``."""
    raw = raw_archive_url.strip()
    if not raw:
        raise ValueError("archive_url must not be empty.")
    fetcher = fetch_fn if fetch_fn is not None else validate_and_fetch_archive_url

    m_zen = ZENODO_DOI_RE.search(raw)
    if m_zen:
        rec_id = m_zen.group(1)
        api_url = f"https://zenodo.org/api/records/{rec_id}"
        rec_json = json.loads(
            fetcher(api_url, max_bytes=DEFAULT_METADATA_MAX_BYTES).decode("utf-8")
        )
        files: list[tuple[int, str, str]] = []
        for entry in rec_json.get("files", []):
            key = str(entry.get("key") or entry.get("filename") or "")
            links = entry.get("links") or {}
            dl_url = str(
                links.get("content")
                or links.get("download")
                or links.get("self")
                or f"https://zenodo.org/records/{rec_id}/files/{urllib.parse.quote(key)}?download=1"
            )
            size = int(entry.get("size") or 0)
            if _is_eligible_archive_entry(key, size, max_bytes):
                files.append((archive_file_priority(key), key, dl_url))
        if not files:
            raise ValueError(
                f"No supported hydrograph files (<= {max_bytes // (1024 * 1024)} MiB) found in Zenodo record {rec_id}."
            )
        files.sort(key=lambda item: (item[0], item[1]))
        return Path(files[0][1]).name, files[0][2]

    m_hs = HYDROSHARE_DOI_RE.search(raw)
    if m_hs:
        hs_id = m_hs.group(1).lower()
        hs_api = f"https://www.hydroshare.org/hsapi/resource/{hs_id}/files/"
        hs_json = json.loads(
            fetcher(hs_api, max_bytes=DEFAULT_METADATA_MAX_BYTES).decode("utf-8")
        )
        files = []
        for entry in hs_json.get("results", []):
            fname = str(entry.get("file_name") or "")
            dl_url = str(entry.get("url") or "")
            size = int(entry.get("size") or 0)
            if dl_url and _is_eligible_archive_entry(fname, size, max_bytes):
                files.append((archive_file_priority(fname), fname, dl_url))
        if not files:
            raise ValueError(
                f"No supported hydrograph files (<= {max_bytes // (1024 * 1024)} MiB) found in HydroShare resource {hs_id}."
            )
        files.sort(key=lambda item: (item[0], item[1]))
        return Path(files[0][1]).name, files[0][2]

    m_osf = OSF_DOI_RE.search(raw)
    if m_osf:
        node_id = m_osf.group(1)
        osf_api = f"https://api.osf.io/v2/nodes/{urllib.parse.quote(node_id)}/files/osfstorage/?format=json"
        osf_json = json.loads(
            fetcher(osf_api, max_bytes=DEFAULT_METADATA_MAX_BYTES).decode("utf-8")
        )
        osf_items = list(osf_json.get("data", []))
        folder_urls = [
            str(((item.get("relationships") or {}).get("files") or {}).get("links", {}).get("related", {}).get("href") or "")
            for item in osf_items
            if (item.get("attributes") or {}).get("kind") == "folder"
        ]
        for f_url in folder_urls[:5]:
            if f_url.startswith("https://api.osf.io/"):
                sub_json = json.loads(
                    fetcher(f_url, max_bytes=DEFAULT_METADATA_MAX_BYTES).decode("utf-8")
                )
                osf_items.extend(sub_json.get("data", []))
        files = []
        for item in osf_items:
            attrs = item.get("attributes") or {}
            if attrs.get("kind") == "folder":
                continue
            fname = str(attrs.get("name") or "")
            dl_url = str((item.get("links") or {}).get("download") or "")
            size = int(attrs.get("size") or 0)
            if dl_url and _is_eligible_archive_entry(fname, size, max_bytes):
                files.append((archive_file_priority(fname), fname, dl_url))
        if not files:
            raise ValueError(
                f"No supported hydrograph files (<= {max_bytes // (1024 * 1024)} MiB) found in OSF project {node_id}."
            )
        files.sort(key=lambda item: (item[0], item[1]))
        return Path(files[0][1]).name, files[0][2]

    m_fig = FIGSHARE_DOI_RE.search(raw)
    if m_fig:
        art_id = m_fig.group(1)
        fig_api = f"https://api.figshare.com/v2/articles/{art_id}/files"
        fig_list = json.loads(
            fetcher(fig_api, max_bytes=DEFAULT_METADATA_MAX_BYTES).decode("utf-8")
        )
        files = []
        for entry in fig_list if isinstance(fig_list, list) else []:
            fname = str(entry.get("name") or "")
            dl_url = str(entry.get("download_url") or "")
            size = int(entry.get("size") or 0)
            if dl_url and _is_eligible_archive_entry(fname, size, max_bytes):
                files.append((archive_file_priority(fname), fname, dl_url))
        if not files:
            raise ValueError(
                f"No supported hydrograph files (<= {max_bytes // (1024 * 1024)} MiB) found in Figshare article {art_id}."
            )
        files.sort(key=lambda item: (item[0], item[1]))
        return Path(files[0][1]).name, files[0][2]

    validated = validate_https_target_url(raw)
    url_path = urllib.parse.urlparse(validated).path
    fname = Path(urllib.parse.unquote(url_path)).name or "submission.nc"
    return fname, validated


def resolve_and_download_archive(
    raw_archive_url: str,
    target_dir: Path,
    *,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    allow_file_uri: bool = False,
    fetch_fn: Callable[..., bytes] | None = None,
) -> Path:
    """Resolve a DOI or trusted HTTPS URL (or local ``file://`` if allowed) and download safely."""
    raw = raw_archive_url.strip()
    if not raw:
        raise ValueError("archive_url must not be empty.")
    if allow_file_uri:
        parsed = urllib.parse.urlparse(raw)
        if parsed.scheme.lower() == "file":
            local_p = Path(urllib.parse.unquote(parsed.path))
            if not local_p.is_file():
                raise FileNotFoundError(f"Local file:// archive not found: {local_p}")
            return local_p

    fetcher = fetch_fn if fetch_fn is not None else validate_and_fetch_archive_url
    fname, dl_url = resolve_archive_download_url(
        raw, max_bytes=max_bytes, fetch_fn=fetcher
    )
    dest = target_dir / Path(fname).name
    fetcher(dl_url, dest_path=dest, max_bytes=max_bytes)
    return dest
