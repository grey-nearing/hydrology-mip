/* Copyright 2026 HydrologyMIP Contributors
 * Tab-driven frontend controller for hydrologymip.org
 */

(function () {
  "use strict";

  const STATE = {
    allLeaderboards: [],
    families: [],
    protocol: null,
    activeView: "landing", // landing | about | leaderboards | models | contribute | publications
    activeSpatialTab: "gauged", // gauged | pub | pur
    activeForcingTab: "single_camels", // single_camels | multi_camels | lagged_streamflow | freeform
    categoryFilter: "",
    searchQuery: "",
    familySearchQuery: "",
    familyTypeFilter: "",
    sortByBoard: {},
    metricPreset: "all", // "all" | "core" | "efficiency" | "fdc_peaks" | "custom"
    visibleMetricCols: null, // Set of col keys, initialized lazily from GAUCH_METRIC_COLUMNS
    figureMode: "table", // "table" | "regime" | "hidden"
    chartParadigmFilter: "", // "" | "Machine learning" | "Hybrid (ML + Physics)" | "Conceptual" | "Process-based"
    inspectedEntryId: null,
    uploadedFile: null,
    lastScoringResult: null,
    contribStep: 1,
    previewInspectData: null,
    activePreviewEntryId: null,
    baseLeaderboards: null,
    baseFamilies: null,
    // Hydrograph Comparison Explorer State
    comparedModels: [], // Array of model_key ("<leaderboard_id>__<entry_id>")
    hydroIndex: null, // Loaded from /api/hydrographs/index
    hydroBasinId: "01022500",
    hydroHucFilter: "",
    hydroBasinSearch: "",
    hydroAddLbFilter: "",
    hydroActivePreset: "peak1",
    hydroZoomStart: 0,
    hydroZoomEnd: 3651,
    hydroForcingsOn: { daymet: true, maurer: true, nldas: true },
    hydroQScale: "linear", // "linear" | "sqrt" | "log"
    hydroTempMode: "tmean_band", // "tmean_band" | "tmean" | "tmax_tmin"
    hydroBasinPayload: null, // Latest response from /api/hydrographs/basin/{basin_id}
    hydroHoverDayIdx: null,
    hydroNavHoverZone: null, // "left" | "right" | "pan" | null
    hydroMapColorBy: "nse_primary", // "nse_primary" | "nse_diff" | "huc02" | "mean_qobs"
    hydroMapVisible: true,
    hydroMapViewBox: { x: 0, y: 0, w: 960, h: 580 },
  };

  const HYDRO_MODEL_COLORS = [
    "#dc2626", // Crimson
    "#2563eb", // Royal Blue
    "#059669", // Emerald
    "#9333ea", // Purple
    "#ea580c", // Orange
    "#0891b2", // Cyan
    "#be185d", // Rose
    "#4f46e5", // Indigo
  ];

  const HYDRO_FORCING_COLORS = {
    daymet: {
      label: "Daymet",
      stroke: "#0284c7",
      fill: "rgba(2, 132, 199, 0.34)",
      band: "rgba(2, 132, 199, 0.16)",
    },
    maurer: {
      label: "Maurer",
      stroke: "#7c3aed",
      fill: "rgba(124, 58, 237, 0.30)",
      band: "rgba(124, 58, 237, 0.15)",
    },
    nldas: {
      label: "NLDAS",
      stroke: "#d97706",
      fill: "rgba(217, 119, 6, 0.32)",
      band: "rgba(217, 119, 6, 0.16)",
    },
  };

  function formatLeaderboardShortLabel(lb) {
    if (!lb) return "";
    const id = String(lb.id || "");
    const map = {
      gauged__single_camels: "Gauged · Single",
      gauged__multi_camels: "Gauged · Multi",
      gauged__lagged_streamflow: "Gauged · Streamflow",
      gauged__freeform: "Gauged · Freeform",
      pub__single_camels: "PUB · Single",
      pub__multi_camels: "PUB · Multi",
      pub__lagged_streamflow: "PUB · Streamflow",
      pub__freeform: "PUB · Freeform",
      pur__single_camels: "PUR · Single",
      pur__multi_camels: "PUR · Multi",
      pur__lagged_streamflow: "PUR · Streamflow",
      pur__freeform: "PUR · Freeform",
    };
    return map[id] || lb.short_title || lb.title || id;
  }

  const FORCING_TAB_META = [
    {
      id: "single_camels",
      name: "Single CAMELS Forcing",
    },
    {
      id: "multi_camels",
      name: "Multiple CAMELS Forcings",
    },
    {
      id: "lagged_streamflow",
      name: "With Lagged Streamflow",
    },
    {
      id: "freeform",
      name: "Freeform Forcings",
    },
  ];

  function esc(val) {
    if (val === null || val === undefined) return "";
    return String(val)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function sanitizeExternalUrl(rawUrl) {
    const s = String(rawUrl || "").trim();
    if (!s || !/^https?:\/\//i.test(s)) return "";
    try {
      const parsed = new URL(s);
      if (parsed.protocol === "https:" || parsed.protocol === "http:") {
        return parsed.href;
      }
    } catch (_e) {
      return "";
    }
    return "";
  }

  function fmtNum(val, digits = 3, asterisk = false) {
    if (val === null || val === undefined || val === "" || Number.isNaN(Number(val))) {
      return '<span style="color: var(--text-muted);">N/A</span>';
    }
    const n = Number(val);
    if (n <= -900) return '<span style="color: var(--text-muted);">N/A</span>';
    return n.toFixed(digits) + (asterisk ? "*" : "");
  }

  function getActiveLeaderboardId() {
    return `${STATE.activeSpatialTab}__${STATE.activeForcingTab}`;
  }

  function selectLeaderboardTab(leaderboardId) {
    if (!leaderboardId || !leaderboardId.includes("__")) return;
    const [sp, fo] = leaderboardId.split("__");
    STATE.activeSpatialTab = sp;
    STATE.activeForcingTab = fo;
    persistNavState();
    renderLeaderboardTabs();
    renderActiveLeaderboardPanel();
  }

  const LOCAL_PREVIEWS_STORAGE_KEY = "hydrologymip_local_previews_v1";
  try {
    sessionStorage.removeItem("hmip_session_submissions_v1");
    sessionStorage.removeItem("hmip_session_submissions_v2");
  } catch (_e) {
    // ignore
  }

  async function parseApiJson(res) {
    const text = await res.text();
    try {
      return JSON.parse(text);
    } catch (_e) {
      const preview = String(text || "")
        .replace(/<[^>]*>/g, " ")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 140);
      throw new Error(
        `Server returned non-JSON response (HTTP ${res.status}): ${preview || "empty response"}`
      );
    }
  }

  function loadLocalPreviews() {
    try {
      const raw = localStorage.getItem(LOCAL_PREVIEWS_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed.filter((x) => x && x.entry && x.entry.entry_id) : [];
    } catch (_e) {
      return [];
    }
  }

  function saveLocalPreviews(items) {
    try {
      localStorage.setItem(LOCAL_PREVIEWS_STORAGE_KEY, JSON.stringify(Array.isArray(items) ? items : []));
    } catch (_e) {
      // ignore localStorage quota errors
    }
  }

  function upsertLocalPreview(record) {
    if (!record || !record.entry || !record.entry.entry_id) return;
    const list = loadLocalPreviews().filter((x) => x.entry.entry_id !== record.entry.entry_id);
    list.push(record);
    saveLocalPreviews(list);
    refreshLeaderboardsWithLocalPreviews();
  }

  function removeLocalPreview(entryId = null) {
    if (!entryId) {
      saveLocalPreviews([]);
      STATE.activePreviewEntryId = null;
    } else {
      const list = loadLocalPreviews().filter((x) => x.entry.entry_id !== entryId);
      saveLocalPreviews(list);
      if (STATE.activePreviewEntryId === entryId) {
        STATE.activePreviewEntryId = null;
      }
    }
    refreshLeaderboardsWithLocalPreviews();
    renderLeaderboardTabs();
    renderActiveLeaderboardPanel();
    renderFamiliesGrid();
  }

  function refreshLeaderboardsWithLocalPreviews() {
    if (!Array.isArray(STATE.baseLeaderboards)) return;
    const lbs = JSON.parse(JSON.stringify(STATE.baseLeaderboards));
    const fams = JSON.parse(JSON.stringify(STATE.baseFamilies || []));
    const previews = loadLocalPreviews();

    for (const item of previews) {
      const entry = { ...item.entry, is_local_preview: true };
      if (!entry.leaderboard_id) continue;
      const lb = lbs.find((b) => b.id === entry.leaderboard_id);
      if (!lb) continue;
      if (!Array.isArray(lb.rows)) lb.rows = [];
      const rIdx = lb.rows.findIndex((r) => r.entry_id === entry.entry_id);
      if (rIdx >= 0) {
        lb.rows[rIdx] = { ...lb.rows[rIdx], ...entry };
      } else {
        lb.rows.push(entry);
      }
      lb.rows.sort((a, b) => {
        const na = typeof a.primary_ranking_nse === "number" ? a.primary_ranking_nse : -999;
        const nb = typeof b.primary_ranking_nse === "number" ? b.primary_ranking_nse : -999;
        if (nb !== na) return nb - na;
        return (a.rank || 9999) - (b.rank || 9999);
      });
      lb.rows.forEach((r, idx) => {
        r.rank = idx + 1;
      });
    }

    STATE.allLeaderboards = lbs;
    STATE.families = fams;
    const totalEntries = lbs.reduce((acc, lb) => acc + ((lb.rows && lb.rows.length) || 0), 0);
    updateHeroStats({ leaderboards: lbs, families: fams, total_entries: totalEntries });
  }

  function computeLocalQuantiles101(vals) {
    const clean = (vals || [])
      .filter((v) => typeof v === "number" && Number.isFinite(v))
      .slice()
      .sort((a, b) => a - b);
    if (!clean.length) return [];
    const out = [];
    const n = clean.length - 1;
    for (let i = 0; i <= 100; i++) {
      const pos = (i / 100) * n;
      const lo = Math.floor(pos);
      const hi = Math.ceil(pos);
      const w = pos - lo;
      const v = clean[lo] * (1 - w) + clean[hi] * w;
      out.push(Math.round(v * 10000) / 10000);
    }
    return out;
  }

  // Fetch all leaderboards and model families
  async function loadAllData() {
    try {
      const res = await fetch("/api/leaderboards");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await parseApiJson(res);
      STATE.baseLeaderboards = data.leaderboards || [];
      STATE.baseFamilies = data.families || [];
      STATE.protocol = data.protocol || null;
      refreshLeaderboardsWithLocalPreviews();

      await loadHydrographIndex();
      renderLeaderboardTabs();
      renderActiveLeaderboardPanel();
      renderFamiliesGrid();
      populateContributeSelects();
      updateCompareUiState();
      if (STATE.activeView === "hydrographs") {
        openHydrographExplorer();
      }
    } catch (err) {
      console.error("Failed to load HydrologyMIP data:", err);
    }
  }

  function updateHeroStats(data) {
    const elLb = document.getElementById("stat-leaderboards-count");
    const elEnt = document.getElementById("stat-entries-count");
    const elFam = document.getElementById("stat-families-count");
    const elHubCount = document.getElementById("hub-camels-us-count");
    const nLbs = (data.leaderboards || []).length;
    const nEnt = data.total_entries || 0;
    const nFam = (data.families || []).length;
    if (elLb) elLb.textContent = String(nLbs);
    if (elEnt) elEnt.textContent = String(nEnt);
    if (elFam) elFam.textContent = String(nFam);
    if (elHubCount) {
      elHubCount.innerHTML = `${nLbs} Leaderboard Tables &bull; ${nEnt} Model Entries &bull; ${nFam} Models`;
    }
  }

  // =========================================================================
  // 1. Render Minimalist Tab Navigation Bars for the 10 Leaderboards
  // =========================================================================
  function renderLeaderboardTabs() {
    // Update Row 1: Spatial Regime Tabs
    document.querySelectorAll("#spatial-tabs-bar .spatial-tab-btn").forEach((btn) => {
      const sp = btn.getAttribute("data-spatial");
      const isAct = sp === STATE.activeSpatialTab;
      btn.classList.toggle("active", isAct);
      btn.setAttribute("aria-selected", isAct ? "true" : "false");
    });

    // Update Row 2: Forcing Sub-Tabs for the active Spatial Regime
    const forcingBar = document.getElementById("forcing-tabs-bar");
    if (forcingBar) {
      let html = `<span class="minimal-tabs-label">Forcing:</span>`;
      for (const fMeta of FORCING_TAB_META) {
        const lbId = `${STATE.activeSpatialTab}__${fMeta.id}`;
        const isAct = fMeta.id === STATE.activeForcingTab;

        html += `
          <button
            type="button"
            role="tab"
            aria-selected="${isAct ? "true" : "false"}"
            class="minimal-tab-btn forcing-tab-btn ${isAct ? "active" : ""}"
            data-forcing="${esc(fMeta.id)}"
            id="forcing-tab-${esc(lbId)}"
          >
            ${esc(fMeta.name)}
          </button>
        `;
      }
      forcingBar.innerHTML = html;

      forcingBar.querySelectorAll(".forcing-tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          STATE.activeForcingTab = btn.getAttribute("data-forcing");
          persistNavState();
          renderLeaderboardTabs();
          renderActiveLeaderboardPanel();
        });
      });
    }
  }

  // =========================================================================
  // 2. Render Buttons for Data Archive & Codebase
  // =========================================================================
  function renderDataArchiveButton(row) {
    const rawArchiveUrl =
      row.data_archive_url ||
      row.zenodo_url ||
      row.hydroshare_url ||
      (row.zenodo_doi ? `https://doi.org/${row.zenodo_doi}` : "");
    const archiveUrl = sanitizeExternalUrl(rawArchiveUrl);

    if (archiveUrl) {
      const tooltip = row.zenodo_doi
        ? `Data Archive DOI: ${row.zenodo_doi} (${archiveUrl})`
        : `Data Archive: ${archiveUrl}`;
      return `
        <a
          href="${esc(archiveUrl)}"
          target="_blank"
          rel="noopener noreferrer"
          class="btn-row-link btn-row-archive"
          title="${esc(tooltip)}"
        >
          Archive &#8599;
        </a>
      `;
    }

    return `<span class="btn-row-absent" title="Literature table result; no standalone hydrograph archive">&mdash;</span>`;
  }

  function renderCodebaseButton(row) {
    const codeUrl = sanitizeExternalUrl(row.code_url || row.github_url || "");
    if (codeUrl) {
      return `
        <a
          href="${esc(codeUrl)}"
          target="_blank"
          rel="noopener noreferrer"
          class="btn-row-link btn-row-code"
          title="Open-Source Codebase: ${esc(codeUrl)}"
        >
          Code &#8599;
        </a>
      `;
    }

    const srcCat = formatSourceCategory(row.category || row.code_provenance);
    if (
      row.closed_source ||
      row.is_closed_source ||
      srcCat === "Contribution" ||
      row.is_user_submission ||
      row.is_local_preview
    ) {
      return `<span class="badge-closed-source" title="Closed-source submission: no public code repository linked">Closed-Source</span>`;
    }

    return `<span class="btn-row-absent" title="No public code repository linked">&mdash;</span>`;
  }

  // =========================================================================
  // 3. Render the Active Leaderboard Tab Panel (Single Table per Tab)
  // =========================================================================
  function formatSourceCategory(rawCat) {
    const s = String(rawCat || "").trim();
    if (!s) return "Contribution";
    if (/archive\*/i.test(s)) return "Archive*";
    if (/archive/i.test(s)) return "Archive";
    if (/re-?run/i.test(s)) return "MIP Re-run";
    if (/contrib/i.test(s)) return "Contribution";
    if (/lit/i.test(s)) return "Literature";
    return s;
  }

  function formatFamilySourceCategories(rawProv) {
    const s = String(rawProv || "").trim();
    const out = [];
    if (/lit/i.test(s)) out.push("Literature");
    if (/re-?run/i.test(s)) out.push("MIP Re-run");
    if (/archive\*/i.test(s)) {
      out.push("Archive*");
    } else if (/archive/i.test(s)) {
      out.push("Archive");
    }
    if (/contrib/i.test(s)) out.push("Contribution");
    return out.length > 0 ? out : [formatSourceCategory(s)];
  }

  function isValidMetric(v) {
    if (v === null || v === undefined || v === "") return false;
    const n = Number(v);
    return !Number.isNaN(n) && n > -900;
  }

  function ensureFlatGauchMetrics(row) {
    if (!row || typeof row !== "object") return row;
    const g = row.gauch_2023_metrics && typeof row.gauch_2023_metrics === "object" ? row.gauch_2023_metrics : {};
    const gmed = (k) => (g[k] && g[k].median !== undefined && g[k].median !== null ? g[k].median : null);
    const gmean = (k) => (g[k] && g[k].mean !== undefined && g[k].mean !== null ? g[k].mean : null);

    if (row.gauch_nse === undefined) {
      row.gauch_nse = gmed("NSE") !== null ? gmed("NSE") : row.primary_ranking_nse;
    }
    if (!isValidMetric(row.mean_nse) && gmean("NSE") !== null) {
      row.mean_nse = gmean("NSE");
    }
    if (row.gauch_kge === undefined) {
      row.gauch_kge = gmed("KGE") !== null ? gmed("KGE") : row.med_kge;
    }
    if (!isValidMetric(row.med_kge) && row.gauch_kge !== null) {
      row.med_kge = row.gauch_kge;
    }
    if (!isValidMetric(row.mean_kge) && gmean("KGE") !== null) {
      row.mean_kge = gmean("KGE");
    }
    if (row.gauch_log_nse === undefined) row.gauch_log_nse = gmed("logNSE");
    if (row.gauch_log_kge === undefined) row.gauch_log_kge = gmed("logKGE");
    if (row.gauch_pearson_r === undefined) row.gauch_pearson_r = gmed("Pearson-r");
    if (row.gauch_alpha_nse === undefined) row.gauch_alpha_nse = gmed("Alpha-NSE");
    if (row.gauch_beta_nse === undefined) row.gauch_beta_nse = gmed("Beta-NSE");
    if (row.gauch_beta_kge === undefined) row.gauch_beta_kge = gmed("Beta-KGE");
    if (row.gauch_mse === undefined) row.gauch_mse = gmed("MSE");
    if (row.gauch_rmse === undefined) row.gauch_rmse = gmed("RMSE");
    if (row.gauch_fhv === undefined) row.gauch_fhv = gmed("FHV");
    if (!isValidMetric(row.med_fhv_abs) && isValidMetric(row.gauch_fhv)) {
      row.med_fhv_abs = Math.abs(Number(row.gauch_fhv));
    }
    if (row.gauch_fms === undefined) row.gauch_fms = gmed("FMS");
    if (row.gauch_flv === undefined) row.gauch_flv = gmed("FLV");
    if (row.gauch_peak_timing === undefined) row.gauch_peak_timing = gmed("Peak-Timing");
    return row;
  }

  const GAUCH_METRIC_COLUMNS = [
    {
      col: "primary_ranking_nse",
      label: "Median NSE",
      title: "Median Nash-Sutcliffe Efficiency (ideal = 1)",
      defaultAsc: false,
      direction: "higher",
      optimalityText: "Higher is better \u2191",
      rawFmt: (v, star) => Number(v).toFixed(3) + (star ? "*" : ""),
      tickFmt: (v) => Number(v).toFixed(2),
      fmt: (v, star) => fmtNum(v, 3, star),
    },
    {
      col: "mean_nse",
      label: "Mean NSE",
      title: "Mean Nash-Sutcliffe Efficiency across 531 basins (ideal = 1)",
      defaultAsc: false,
      direction: "higher",
      optimalityText: "Higher is better \u2191",
      rawFmt: (v, star) => Number(v).toFixed(3) + (star ? "*" : ""),
      tickFmt: (v) => Number(v).toFixed(2),
      fmt: (v, star) => fmtNum(v, 3, star),
    },
    {
      col: "med_kge",
      label: "KGE",
      title: "Median Kling-Gupta Efficiency (ideal = 1)",
      defaultAsc: false,
      direction: "higher",
      optimalityText: "Higher is better \u2191",
      rawFmt: (v, star) => Number(v).toFixed(3) + (star ? "*" : ""),
      tickFmt: (v) => Number(v).toFixed(2),
      fmt: (v, star) => fmtNum(v, 3, star),
    },
    {
      col: "mean_kge",
      label: "Mean KGE",
      title: "Mean Kling-Gupta Efficiency across 531 basins (ideal = 1)",
      defaultAsc: false,
      direction: "higher",
      optimalityText: "Higher is better \u2191",
      rawFmt: (v, star) => Number(v).toFixed(3) + (star ? "*" : ""),
      tickFmt: (v) => Number(v).toFixed(2),
      fmt: (v, star) => fmtNum(v, 3, star),
    },
    {
      col: "gauch_log_nse",
      label: "logNSE",
      title: "Median NSE on log(max(Q, 0) + 1e-5) emphasizing low-flow performance (ideal = 1)",
      defaultAsc: false,
      direction: "higher",
      optimalityText: "Higher is better \u2191",
      rawFmt: (v, star) => Number(v).toFixed(3) + (star ? "*" : ""),
      tickFmt: (v) => Number(v).toFixed(2),
      fmt: (v, star) => fmtNum(v, 3, star),
    },
    {
      col: "gauch_log_kge",
      label: "logKGE",
      title: "Median KGE on log(max(Q, 0) + 1e-5) emphasizing low-flow performance (ideal = 1)",
      defaultAsc: false,
      direction: "higher",
      optimalityText: "Higher is better \u2191",
      rawFmt: (v, star) => Number(v).toFixed(3) + (star ? "*" : ""),
      tickFmt: (v) => Number(v).toFixed(2),
      fmt: (v, star) => fmtNum(v, 3, star),
    },
    {
      col: "gauch_pearson_r",
      label: "Pearson-r",
      title: "Median Pearson correlation coefficient r (ideal = 1)",
      defaultAsc: false,
      direction: "higher",
      optimalityText: "Higher is better \u2191",
      rawFmt: (v, star) => Number(v).toFixed(3) + (star ? "*" : ""),
      tickFmt: (v) => Number(v).toFixed(2),
      fmt: (v, star) => fmtNum(v, 3, star),
    },
    {
      col: "gauch_alpha_nse",
      label: "\u03b1-NSE",
      title: "Median Alpha-NSE: flow variability ratio \u03c3_sim / \u03c3_obs (ideal = 1)",
      defaultAsc: false,
      direction: "target",
      target: 1.0,
      idealLabel: "Ideal = 1.00",
      optimalityText: "Closer to 1.000 is better \u25ce",
      rawFmt: (v, star) => Number(v).toFixed(3) + (star ? "*" : ""),
      tickFmt: (v) => Number(v).toFixed(2),
      fmt: (v, star) => fmtNum(v, 3, star),
    },
    {
      col: "gauch_beta_nse",
      label: "\u03b2-NSE",
      title: "Median Beta-NSE: normalized mean bias (\u03bc_sim - \u03bc_obs) / \u03c3_obs (ideal = 0)",
      defaultAsc: true,
      sortAbs: true,
      direction: "target",
      target: 0.0,
      idealLabel: "Ideal = 0.00",
      optimalityText: "Closer to 0.000 is better \u25ce",
      rawFmt: (v, star) => Number(v).toFixed(3) + (star ? "*" : ""),
      tickFmt: (v) => Number(v).toFixed(2),
      fmt: (v, star) => fmtNum(v, 3, star),
    },
    {
      col: "gauch_beta_kge",
      label: "\u03b2-KGE",
      title: "Median Beta-KGE: mean flow ratio \u03bc_sim / \u03bc_obs (ideal = 1)",
      defaultAsc: false,
      direction: "target",
      target: 1.0,
      idealLabel: "Ideal = 1.00",
      optimalityText: "Closer to 1.000 is better \u25ce",
      rawFmt: (v, star) => Number(v).toFixed(3) + (star ? "*" : ""),
      tickFmt: (v) => Number(v).toFixed(2),
      fmt: (v, star) => fmtNum(v, 3, star),
    },
    {
      col: "gauch_mse",
      label: "MSE",
      title: "Median Mean Squared Error (mm\u00b2/d\u00b2, ideal = 0)",
      defaultAsc: true,
      direction: "lower",
      optimalityText: "Lower is better \u2193",
      rawFmt: (v, star) => Number(v).toFixed(3) + (star ? "*" : ""),
      tickFmt: (v) => Number(v).toFixed(2),
      fmt: (v, star) => fmtNum(v, 3, star),
    },
    {
      col: "gauch_rmse",
      label: "RMSE",
      title: "Median Root Mean Squared Error (mm/d, ideal = 0)",
      defaultAsc: true,
      direction: "lower",
      optimalityText: "Lower is better \u2193",
      rawFmt: (v, star) => Number(v).toFixed(3) + (star ? "*" : ""),
      tickFmt: (v) => Number(v).toFixed(2),
      fmt: (v, star) => fmtNum(v, 3, star),
    },
    {
      col: "gauch_fhv",
      label: "FHV (%)",
      title: "Median top 2% peak flow bias FHV (%, ideal = 0%)",
      defaultAsc: true,
      sortAbs: true,
      direction: "target",
      target: 0.0,
      idealLabel: "Ideal = 0%",
      optimalityText: "Closer to 0% is better \u25ce",
      rawFmt: (v, star) => Number(v).toFixed(1) + "%" + (star ? "*" : ""),
      tickFmt: (v) => Number(v).toFixed(0) + "%",
      fmt: (v, star) => Number(v).toFixed(1) + "%" + (star ? "*" : ""),
    },
    {
      col: "gauch_fms",
      label: "FMS (%)",
      title: "Median FDC mid-segment slope bias FMS (%, ideal = 0%)",
      defaultAsc: true,
      sortAbs: true,
      direction: "target",
      target: 0.0,
      idealLabel: "Ideal = 0%",
      optimalityText: "Closer to 0% is better \u25ce",
      rawFmt: (v, star) => Number(v).toFixed(1) + "%" + (star ? "*" : ""),
      tickFmt: (v) => Number(v).toFixed(0) + "%",
      fmt: (v, star) => Number(v).toFixed(1) + "%" + (star ? "*" : ""),
    },
    {
      col: "gauch_flv",
      label: "FLV (%)",
      title: "Median bottom 30% low-flow bias FLV (%, ideal = 0%)",
      defaultAsc: true,
      sortAbs: true,
      direction: "target",
      target: 0.0,
      idealLabel: "Ideal = 0%",
      optimalityText: "Closer to 0% is better \u25ce",
      rawFmt: (v, star) => Number(v).toFixed(1) + "%" + (star ? "*" : ""),
      tickFmt: (v) => Number(v).toFixed(0) + "%",
      fmt: (v, star) => Number(v).toFixed(1) + "%" + (star ? "*" : ""),
    },
    {
      col: "gauch_peak_timing",
      label: "Peak-Timing (d)",
      title: "Median mean peak-timing error (days, ideal = 0)",
      defaultAsc: true,
      direction: "lower",
      optimalityText: "Lower is better \u2193",
      rawFmt: (v, star) => Number(v).toFixed(2) + " d" + (star ? "*" : ""),
      tickFmt: (v) => Number(v).toFixed(2) + "d",
      fmt: (v, star) => fmtNum(v, 2, star),
    },
    {
      col: "nse_le_0",
      label: "NSE \u2264 0",
      title: "Number of basins (out of 531) with NSE \u2264 0 (ideal = 0)",
      defaultAsc: true,
      direction: "lower",
      optimalityText: "Lower is better \u2193",
      rawFmt: (v, star) => String(Math.round(Number(v))) + (star ? "*" : ""),
      tickFmt: (v) => String(Math.round(Number(v))),
      fmt: (v, star) => String(v) + (star ? "*" : ""),
    },
  ];

  // =========================================================================
  // 3A. Interactive Web Chart Helpers (Year, Paradigm, Pareto Frontier & SVG)
  // =========================================================================
  const MIP_STUDY_YEAR = 2026;
  const SERIES1_FAMILY_YEARS = {
    newman_sacsma: 2015,
    kratzert_ealstm: 2019,
    kratzert_lstm: 2021,
    hoedt_mclstm: 2021,
    kratzert_smtslstm: 2021,
    gauch_xgboost: 2021,
    gauch2021_xgboost: 2021,
    feng_dhbv10: 2022,
    das_tide: 2023,
    das2023_tide: 2023,
    rasiya_koya_tft: 2023,
    nh_tft_rerun: 2023,
    hy2dl_hbv: 2024,
    song_dhbv11p: 2024,
    wang_s4dft: 2025,
    wang2025_s4d: 2025,
    hydrodiffusion: 2025,
    poudel_mlp_hbv: 2026,
    poudel2026_mlp_hbv: 2026,
  };

  const WEBCHART_PARADIGM_META = {
    "Machine learning": {
      label: "Machine Learning",
      shortLabel: "ML",
      color: "#2563eb",
      softBg: "rgba(37, 99, 235, 0.10)",
    },
    "Hybrid (ML + Physics)": {
      label: "Hybrid (Physics + ML)",
      shortLabel: "Hybrid",
      color: "#7c3aed",
      softBg: "rgba(124, 58, 237, 0.10)",
    },
    Conceptual: {
      label: "Conceptual",
      shortLabel: "Conceptual",
      color: "#d97706",
      softBg: "rgba(217, 119, 6, 0.11)",
    },
    "Process-based": {
      label: "Process-Based",
      shortLabel: "Process",
      color: "#0d9488",
      softBg: "rgba(13, 148, 136, 0.11)",
    },
  };

  function classifyRowParadigm(row) {
    const fid = String(row.family_id || "").toLowerCase();
    const paradigm = String(row.paradigm || "").toLowerCase();
    const name = String(row.model_name || "").toLowerCase();
    if (
      ["dhbv", "\u03b4hbv", "mlp_hbv", "mlp-hbv", "16hbv", "dpl", "hy2dl"].some(
        (t) => fid.includes(t) || name.includes(t)
      )
    ) {
      return "Hybrid (ML + Physics)";
    }
    if (row.model_category && WEBCHART_PARADIGM_META[row.model_category]) {
      return String(row.model_category);
    }
    if (
      paradigm.includes("hybrid") &&
      ["hbv", "conceptual", "process", "surrogate", "multi-model"].some(
        (t) => paradigm.includes(t) || name.includes(t)
      )
    ) {
      return "Hybrid (ML + Physics)";
    }
    if (["vic", "mhm", "nwm", "summa"].some((t) => fid.includes(t) || name.includes(t))) {
      return "Process-based";
    }
    if (
      ["sacsma", "sac-sma", "hbv", "fuse", "marrmot", "gr4j"].some(
        (t) => fid.includes(t) || name.includes(t)
      )
    ) {
      return "Conceptual";
    }
    if (paradigm.includes("conceptual")) return "Conceptual";
    if (paradigm.includes("land-surface") || paradigm.includes("distributed process")) {
      return "Process-based";
    }
    return "Machine learning";
  }

  function isMipResultRow(row) {
    const cat = formatSourceCategory(row.category);
    if (cat === "MIP Re-run" || cat === "Contribution") return true;
    if (cat === "Literature" || cat === "Archive" || cat === "Archive*") return false;
    if (row.is_user_submission || row.submitted_at) return true;
    const text = `${row.citation || ""} ${row.model_name || ""}`.toLowerCase();
    return text.includes("this study");
  }

  function extractRowYear(row) {
    const coerceYr = (v) => {
      if (v === null || v === undefined || v === "" || typeof v === "boolean") return null;
      const n = Number(v);
      if (Number.isInteger(n) && n >= 1990 && n <= 2035) return n;
      return null;
    };
    const yearRegex = /(?<!\d)(199[0-9]|20[0-2][0-9]|203[0-5])(?!\d)/g;
    const findYears = (str) => {
      const m = String(str || "").match(yearRegex);
      return m ? m.map(Number) : [];
    };

    const explicit = coerceYr(row.publication_year);
    const nameYears = findYears(row.model_name);
    const citYears = findYears(row.citation);
    const citedYear =
      explicit !== null
        ? explicit
        : nameYears.length > 0
        ? nameYears[0]
        : citYears.length > 0
        ? citYears[0]
        : null;

    const detectSeries1Year = () => {
      if (explicit !== null && explicit !== MIP_STUDY_YEAR) return explicit;
      const allY = [...nameYears, ...citYears];
      const prior = allY.filter((y) => y !== MIP_STUDY_YEAR);
      if (prior.length > 0) return prior[0];
      const fid = String(row.family_id || "").trim();
      if (SERIES1_FAMILY_YEARS[fid]) return SERIES1_FAMILY_YEARS[fid];
      if (explicit !== null) return explicit;
      if (allY.length > 0) return allY[0];
      return null;
    };

    const cat = formatSourceCategory(row.category);
    const s1Year = detectSeries1Year();
    if (
      cat === "MIP Re-run" ||
      (isMipResultRow(row) && cat !== "Contribution" && s1Year !== null)
    ) {
      if (s1Year !== null) return s1Year;
      return MIP_STUDY_YEAR;
    }
    if (isMipResultRow(row)) {
      if (row.submitted_at) {
        const m = String(row.submitted_at).match(/^\s*(\d{4})/);
        if (m) {
          const sy = coerceYr(m[1]);
          if (sy !== null) return sy;
        }
      }
      return citedYear === null ? MIP_STUDY_YEAR : Math.max(MIP_STUDY_YEAR, citedYear);
    }
    return citedYear !== null ? citedYear : MIP_STUDY_YEAR;
  }

  function formatShortLeaderName(row) {
    const raw = String(row.model_name || "").trim();
    let s = raw
      .replace(/^[A-Z][A-Za-z\s,&.\-]+?\(\d{4}[a-z]?(?:,\s*\d{4}[a-z]?)*\)\s*/, "")
      .trim();
    if (!s) {
      s = raw.replace(/\s*\(\d{4}[a-z]?(?:,\s*\d{4}[a-z]?)*\)/, "").trim() || raw;
    }
    s = s
      .replace(/\(\d+-model ensemble\)/gi, "Ens")
      .replace(/\([^,)]+,\s*\d+-model ensemble\)/gi, "Ens")
      .replace(/\(3-Forcing Gauged Ensemble\)/gi, "Ens")
      .replace(/\(\d+-calibration[^)]*ensemble\)/gi, "Ens")
      .replace(/\s+Ensemble\s+Ens/gi, " Ens")
      .replace(/\s*\(Regionalized\)/gi, "")
      .replace(/\s*\(no holdout\)/gi, "")
      .replace(/\s+Lagged Discharge\s+/gi, " ")
      .replace(/\s+/g, " ")
      .trim();
    if (s.length > 22) {
      s = s.slice(0, 20).trim() + "\u2026";
    }
    return s;
  }

  function computeMetricOptimalityScore(mc, val) {
    const n = Number(val);
    if (mc.direction === "lower") return -n;
    if (mc.direction === "target") return -Math.abs(n - (mc.target || 0));
    return n;
  }

  function computeNiceAxisTicks(minVal, maxVal, targetCount = 5) {
    if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return [0, 0.5, 1];
    if (Math.abs(maxVal - minVal) < 1e-6) {
      const pad = Math.abs(minVal) * 0.1 || 0.1;
      minVal -= pad;
      maxVal += pad;
    }
    const rawStep = (maxVal - minVal) / Math.max(2, targetCount - 1);
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const residual = rawStep / mag;
    let step;
    if (residual > 5) step = 10 * mag;
    else if (residual > 2.5) step = 5 * mag;
    else if (residual > 2) step = 2.5 * mag;
    else if (residual > 1) step = 2 * mag;
    else step = mag;

    const niceMin = Math.floor(minVal / step) * step;
    const niceMax = Math.ceil(maxVal / step) * step;
    const ticks = [];
    for (let v = niceMin; v <= niceMax + step * 0.25; v += step) {
      ticks.push(Number(v.toFixed(6)));
    }
    return ticks;
  }

  function renderInteractiveLeaderboardChart(lb, filteredRows, activeSortCol) {
    const chartCol =
      activeSortCol && activeSortCol !== "rank" ? activeSortCol : "primary_ranking_nse";
    const mc = GAUCH_METRIC_COLUMNS.find((c) => c.col === chartCol) || GAUCH_METRIC_COLUMNS[0];

    // Collect plottable points for the active metric
    const paradigmCounts = {};
    const rawPoints = [];
    for (const row of filteredRows) {
      if (!isValidMetric(row[mc.col])) continue;
      const val = Number(row[mc.col]);
      const year = extractRowYear(row);
      const paradigm = classifyRowParadigm(row);
      paradigmCounts[paradigm] = (paradigmCounts[paradigm] || 0) + 1;
      const isMip = isMipResultRow(row);
      const optScore = computeMetricOptimalityScore(mc, val);
      const excludeFrontier = /oracle/i.test(String(row.model_name || ""));
      rawPoints.push({
        row,
        entryId: row.entry_id,
        year,
        val,
        optScore,
        paradigm,
        isMip,
        excludeFrontier,
        isDimmed: Boolean(
          STATE.chartParadigmFilter && STATE.chartParadigmFilter !== paradigm
        ),
      });
    }

    // Build Metric Dropdown options
    const metricSelectOptions = GAUCH_METRIC_COLUMNS.map(
      (c) =>
        `<option value="${esc(c.col)}" ${c.col === mc.col ? "selected" : ""}>${esc(
          c.label
        )}</option>`
    ).join("");

    // Build Paradigm Filter pills
    const activeParadigms = Object.keys(WEBCHART_PARADIGM_META).filter(
      (p) => paradigmCounts[p] > 0
    );
    const paradigmPillsHtml = [
      `<button type="button" class="webchart-paradigm-pill ${
        !STATE.chartParadigmFilter ? "active" : ""
      }" data-chart-paradigm="">All (${rawPoints.length})</button>`,
      ...activeParadigms.map((p) => {
        const meta = WEBCHART_PARADIGM_META[p];
        const isAct = STATE.chartParadigmFilter === p;
        return `
          <button
            type="button"
            class="webchart-paradigm-pill ${isAct ? "active" : ""}"
            data-chart-paradigm="${esc(p)}"
            style="--pill-color: ${meta.color}; --pill-bg: ${meta.softBg};"
          >
            <span class="paradigm-dot" style="background:${meta.color};"></span>
            ${esc(meta.shortLabel)} <span class="paradigm-count">${paradigmCounts[p]}</span>
          </button>`;
      }),
    ].join("");

    if (rawPoints.length === 0) {
      return `
        <div class="lb-figure-stage webchart-stage">
          <div class="webchart-toolbar">
            <div class="webchart-title-group">
              <span class="webchart-eyebrow">Metric over time</span>
              <select class="webchart-metric-select" data-chart-metric-select aria-label="Select plotted metric">
                ${metricSelectOptions}
              </select>
              <span class="webchart-opt-badge">${esc(mc.optimalityText)}</span>
            </div>
          </div>
          <div class="webchart-empty">No models in this view report ${esc(mc.label)}.</div>
        </div>
      `;
    }

    // Compute Pareto frontier across years on non-dimmed eligible points
    const eligible = rawPoints.filter((p) => !p.excludeFrontier && !p.isDimmed);
    const uniqueYears = Array.from(new Set(eligible.map((p) => p.year))).sort((a, b) => a - b);
    const leaders = [];
    const leaderIds = new Set();
    let runningBest = -Infinity;
    for (const yr of uniqueYears) {
      const atYr = eligible.filter((p) => p.year === yr);
      atYr.sort((a, b) => {
        if (Math.abs(b.optScore - a.optScore) > 1e-9) return b.optScore - a.optScore;
        return (a.row.rank || 9999) - (b.row.rank || 9999);
      });
      const bestAtYr = atYr[0];
      if (bestAtYr && bestAtYr.optScore > runningBest + 1e-9) {
        runningBest = bestAtYr.optScore;
        leaders.push(bestAtYr);
        leaderIds.add(bestAtYr.entryId);
      }
    }

    // SVG Coordinate system
    const svgW = 1040;
    const svgH = 315;
    const padLeft = 64;
    const padRight = 32;
    const padTop = 32;
    const padBottom = 38;
    const plotW = svgW - padLeft - padRight;
    const plotH = svgH - padTop - padBottom;

    // X-axis domain (years)
    const minYear = Math.min(2015, ...rawPoints.map((p) => p.year));
    const maxYear = Math.max(MIP_STUDY_YEAR, ...rawPoints.map((p) => p.year));
    const xDomMin = minYear - 0.55;
    const xDomMax = maxYear + 0.65;
    const xToPx = (yr) => padLeft + ((yr - xDomMin) / (xDomMax - xDomMin)) * plotW;

    // Y-axis domain
    const vals = rawPoints.map((p) => p.val);
    let rawMin = Math.min(...vals);
    let rawMax = Math.max(...vals);
    if (mc.direction === "target" && mc.target !== undefined) {
      rawMin = Math.min(rawMin, mc.target);
      rawMax = Math.max(rawMax, mc.target);
    }
    const span = Math.max(1e-4, rawMax - rawMin);
    let yMin = rawMin - span * 0.12;
    let yMax = rawMax + span * 0.16;
    if (
      (mc.col === "nse_le_0" ||
        mc.col === "gauch_mse" ||
        mc.col === "gauch_rmse" ||
        mc.col === "gauch_peak_timing") &&
      rawMin >= 0 &&
      yMin < 0
    ) {
      yMin = 0;
    }
    const yTicks = computeNiceAxisTicks(yMin, yMax, 5);
    const yDomMin = Math.min(yMin, yTicks[0]);
    const yDomMax = Math.max(yMax, yTicks[yTicks.length - 1]);
    const yToPx = (v) =>
      padTop + plotH - ((v - yDomMin) / Math.max(1e-9, yDomMax - yDomMin)) * plotH;

    // Compute pixel coordinates + gentle horizontal micro-jitter for overlapping same-year non-leaders
    const byYear = {};
    for (const p of rawPoints) {
      p.isLeader = leaderIds.has(p.entryId);
      p.baseX = xToPx(p.year);
      p.py = yToPx(p.val);
      p.px = p.baseX;
      if (!byYear[p.year]) byYear[p.year] = [];
      byYear[p.year].push(p);
    }
    for (const yr of Object.keys(byYear)) {
      const grp = byYear[yr].sort((a, b) => a.py - b.py);
      if (grp.length <= 1) continue;
      const jitterOffsets = [0, -9, 9, -17, 17, -24, 24];
      for (let i = 0; i < grp.length; i++) {
        if (grp[i].isLeader) {
          grp[i].px = grp[i].baseX;
          continue;
        }
        let clusterIdx = 0;
        for (let j = 0; j < i; j++) {
          if (Math.abs(grp[i].py - grp[j].py) < 14) {
            clusterIdx++;
          }
        }
        if (clusterIdx > 0) {
          grp[i].px = grp[i].baseX + jitterOffsets[clusterIdx % jitterOffsets.length];
        }
      }
    }

    // Build Gridlines & Y-Ticks SVG
    const yGridSvg = yTicks
      .map((tv) => {
        const y = yToPx(tv);
        if (y < padTop - 4 || y > padTop + plotH + 4) return "";
        return `
          <g class="webchart-y-tick">
            <line x1="${padLeft}" y1="${y.toFixed(1)}" x2="${svgW - padRight}" y2="${y.toFixed(
          1
        )}" class="webchart-grid-line" />
            <text x="${padLeft - 10}" y="${(y + 4).toFixed(
          1
        )}" text-anchor="end" class="webchart-axis-text">${esc(mc.tickFmt(tv))}</text>
          </g>`;
      })
      .join("");

    // Ideal Target Reference Line (if applicable)
    let idealRefSvg = "";
    if (mc.direction === "target" && mc.target !== undefined) {
      const iy = yToPx(mc.target);
      if (iy >= padTop - 2 && iy <= padTop + plotH + 2) {
        idealRefSvg = `
          <g class="webchart-ideal-ref">
            <line x1="${padLeft}" y1="${iy.toFixed(1)}" x2="${svgW - padRight}" y2="${iy.toFixed(
          1
        )}" class="webchart-ideal-line" />
            <rect x="${svgW - padRight - 84}" y="${(iy - 10).toFixed(
          1
        )}" width="80" height="18" rx="4" class="webchart-ideal-pill" />
            <text x="${svgW - padRight - 44}" y="${(iy + 3).toFixed(
          1
        )}" text-anchor="middle" class="webchart-ideal-text">${esc(
          mc.idealLabel || "Ideal"
        )}</text>
          </g>`;
      }
    }

    // Build X-Axis Year Ticks SVG
    const yearTicks = [];
    const yrStep = maxYear - minYear <= 14 ? 1 : 2;
    for (let y = minYear; y <= maxYear; y += yrStep) {
      yearTicks.push(y);
    }
    if (yearTicks[yearTicks.length - 1] !== maxYear) yearTicks.push(maxYear);
    const xGridSvg = yearTicks
      .map((yr) => {
        const x = xToPx(yr);
        return `
          <g class="webchart-x-tick">
            <line x1="${x.toFixed(1)}" y1="${padTop}" x2="${x.toFixed(1)}" y2="${
          padTop + plotH
        }" class="webchart-vgrid-line" />
            <line x1="${x.toFixed(1)}" y1="${padTop + plotH}" x2="${x.toFixed(1)}" y2="${
          padTop + plotH + 5
        }" class="webchart-axis-tick" />
            <text x="${x.toFixed(1)}" y="${
          padTop + plotH + 21
        }" text-anchor="middle" class="webchart-axis-text">${yr}</text>
          </g>`;
      })
      .join("");

    // Build Pareto Frontier Step Line & Soft Shaded Area
    let frontierSvg = "";
    if (leaders.length > 0) {
      const stepPts = [];
      for (let i = 0; i < leaders.length; i++) {
        const curr = leaders[i];
        if (i > 0) {
          stepPts.push([curr.baseX, leaders[i - 1].py]);
        }
        stepPts.push([curr.baseX, curr.py]);
      }
      const endX = xToPx(maxYear + 0.35);
      stepPts.push([endX, leaders[leaders.length - 1].py]);

      const linePath = stepPts
        .map((pt, idx) => `${idx === 0 ? "M" : "L"}${pt[0].toFixed(1)},${pt[1].toFixed(1)}`)
        .join(" ");

      const baselineY =
        mc.direction === "target" && mc.target !== undefined
          ? yToPx(mc.target)
          : padTop + plotH;
      const areaPath = `${linePath} L${endX.toFixed(1)},${baselineY.toFixed(
        1
      )} L${stepPts[0][0].toFixed(1)},${baselineY.toFixed(1)} Z`;

      frontierSvg = `
        <path d="${areaPath}" fill="url(#webchart-frontier-grad)" class="webchart-frontier-area" />
        <path d="${linePath}" class="webchart-frontier-line" />
      `;
    }

    // Build Points SVG (draw dimmed first, then non-leaders, then leaders on top)
    const orderedPoints = [...rawPoints].sort((a, b) => {
      if (a.isDimmed !== b.isDimmed) return a.isDimmed ? -1 : 1;
      if (a.isLeader !== b.isLeader) return a.isLeader ? 1 : -1;
      return a.year - b.year;
    });

    const pointsSvg = orderedPoints
      .map((p) => {
        const pMeta = WEBCHART_PARADIGM_META[p.paradigm] || WEBCHART_PARADIGM_META["Machine learning"];
        const r = p.isLeader ? 7.2 : 5.6;
        const fill = p.isMip ? "#ffffff" : pMeta.color;
        const stroke = p.isMip ? pMeta.color : "#ffffff";
        const strokeW = p.isMip ? 2.4 : 1.6;
        const halo = p.isLeader
          ? `<circle cx="${p.px.toFixed(1)}" cy="${p.py.toFixed(
              1
            )}" r="12.5" fill="${pMeta.color}" opacity="0.16" class="webchart-point-halo" />`
          : "";
        const centerDot =
          p.isMip && p.isLeader
            ? `<circle cx="${p.px.toFixed(1)}" cy="${p.py.toFixed(
                1
              )}" r="2.1" fill="${pMeta.color}" />`
            : "";
        return `
          <g
            class="webchart-point ${p.isLeader ? "is-leader" : ""} ${
          p.isDimmed ? "is-dimmed" : ""
        }"
            data-entry-id="${esc(p.entryId)}"
            tabindex="0"
            role="button"
            aria-label="${esc(p.row.model_name)}: ${esc(mc.label)} ${esc(
          mc.rawFmt(p.val, p.row.asterisk)
        )}"
          >
            ${halo}
            <circle
              cx="${p.px.toFixed(1)}"
              cy="${p.py.toFixed(1)}"
              r="13"
              fill="transparent"
              class="webchart-hit-target"
            />
            <circle
              cx="${p.px.toFixed(1)}"
              cy="${p.py.toFixed(1)}"
              r="${r}"
              fill="${fill}"
              stroke="${stroke}"
              stroke-width="${strokeW}"
              class="webchart-marker"
            />
            ${centerDot}
          </g>`;
      })
      .join("");

    // Build Clean Non-Overlapping Leader Pill Callouts
    const pillBoxes = leaders.map((p, idx) => {
      const shortName = formatShortLeaderName(p.row);
      const valStr = mc.rawFmt(p.val, p.row.asterisk);
      const text = `${shortName} \u00b7 ${valStr}`;
      const w = Math.max(88, Math.min(215, text.length * 6.3 + 18));
      const h = 21;
      const preferAbove = mc.direction === "lower" ? idx % 2 === 1 : idx % 2 === 0 || p.py > padTop + 38;
      let cx = Math.max(padLeft + w / 2 + 4, Math.min(svgW - padRight - w / 2 - 4, p.px));
      let cy = preferAbove ? p.py - 21 : p.py + 21;
      cy = Math.max(14, Math.min(padTop + plotH - 14, cy));
      return {
        p,
        text,
        w,
        h,
        cx,
        cy,
        color: (WEBCHART_PARADIGM_META[p.paradigm] || WEBCHART_PARADIGM_META["Machine learning"]).color,
      };
    });

    // Simple 2D relaxation pass so leader pills never overlap each other
    for (let iter = 0; iter < 18; iter++) {
      for (let i = 0; i < pillBoxes.length; i++) {
        for (let j = i + 1; j < pillBoxes.length; j++) {
          const a = pillBoxes[i];
          const b = pillBoxes[j];
          const dx = b.cx - a.cx;
          const dy = b.cy - a.cy;
          const minDx = (a.w + b.w) / 2 + 6;
          const minDy = (a.h + b.h) / 2 + 4;
          if (Math.abs(dx) < minDx && Math.abs(dy) < minDy) {
            const overlapY = minDy - Math.abs(dy);
            const pushY = (overlapY / 2 + 1) * (dy >= 0 ? 1 : -1);
            a.cy -= pushY;
            b.cy += pushY;
            a.cy = Math.max(13, Math.min(padTop + plotH - 13, a.cy));
            b.cy = Math.max(13, Math.min(padTop + plotH - 13, b.cy));
          }
        }
      }
    }

    const pillsSvg = pillBoxes
      .map((box) => {
        const rx = box.cx - box.w / 2;
        const ry = box.cy - box.h / 2;
        return `
          <g class="webchart-leader-pill" data-entry-id="${esc(box.p.entryId)}">
            <line
              x1="${box.p.px.toFixed(1)}"
              y1="${box.p.py.toFixed(1)}"
              x2="${box.cx.toFixed(1)}"
              y2="${box.cy.toFixed(1)}"
              stroke="${box.color}"
              stroke-width="1.1"
              stroke-dasharray="2,2"
              opacity="0.55"
            />
            <rect
              x="${rx.toFixed(1)}"
              y="${ry.toFixed(1)}"
              width="${box.w.toFixed(1)}"
              height="${box.h}"
              rx="5"
              class="webchart-pill-bg"
              stroke="${box.color}"
            />
            <circle
              cx="${(rx + 9).toFixed(1)}"
              cy="${box.cy.toFixed(1)}"
              r="3"
              fill="${box.color}"
            />
            <text
              x="${(rx + 16).toFixed(1)}"
              y="${(box.cy + 3.8).toFixed(1)}"
              class="webchart-pill-text"
            >${esc(box.text)}</text>
          </g>`;
      })
      .join("");

    return `
      <div class="lb-figure-stage webchart-stage" id="lb-webchart-stage">
        <div class="webchart-toolbar">
          <div class="webchart-title-group">
            <span class="webchart-eyebrow">Benchmark Progress</span>
            <label class="webchart-select-wrap" title="Switch plotted metric (also sorts table)">
              <select class="webchart-metric-select" data-chart-metric-select aria-label="Select plotted metric">
                ${metricSelectOptions}
              </select>
            </label>
            <span class="webchart-opt-badge">${esc(mc.optimalityText)}</span>
            <span class="webchart-frontier-key">
              <span class="frontier-line-swatch"></span> Pareto Frontier
            </span>
          </div>

          <div class="webchart-legend-group">
            <div class="webchart-paradigm-pills">
              ${paradigmPillsHtml}
            </div>
            <div class="webchart-marker-key" title="Solid markers = Literature or Author Archive numbers; Hollow ring markers = MIP Re-run or Community Contribution">
              <span class="marker-key-item"><span class="key-dot-solid"></span> Literature / Archive</span>
              <span class="marker-key-item"><span class="key-dot-ring"></span> MIP Re-run / Contribution</span>
            </div>
          </div>
        </div>

        <div class="webchart-canvas-wrap" id="webchart-canvas-wrap">
          <svg
            viewBox="0 0 ${svgW} ${svgH}"
            class="webchart-svg"
            role="img"
            aria-label="Interactive ${esc(mc.label)} progress chart for ${esc(lb.title)}"
          >
            <defs>
              <linearGradient id="webchart-frontier-grad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#2563eb" stop-opacity="0.14" />
                <stop offset="100%" stop-color="#2563eb" stop-opacity="0.01" />
              </linearGradient>
            </defs>
            ${yGridSvg}
            ${xGridSvg}
            <line
              x1="${padLeft}"
              y1="${padTop + plotH}"
              x2="${svgW - padRight}"
              y2="${padTop + plotH}"
              class="webchart-baseline"
            />
            ${idealRefSvg}
            ${frontierSvg}
            ${pointsSvg}
            ${pillsSvg}
          </svg>
          <div class="webchart-tooltip" id="webchart-tooltip" hidden></div>
        </div>
      </div>
    `;
  }

  const METRIC_PRESETS = {
    all: {
      label: "Full 14 (+ Summary)",
      cols: GAUCH_METRIC_COLUMNS.map((c) => c.col),
    },
    core: {
      label: "Core Metrics",
      cols: [
        "primary_ranking_nse",
        "mean_nse",
        "med_kge",
        "mean_kge",
        "gauch_log_nse",
        "gauch_log_kge",
        "gauch_fhv",
        "nse_le_0",
      ],
    },
    efficiency: {
      label: "Efficiency & Bias",
      cols: [
        "primary_ranking_nse",
        "mean_nse",
        "med_kge",
        "mean_kge",
        "gauch_log_nse",
        "gauch_log_kge",
        "gauch_pearson_r",
        "gauch_alpha_nse",
        "gauch_beta_nse",
        "gauch_beta_kge",
        "gauch_mse",
        "gauch_rmse",
        "nse_le_0",
      ],
    },
    fdc_peaks: {
      label: "FDC & Peaks",
      cols: [
        "primary_ranking_nse",
        "gauch_fhv",
        "gauch_fms",
        "gauch_flv",
        "gauch_peak_timing",
      ],
    },
  };

  function getActiveMetricColumns(activeSortCol) {
    if (!STATE.visibleMetricCols) {
      STATE.visibleMetricCols = new Set(METRIC_PRESETS.all.cols);
    }
    if (activeSortCol && activeSortCol !== "rank") {
      STATE.visibleMetricCols.add(activeSortCol);
    }
    const cols = GAUCH_METRIC_COLUMNS.filter((c) => STATE.visibleMetricCols.has(c.col));
    return cols.length > 0 ? cols : [GAUCH_METRIC_COLUMNS[0]];
  }

  function computeMetricBarPct(col, val) {
    if (!isValidMetric(val)) return 0;
    const n = Number(val);
    if (col === "primary_ranking_nse" || col === "rank") {
      if (n <= 0) return 0;
      return Math.max(5, Math.min(100, ((n - 0.35) / 0.48) * 100));
    }
    if (col === "mean_nse") {
      if (n <= 0) return 0;
      return Math.max(5, Math.min(100, ((n - 0.25) / 0.56) * 100));
    }
    if (col === "med_kge") {
      if (n <= 0) return 0;
      return Math.max(5, Math.min(100, ((n - 0.30) / 0.54) * 100));
    }
    if (col === "mean_kge") {
      if (n <= 0) return 0;
      return Math.max(5, Math.min(100, ((n - 0.25) / 0.56) * 100));
    }
    if (col === "gauch_log_nse" || col === "gauch_log_kge") {
      if (n <= 0) return 0;
      return Math.max(5, Math.min(100, ((n - 0.15) / 0.80) * 100));
    }
    if (col === "gauch_pearson_r") {
      if (n <= 0) return 0;
      return Math.max(5, Math.min(100, ((n - 0.50) / 0.45) * 100));
    }
    if (col === "gauch_alpha_nse") {
      return Math.max(5, Math.min(100, (1 - Math.min(Math.abs(n - 1), 0.5) / 0.5) * 100));
    }
    if (col === "gauch_beta_nse") {
      return Math.max(5, Math.min(100, (1 - Math.min(Math.abs(n), 0.25) / 0.25) * 100));
    }
    if (col === "gauch_beta_kge") {
      return Math.max(5, Math.min(100, (1 - Math.min(Math.abs(n - 1), 0.4) / 0.4) * 100));
    }
    if (col === "gauch_mse") {
      return Math.max(5, Math.min(100, ((3.5 - n) / 2.8) * 100));
    }
    if (col === "gauch_rmse") {
      return Math.max(5, Math.min(100, ((2.0 - n) / 1.2) * 100));
    }
    if (col === "gauch_fhv" || col === "med_fhv_abs") {
      return Math.max(5, Math.min(100, ((65 - Math.abs(n)) / 55) * 100));
    }
    if (col === "gauch_fms") {
      return Math.max(5, Math.min(100, ((50 - Math.abs(n)) / 45) * 100));
    }
    if (col === "gauch_flv") {
      return Math.max(5, Math.min(100, ((120 - Math.abs(n)) / 110) * 100));
    }
    if (col === "gauch_peak_timing") {
      return Math.max(5, Math.min(100, ((1.0 - n) / 0.8) * 100));
    }
    if (col === "nse_le_0") {
      const ratio = Math.max(0, 1 - Math.min(n, 130) / 130);
      return Math.max(5, Math.min(100, Math.pow(ratio, 2.2) * 100));
    }
    return 0;
  }

  function filterRow(row) {
    if (STATE.categoryFilter) {
      const namedCat = formatSourceCategory(row.category);
      if (!namedCat.startsWith(STATE.categoryFilter)) {
        return false;
      }
    }
    if (STATE.searchQuery) {
      const q = STATE.searchQuery.toLowerCase();
      const hay = [
        row.model_name,
        row.family_name,
        row.paradigm,
        row.citation,
        row.forcing,
        formatSourceCategory(row.category),
        row.github_url,
        row.zenodo_doi,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  }

  function renderActiveLeaderboardPanel() {
    const panel = document.getElementById("active-leaderboard-tabpanel");
    if (!panel) return;

    const filterToolbar = document.getElementById("leaderboard-filter-toolbar");
    const activeId = getActiveLeaderboardId();
    if (filterToolbar) filterToolbar.style.display = "";

    const lb = STATE.allLeaderboards.find((b) => b.id === activeId);
    if (!lb) {
      panel.innerHTML = `<div class="empty-board-notice"><p>Select a leaderboard tab above.</p></div>`;
      return;
    }

    const filteredRows = (lb.rows || []).map(ensureFlatGauchMetrics).filter(filterRow);
    const sortSpec = STATE.sortByBoard[lb.id] || { col: "rank", asc: true };
    const activeSortCol = sortSpec.col;
    const activeColMeta = GAUCH_METRIC_COLUMNS.find((c) => c.col === activeSortCol) || null;

    const sortedRows = [...filteredRows].sort((a, b) => {
      const col = sortSpec.col;
      const validA = isValidMetric(a[col]);
      const validB = isValidMetric(b[col]);
      if (!validA && !validB) return (a.rank || 9999) - (b.rank || 9999);
      if (!validA) return 1;
      if (!validB) return -1;
      const rawA = Number(a[col]);
      const rawB = Number(b[col]);
      const numA = activeColMeta && activeColMeta.sortAbs ? Math.abs(rawA) : rawA;
      const numB = activeColMeta && activeColMeta.sortAbs ? Math.abs(rawB) : rawB;
      if (numA !== numB) {
        return sortSpec.asc ? numA - numB : numB - numA;
      }
      return (a.rank || 9999) - (b.rank || 9999);
    });

    function sortThAttrs(col, extraClass = "", extraStyle = "", customTitle = "") {
      const isAct = activeSortCol === col;
      const cls = ["sortable-th", extraClass, isAct ? "active-sort-th" : ""].filter(Boolean).join(" ");
      const st = extraStyle ? ` style="${extraStyle}"` : "";
      const tip = customTitle || "Click to sort by this metric";
      return `class="${cls}"${st} data-sort-col="${col}" title="${esc(tip)}"`;
    }

    function sortArrow(col) {
      if (activeSortCol !== col) return "";
      return `<span class="sort-arrow">${sortSpec.asc ? "&uarr;" : "&darr;"}</span>`;
    }

    const activeFigFile = `figure_${lb.id}.png`;
    const isFigHidden = STATE.figureMode === "hidden";
    const visibleMetrics = getActiveMetricColumns(activeSortCol);
    const allLocalPreviews = loadLocalPreviews();
    const boardPreviewCount = allLocalPreviews.filter(
      (p) => p.entry && p.entry.leaderboard_id === lb.id
    ).length;

    const localPreviewBannerHtml =
      allLocalPreviews.length > 0
        ? `
      <div class="local-preview-banner" id="local-preview-banner" role="status">
        <div class="local-preview-banner-body">
          <div class="local-preview-banner-title">
            <span class="local-preview-badge">LOCAL PREVIEW &middot; PENDING PR</span>
            <span>${allLocalPreviews.length} local preview ${allLocalPreviews.length === 1 ? "entry" : "entries"} active (${boardPreviewCount} on this table)</span>
          </div>
          <div>
            Local Preview Only &mdash; Your preview model is stored only in this browser so you can inspect its rank, metrics, and hydrographs right away. Other visitors cannot see it, and clearing browser storage or switching devices will remove it. To publish permanently to the official leaderboard, submit a GitHub PR below.
          </div>
        </div>
        <div class="local-preview-banner-actions">
          <button type="button" class="btn-primary-cta" data-preview-submit-pr="1" style="padding: 0.36rem 0.75rem; font-size: 0.76rem;">
            Submit Official GitHub PR
          </button>
          <button type="button" class="btn-secondary" data-preview-download-manifest="1" style="padding: 0.34rem 0.65rem; font-size: 0.76rem;">
            Download Manifest JSON
          </button>
          <button type="button" class="btn-danger-ghost" data-preview-clear-all="1">
            Clear Preview
          </button>
        </div>
      </div>`
        : "";

    const presetBtnsHtml = Object.entries(METRIC_PRESETS)
      .map(
        ([pKey, pDef]) => `
          <button
            type="button"
            class="metric-preset-btn ${STATE.metricPreset === pKey ? "active" : ""}"
            data-metric-preset="${esc(pKey)}"
          >
            ${esc(pDef.label)}
          </button>`
      )
      .join("");

    const metricChipsHtml = GAUCH_METRIC_COLUMNS.map((mc) => {
      const isOn = STATE.visibleMetricCols && STATE.visibleMetricCols.has(mc.col);
      return `
        <button
          type="button"
          class="metric-chip ${isOn ? "active" : ""}"
          data-metric-toggle="${esc(mc.col)}"
          title="${esc(mc.title)} (click to show/hide column)"
        >
          ${esc(mc.label)}
        </button>`;
    }).join("");

    const metricHeaderCells = visibleMetrics
      .map((mc) => `<th ${sortThAttrs(mc.col, "num-col", "", mc.title)}>${esc(mc.label)}${sortArrow(mc.col)}</th>`)
      .join("\n              ");

    let html = `
      <div class="leaderboard-card-header">
        <div class="lb-title-wrap">
          <h3 class="lb-panel-title">${esc(lb.title)}</h3>
        </div>

        <div class="lb-header-controls">
          <a
            href="/api/figures/${esc(activeFigFile)}"
            target="_blank"
            rel="noopener noreferrer"
            class="btn-row-link"
            title="Open static publication PNG figure (${esc(activeFigFile)})"
          >
            Static PNG &#8599;
          </a>
          <button
            type="button"
            data-fig-toggle="1"
            class="btn-row-link"
          >
            ${isFigHidden ? "Show Plot" : "Hide Plot"}
          </button>
        </div>
      </div>

      ${localPreviewBannerHtml}
      ${!isFigHidden ? renderInteractiveLeaderboardChart(lb, filteredRows, activeSortCol) : ""}

      <div class="metric-selector-bar">
        <div class="metric-presets-row">
          <div class="metric-presets-left">
            <span class="metric-bar-label">Metric View:</span>
            ${presetBtnsHtml}
          </div>
          <span class="metric-bar-label" style="margin-right: 0;">Click any metric below to toggle column &middot; Click column header to sort</span>
        </div>
        <div class="metric-chips-row">
          ${metricChipsHtml}
        </div>
      </div>

      <div class="table-top-scroll-bar" id="lb-top-scroll-bar">
        <span class="metric-bar-label" style="margin-right: 0.2rem;">Scroll Columns:</span>
        <div class="scroll-nudge-group">
          <button type="button" class="scroll-nudge-btn" data-scroll-dir="-1" title="Scroll table left">&larr;</button>
          <button type="button" class="scroll-nudge-btn" data-scroll-dir="1" title="Scroll table right">&rarr;</button>
        </div>
        <div class="table-top-scroll" id="lb-top-scroll">
          <div class="table-top-scroll-inner" id="lb-top-scroll-inner"></div>
        </div>
      </div>

      <div class="table-scroll-wrap" id="lb-table-scroll-wrap">
        <table class="lb-table" id="lb-active-table" data-lb-id="${esc(lb.id)}">
          <thead>
            <tr>
              <th ${sortThAttrs("rank", "sticky-col-rank", "width: 48px;", "Rank by Median NSE")}>#${sortArrow("rank")}</th>
              <th class="sticky-col-model" style="min-width: 220px;">Model</th>
              <th title="Select one or more models in this leaderboard or across leaderboards to compare daily hydrographs">Hydrographs</th>
              <th>Source</th>
              <th>Forcing</th>
              ${metricHeaderCells}
              <th>Archive</th>
              <th>Code</th>
            </tr>
          </thead>
          <tbody>
    `;

    const totalCols = 7 + visibleMetrics.length;

    if (sortedRows.length === 0) {
      html += `
        <tr>
          <td colspan="${totalCols}" class="empty-table-cell">
            <div class="empty-table-inner">
              <strong>No entries in this leaderboard yet</strong>
              <button type="button" class="btn-secondary btn-contribute-to-this-tab" data-lb-id="${esc(lb.id)}">
                + Contribute Model
              </button>
            </div>
          </td>
        </tr>
      `;
    } else {
      for (const row of sortedRows) {
        const rankClass =
          row.rank === 1
            ? "rank-1"
            : row.rank === 2
            ? "rank-2"
            : row.rank === 3
            ? "rank-3"
            : "";
        const isSelected = STATE.inspectedEntryId === row.entry_id;
        const star = Boolean(row.asterisk);
        const namedCategory = formatSourceCategory(row.category);
        const modelKey = row.model_key || `${lb.id}__${row.entry_id}`;
        const hasHydro = isModelHydrographAvailable(row, lb.id);
        const isComparing = STATE.comparedModels.includes(modelKey);

        const hydroCompareCell = hasHydro
          ? `<div style="display: inline-flex; align-items: center; gap: 0.28rem;">
              <button
                type="button"
                class="btn-compare-toggle ${isComparing ? "comparing" : ""}"
                data-compare-model-key="${esc(modelKey)}"
                title="${isComparing ? "Remove from hydrograph comparison" : "Select model to compare hydrographs (supports multiple models across leaderboards)"}"
              >
                ${isComparing ? "&#10003; Comparing" : "+ Compare"}
              </button>
              <button
                type="button"
                class="btn-row-link"
                data-open-hydro-model="${esc(modelKey)}"
                title="View daily hydrographs for selected models"
                style="padding: 0.2rem 0.42rem;"
              >
                Plot &rarr;
              </button>
            </div>`
          : `<span class="btn-row-absent" title="Literature summary metrics only; daily hydrograph time series not in archive">&mdash;</span>`;

        const metricBodyCells = visibleMetrics.map((mc) => {
          const val = row[mc.col];
          const hasVal = isValidMetric(val);
          const showBar =
            hasVal &&
            (activeSortCol === mc.col ||
              (mc.col === "primary_ranking_nse" && activeSortCol === "rank"));
          const barPct = showBar ? computeMetricBarPct(mc.col, val) : 0;
          const displayHtml = hasVal
            ? esc(mc.fmt(val, star))
            : '<span class="muted-dash">&mdash;</span>';
          const barHtml =
            barPct > 0
              ? `<div class="nse-bar-track"><div class="nse-bar-fill" style="width: ${barPct.toFixed(1)}%;"></div></div>`
              : "";
          return `
            <td class="num-col">
              <div class="nse-score-wrap">
                <span class="${showBar ? "nse-primary-val" : ""}">${displayHtml}</span>
                ${barHtml}
              </div>
            </td>`;
        }).join("");

        const isPreview = Boolean(row.is_local_preview);
        const previewBadgeAndActions = isPreview
          ? `<div style="margin-top: 0.28rem;">
              <span class="local-preview-badge">LOCAL PREVIEW &middot; PENDING PR</span>
              <div class="preview-row-actions">
                <button type="button" class="btn-preview-inline" data-preview-submit-row="${esc(row.entry_id)}">Submit Official GitHub PR</button>
                <button type="button" class="btn-preview-inline" data-preview-download-row="${esc(row.entry_id)}">Download Manifest JSON</button>
                <button type="button" class="btn-preview-inline" data-preview-clear-row="${esc(row.entry_id)}">Clear Preview</button>
              </div>
            </div>`
          : "";

        html += `
          <tr class="${isSelected ? "selected-row" : ""} ${isComparing ? "comparing-row" : ""} ${isPreview ? "local-preview-row" : ""}" id="row-${esc(row.entry_id)}">
            <td class="sticky-col-rank">
              <span class="rank-badge ${rankClass}">${row.rank}</span>
            </td>
            <td class="sticky-col-model">
              <div class="model-cell-main">
                <button
                  type="button"
                  class="model-name-btn"
                  data-inspect-entry="${esc(row.entry_id)}"
                  title="${row.asterisk_reason ? esc(row.asterisk_reason) + " — Click to inspect 531-basin CDF & regional breakdown" : "Click to inspect 531-basin CDF & regional breakdown"}"
                >
                  ${esc(row.model_name)}
                </button>
                ${row.citation ? `<span class="model-cite-inline">${esc(row.citation)}</span>` : ""}
                ${previewBadgeAndActions}
              </div>
            </td>
            <td>
              ${hydroCompareCell}
            </td>
            <td>
              <span class="cat-badge">${esc(namedCategory)}</span>
            </td>
            <td class="forcing-cell">
              ${esc(row.forcing)}
            </td>
            ${metricBodyCells}
            <td>
              ${renderDataArchiveButton(row)}
            </td>
            <td>
              ${renderCodebaseButton(row)}
            </td>
          </tr>
        `;
      }
    }

    const footnoteRows = sortedRows.filter((r) => r.asterisk || r.asterisk_reason);
    const footnotesHtml =
      footnoteRows.length > 0
        ? `
      <div class="lb-footnotes-box" style="margin-top: 0.75rem; padding: 0.75rem 1rem; border-radius: 8px; background: var(--surface-subtle, rgba(15, 23, 42, 0.03)); border: 1px solid var(--border-subtle, rgba(15, 23, 42, 0.09)); font-size: 0.8rem; line-height: 1.45; color: var(--text-secondary);">
        <div style="font-weight: 700; color: var(--text-primary); margin-bottom: 0.35rem;">Model-Specific Footnotes (<code>*</code>)</div>
        <ul style="margin: 0; padding-left: 1.15rem; display: flex; flex-direction: column; gap: 0.22rem;">
          ${footnoteRows
            .map((r) => {
              const reasonText =
                r.asterisk_reason ||
                (r.num_evaluated_basins && r.num_evaluated_basins < 531
                  ? `Evaluated on ${r.num_evaluated_basins} of 531 CAMELS basins.`
                  : "Non-standard evaluation subset or period.");
              return `<li><strong>${esc(r.model_name)}</strong>${r.citation ? ` (${esc(r.citation)})` : ""}: ${esc(reasonText)}</li>`;
            })
            .join("")}
        </ul>
      </div>`
        : "";

    html += `
          </tbody>
        </table>
      </div>
      ${footnotesHtml}
    `;

    panel.innerHTML = html;

    // Bind Metric Preset buttons
    panel.querySelectorAll("[data-metric-preset]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const pKey = btn.getAttribute("data-metric-preset");
        if (METRIC_PRESETS[pKey]) {
          STATE.metricPreset = pKey;
          STATE.visibleMetricCols = new Set(METRIC_PRESETS[pKey].cols);
          persistNavState();
          renderActiveLeaderboardPanel();
        }
      });
    });

    // Bind individual Metric Chip toggles
    panel.querySelectorAll("[data-metric-toggle]").forEach((chip) => {
      chip.addEventListener("click", () => {
        const col = chip.getAttribute("data-metric-toggle");
        if (!STATE.visibleMetricCols) {
          STATE.visibleMetricCols = new Set(METRIC_PRESETS.all.cols);
        }
        if (STATE.visibleMetricCols.has(col)) {
          if (STATE.visibleMetricCols.size > 1) {
            STATE.visibleMetricCols.delete(col);
          }
        } else {
          STATE.visibleMetricCols.add(col);
        }
        STATE.metricPreset = "custom";
        renderActiveLeaderboardPanel();
      });
    });

    // Sync top horizontal scrollbar and left/right nudge buttons with main table container
    const topScrollBar = document.getElementById("lb-top-scroll-bar");
    const topScroll = document.getElementById("lb-top-scroll");
    const topScrollInner = document.getElementById("lb-top-scroll-inner");
    const tableWrap = document.getElementById("lb-table-scroll-wrap");
    const activeTable = document.getElementById("lb-active-table");
    if (topScroll && topScrollInner && tableWrap && activeTable) {
      const syncWidths = () => {
        const sw = activeTable.scrollWidth;
        const cw = tableWrap.clientWidth;
        topScrollInner.style.width = `${sw}px`;
        if (topScrollBar) {
          topScrollBar.style.display = sw > cw + 4 ? "flex" : "none";
        }
      };
      syncWidths();
      requestAnimationFrame(syncWidths);

      let syncingTop = false;
      let syncingBottom = false;
      topScroll.addEventListener("scroll", () => {
        if (syncingTop) return;
        syncingBottom = true;
        tableWrap.scrollLeft = topScroll.scrollLeft;
        syncingBottom = false;
      });
      tableWrap.addEventListener("scroll", () => {
        if (syncingBottom) return;
        syncingTop = true;
        topScroll.scrollLeft = tableWrap.scrollLeft;
        syncingTop = false;
      });

      panel.querySelectorAll("[data-scroll-dir]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const dir = Number(btn.getAttribute("data-scroll-dir")) || 1;
          tableWrap.scrollBy({ left: dir * 320, behavior: "smooth" });
        });
      });
    }

    // Bind column sorting
    panel.querySelectorAll("th[data-sort-col]").forEach((th) => {
      th.addEventListener("click", () => {
        const col = th.getAttribute("data-sort-col");
        const curr = STATE.sortByBoard[lb.id] || { col: "rank", asc: true };
        if (curr.col === col) {
          curr.asc = !curr.asc;
        } else {
          const mc = GAUCH_METRIC_COLUMNS.find((c) => c.col === col);
          curr.col = col;
          curr.asc = col === "rank" ? true : mc ? Boolean(mc.defaultAsc) : false;
        }
        STATE.sortByBoard[lb.id] = curr;
        renderActiveLeaderboardPanel();
      });
    });

    // Bind "+ Contribute Model" buttons -> switch to Contribute tab pre-selecting spatial regime & forcing
    panel.querySelectorAll(".btn-contribute-to-this-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetLbId = btn.getAttribute("data-lb-id") || "";
        const [sp, fo] = targetLbId.split("__");
        const spSel = document.getElementById("sub-spatial-regime");
        const foSel = document.getElementById("sub-forcing");
        if (spSel && sp) spSel.value = sp;
        if (foSel && fo) {
          const defaultForcingByCat = {
            single_camels: "Daymet",
            multi_camels: "Daymet + Maurer + NLDAS",
            lagged_streamflow: "Daymet + Lagged Streamflow",
            freeform: "Freeform (None of the above)",
          };
          if (defaultForcingByCat[fo]) foSel.value = defaultForcingByCat[fo];
        }
        syncContributeTargetLeaderboard();
        switchMainTab("contribute");
      });
    });

    // Bind 531-Basin CDF inspector buttons
    panel.querySelectorAll("[data-inspect-entry]").forEach((btn) => {
      btn.addEventListener("click", () => {
        inspectEntryBasins(btn.getAttribute("data-inspect-entry"));
      });
    });

    // Bind Local Preview Banner & Row Actions
    const getPreviewRecordById = (eid) => {
      const list = loadLocalPreviews();
      if (eid) return list.find((p) => p.entry && p.entry.entry_id === eid) || null;
      return (
        list.find((p) => p.entry && p.entry.leaderboard_id === lb.id) ||
        list[list.length - 1] ||
        null
      );
    };
    const openPreviewInStep4 = (eid) => {
      const rec = getPreviewRecordById(eid);
      if (rec) loadPreviewRecordIntoContribute(rec);
      setContribStep(4);
      switchMainTab("contribute");
    };
    const downloadPreviewManifest = (eid) => {
      const rec = getPreviewRecordById(eid);
      if (rec && rec.manifest) {
        downloadManifestJsonObject(rec.manifest);
      } else {
        exportPullRequestManifest();
      }
    };

    panel.querySelectorAll("[data-preview-submit-pr]").forEach((btn) => {
      btn.addEventListener("click", () => openPreviewInStep4(null));
    });
    panel.querySelectorAll("[data-preview-download-manifest]").forEach((btn) => {
      btn.addEventListener("click", () => downloadPreviewManifest(null));
    });
    panel.querySelectorAll("[data-preview-clear-all]").forEach((btn) => {
      btn.addEventListener("click", () => removeLocalPreview(null));
    });
    panel.querySelectorAll("[data-preview-submit-row]").forEach((btn) => {
      btn.addEventListener("click", () => openPreviewInStep4(btn.getAttribute("data-preview-submit-row")));
    });
    panel.querySelectorAll("[data-preview-download-row]").forEach((btn) => {
      btn.addEventListener("click", () => downloadPreviewManifest(btn.getAttribute("data-preview-download-row")));
    });
    panel.querySelectorAll("[data-preview-clear-row]").forEach((btn) => {
      btn.addEventListener("click", () => removeLocalPreview(btn.getAttribute("data-preview-clear-row")));
    });

    // Bind Hydrograph "+ Compare" toggle buttons
    panel.querySelectorAll("[data-compare-model-key]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const modelKey = btn.getAttribute("data-compare-model-key");
        toggleModelInHydroComparison(modelKey);
      });
    });

    // Bind Hydrograph "Plot ->" quick-open buttons
    panel.querySelectorAll("[data-open-hydro-model]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const modelKey = btn.getAttribute("data-open-hydro-model");
        if (modelKey && !STATE.comparedModels.includes(modelKey)) {
          if (STATE.comparedModels.length >= 8) {
            STATE.comparedModels.shift();
          }
          STATE.comparedModels.push(modelKey);
        }
        updateCompareUiState();
        openHydrographExplorer();
      });
    });

    // Bind Figure show/hide toggle button
    panel.querySelectorAll("[data-fig-toggle]").forEach((btn) => {
      btn.addEventListener("click", () => {
        STATE.figureMode = STATE.figureMode === "hidden" ? "table" : "hidden";
        renderActiveLeaderboardPanel();
      });
    });

    // Bind Interactive Web Chart events (metric selector, paradigm filter, tooltip, row sync)
    const chartMetricSel = panel.querySelector("[data-chart-metric-select]");
    if (chartMetricSel) {
      chartMetricSel.addEventListener("change", () => {
        const col = chartMetricSel.value;
        const mc = GAUCH_METRIC_COLUMNS.find((c) => c.col === col);
        if (mc) {
          if (!STATE.visibleMetricCols) {
            STATE.visibleMetricCols = new Set(METRIC_PRESETS.all.cols);
          }
          STATE.visibleMetricCols.add(col);
          STATE.sortByBoard[lb.id] = { col, asc: Boolean(mc.defaultAsc) };
          renderActiveLeaderboardPanel();
        }
      });
    }

    panel.querySelectorAll("[data-chart-paradigm]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const val = btn.getAttribute("data-chart-paradigm") || "";
        STATE.chartParadigmFilter = STATE.chartParadigmFilter === val ? "" : val;
        renderActiveLeaderboardPanel();
      });
    });

    const canvasWrap = document.getElementById("webchart-canvas-wrap");
    const tooltipEl = document.getElementById("webchart-tooltip");
    if (canvasWrap && tooltipEl) {
      const chartCol =
        activeSortCol && activeSortCol !== "rank" ? activeSortCol : "primary_ranking_nse";
      const mc = GAUCH_METRIC_COLUMNS.find((c) => c.col === chartCol) || GAUCH_METRIC_COLUMNS[0];
      const rowById = new Map(filteredRows.map((r) => [String(r.entry_id), r]));

      const showPointTooltip = (entryId, clientX, clientY) => {
        const row = rowById.get(String(entryId));
        if (!row) return;
        const paradigm = classifyRowParadigm(row);
        const pMeta = WEBCHART_PARADIGM_META[paradigm] || WEBCHART_PARADIGM_META["Machine learning"];
        const year = extractRowYear(row);
        const namedCat = formatSourceCategory(row.category);
        const star = Boolean(row.asterisk);
        const activeValStr = isValidMetric(row[mc.col])
          ? mc.rawFmt(row[mc.col], star)
          : "N/A";
        const medNseStr = isValidMetric(row.primary_ranking_nse)
          ? Number(row.primary_ranking_nse).toFixed(3) + (star ? "*" : "")
          : "N/A";
        const kgeStr = isValidMetric(row.med_kge)
          ? Number(row.med_kge).toFixed(3) + (star ? "*" : "")
          : "N/A";

        tooltipEl.innerHTML = `
          <div class="wct-head">
            <span class="wct-rank">#${esc(row.rank)}</span>
            <span class="wct-name">${esc(row.model_name)}</span>
          </div>
          <div class="wct-sub">
            ${row.citation ? `${esc(row.citation)} &middot; ` : ""}Plotted Year: <strong>${year}</strong>
          </div>
          <div class="wct-badges">
            <span class="wct-badge" style="color:${pMeta.color}; background:${pMeta.softBg};">
              <span class="paradigm-dot" style="background:${pMeta.color};"></span>
              ${esc(pMeta.label)}
            </span>
            <span class="wct-badge wct-badge-neutral">${esc(namedCat)}</span>
            <span class="wct-badge wct-badge-neutral">${esc(row.forcing)}</span>
          </div>
          <div class="wct-metrics">
            <div class="wct-metric-primary">
              <span class="wct-m-label">${esc(mc.label)}</span>
              <span class="wct-m-val">${esc(activeValStr)}</span>
            </div>
            ${
              mc.col !== "primary_ranking_nse"
                ? `<div class="wct-metric-sec"><span class="wct-m-label">Median NSE</span><span class="wct-m-val">${esc(medNseStr)}</span></div>`
                : ""
            }
            ${
              mc.col !== "med_kge"
                ? `<div class="wct-metric-sec"><span class="wct-m-label">KGE</span><span class="wct-m-val">${esc(kgeStr)}</span></div>`
                : ""
            }
          </div>
          ${
            row.asterisk_reason
              ? `<div class="wct-sub" style="margin-top:0.35rem; font-size:0.74rem; color:var(--text-secondary);">* ${esc(row.asterisk_reason)}</div>`
              : ""
          }
          <div class="wct-hint">Click marker to highlight row in table &darr;</div>
        `;
        tooltipEl.hidden = false;

        const rect = canvasWrap.getBoundingClientRect();
        const relX = clientX - rect.left;
        const relY = clientY - rect.top;
        const ttW = tooltipEl.offsetWidth || 270;
        const ttH = tooltipEl.offsetHeight || 140;
        let left = relX + 16;
        if (left + ttW > rect.width - 8) {
          left = Math.max(8, relX - ttW - 16);
        }
        let top = relY - ttH / 2;
        top = Math.max(8, Math.min(rect.height - ttH - 8, top));
        tooltipEl.style.left = `${left}px`;
        tooltipEl.style.top = `${top}px`;

        panel.querySelectorAll(".webchart-point, .webchart-leader-pill").forEach((el) => {
          el.classList.toggle("is-hovered", el.getAttribute("data-entry-id") === String(entryId));
        });
        panel.querySelectorAll("tbody tr[id^='row-']").forEach((tr) => {
          tr.classList.toggle("chart-hover-row", tr.id === `row-${entryId}`);
        });
      };

      const hidePointTooltip = () => {
        tooltipEl.hidden = true;
        panel.querySelectorAll(".webchart-point.is-hovered, .webchart-leader-pill.is-hovered").forEach((el) => {
          el.classList.remove("is-hovered");
        });
        panel.querySelectorAll("tbody tr.chart-hover-row").forEach((tr) => {
          tr.classList.remove("chart-hover-row");
        });
      };

      const jumpToTableRow = (entryId) => {
        const targetTr = document.getElementById(`row-${entryId}`);
        if (!targetTr) return;
        targetTr.scrollIntoView({ behavior: "smooth", block: "center" });
        targetTr.classList.remove("pulse-highlight-row");
        void targetTr.offsetWidth;
        targetTr.classList.add("pulse-highlight-row");
        setTimeout(() => targetTr.classList.remove("pulse-highlight-row"), 2200);
      };

      panel.querySelectorAll(".webchart-point, .webchart-leader-pill").forEach((el) => {
        const entryId = el.getAttribute("data-entry-id");
        el.addEventListener("mouseenter", (ev) => {
          showPointTooltip(entryId, ev.clientX, ev.clientY);
        });
        el.addEventListener("mousemove", (ev) => {
          showPointTooltip(entryId, ev.clientX, ev.clientY);
        });
        el.addEventListener("mouseleave", hidePointTooltip);
        el.addEventListener("click", () => {
          jumpToTableRow(entryId);
        });
      });

      // Bidirectional sync: hovering a table row highlights its marker in the chart
      panel.querySelectorAll("tbody tr[id^='row-']").forEach((tr) => {
        const entryId = tr.id.replace(/^row-/, "");
        tr.addEventListener("mouseenter", () => {
          panel.querySelectorAll(".webchart-point, .webchart-leader-pill").forEach((el) => {
            el.classList.toggle("is-hovered", el.getAttribute("data-entry-id") === entryId);
          });
        });
        tr.addEventListener("mouseleave", () => {
          panel.querySelectorAll(".webchart-point.is-hovered, .webchart-leader-pill.is-hovered").forEach((el) => {
            el.classList.remove("is-hovered");
          });
        });
      });
    }
  }

  // =========================================================================
  // 4. Interactive 531-Basin Empirical NSE CDF & HUC-02 Regional Inspector
  // =========================================================================
  async function inspectEntryBasins(entryId) {
    try {
      STATE.inspectedEntryId = entryId;
      renderActiveLeaderboardPanel();

      let localFoundRow = null;
      for (const lb of STATE.allLeaderboards || []) {
        for (const r of lb.rows || []) {
          if (r.entry_id === entryId) {
            localFoundRow = r;
            break;
          }
        }
        if (localFoundRow) break;
      }

      let data = null;
      if (!localFoundRow || !localFoundRow.is_local_preview) {
        const res = await fetch(`/api/entry/${encodeURIComponent(entryId)}/basins`);
        if (res.ok) {
          data = await parseApiJson(res);
        }
      }
      if (!data) {
        if (!localFoundRow) throw new Error(`Entry '${entryId}' not found`);
        const hucRows = [];
        if (localFoundRow.regional_huc02 && typeof localFoundRow.regional_huc02 === "object") {
          for (const hucKey of Object.keys(localFoundRow.regional_huc02).sort()) {
            const hd = localFoundRow.regional_huc02[hucKey] || {};
            hucRows.push({
              huc_02: hucKey,
              n_basins: hd.num_basins ?? hd.n_basins ?? 0,
              median_NSE: hd.median_nse ?? hd.median_NSE ?? null,
              mean_NSE: hd.mean_nse ?? hd.mean_NSE ?? null,
              median_KGE: hd.median_kge ?? hd.median_KGE ?? null,
              n_NSE_le_0: hd.nse_le_zero_count ?? hd.n_NSE_le_0 ?? 0,
            });
          }
        }
        data = {
          entry: localFoundRow,
          source:
            Array.isArray(localFoundRow.per_basin_nse) && localFoundRow.per_basin_nse.length > 0
              ? "empirical_531_basins"
              : "summary_anchored_distribution",
          nse_cdf_101: computeLocalQuantiles101(localFoundRow.per_basin_nse || []),
          huc_02_breakdown: hucRows,
        };
      }

      const drawer = document.getElementById("basin-inspector-drawer");
      if (!drawer) return;
      drawer.classList.add("open");

      const entry = data.entry || {};
      const lbDef = STATE.allLeaderboards.find((b) => b.id === entry.leaderboard_id);
      const fam = STATE.families.find((f) => f.family_id === entry.family_id) || {};
      const enrichedEntry = {
        ...entry,
        github_url: entry.github_url || fam.github_url || "",
        zenodo_doi: entry.zenodo_doi || fam.zenodo_doi || "",
        zenodo_url: entry.zenodo_url || fam.zenodo_url || "",
        hydroshare_url: entry.hydroshare_url || fam.hydroshare_url || "",
      };

      document.getElementById("insp-badge-leaderboard").textContent = lbDef
        ? formatLeaderboardShortLabel(lbDef)
        : entry.leaderboard_id;
      document.getElementById("insp-badge-source").textContent =
        data.source === "empirical_531_basins"
          ? "Verified 531-Basin Archive Hydrographs"
          : "Summary-Anchored 531-Basin Distribution";
      document.getElementById("insp-badge-category").textContent = formatSourceCategory(
        entry.category || "Contribution"
      );
      document.getElementById("insp-model-title").textContent = `${entry.model_name} (${entry.forcing})`;
      document.getElementById("insp-model-subtitle").textContent =
        `Median NSE: ${entry.primary_ranking_nse > -100 ? Number(entry.primary_ranking_nse).toFixed(3) : "N/A"} • ` +
        `Median KGE: ${entry.med_kge != null ? Number(entry.med_kge).toFixed(3) : "N/A"} • ` +
        `Evaluated on 531 CAMELS-US basins (1989-10-01 to 1999-09-30)`;

      const repoCell = document.getElementById("insp-repo-links");
      if (repoCell) {
        repoCell.innerHTML =
          renderDataArchiveButton(enrichedEntry) + renderCodebaseButton(enrichedEntry);
      }

      renderSvgCdfChart(data.nse_cdf_101 || []);
      renderHucBreakdownTable(data.huc_02_breakdown || []);

      drawer.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) {
      console.error("Failed to inspect entry basins:", err);
    }
  }

  function renderSvgCdfChart(modelCdf) {
    const container = document.getElementById("insp-svg-cdf-container");
    if (!container) return;

    const W = 520;
    const H = 255;
    const padL = 48;
    const padR = 18;
    const padT = 14;
    const padB = 34;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const xMin = 0.0;
    const xMax = 1.0;

    function xPos(nse) {
      const clamped = Math.max(xMin, Math.min(xMax, Number(nse)));
      return padL + ((clamped - xMin) / (xMax - xMin)) * plotW;
    }

    function yPos(q) {
      return padT + (1.0 - q) * plotH;
    }

    function buildPath(cdfArr) {
      if (!cdfArr || cdfArr.length === 0) return "";
      const pts = [];
      const n = cdfArr.length;
      for (let i = 0; i < n; i++) {
        const q = i / (n - 1);
        const x = xPos(cdfArr[i]);
        const y = yPos(q);
        pts.push(`${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`);
      }
      return pts.join(" ");
    }

    let gridHtml = "";
    for (const tick of [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) {
      const x = xPos(tick);
      const y = yPos(tick);
      gridHtml += `
        <line x1="${x}" y1="${padT}" x2="${x}" y2="${padT + plotH}" stroke="hsl(210, 18%, 90%)" stroke-width="1" />
        <text x="${x}" y="${H - 10}" text-anchor="middle" font-size="10.5" font-family="JetBrains Mono" fill="hsl(212, 16%, 42%)">${tick.toFixed(1)}</text>
        <line x1="${padL}" y1="${y}" x2="${padL + plotW}" y2="${y}" stroke="hsl(210, 18%, 90%)" stroke-width="1" />
        <text x="${padL - 8}" y="${y + 3.5}" text-anchor="end" font-size="10.5" font-family="JetBrains Mono" fill="hsl(212, 16%, 42%)">${tick.toFixed(1)}</text>
      `;
    }

    const yMed = yPos(0.5);
    gridHtml += `<line x1="${padL}" y1="${yMed}" x2="${padL + plotW}" y2="${yMed}" stroke="hsl(210, 20%, 72%)" stroke-dasharray="3,3" stroke-width="1" />`;

    const pathModel = buildPath(modelCdf);

    container.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" width="100%" height="100%" role="img" aria-label="531-Basin NSE Cumulative Distribution Function">
        ${gridHtml}
        ${pathModel ? `<path d="${pathModel}" fill="none" stroke="hsl(204, 80%, 36%)" stroke-width="2.8" />` : ""}
        <rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="none" stroke="hsl(210, 18%, 76%)" stroke-width="1" />
        <text x="${padL + plotW / 2}" y="${H - 1}" text-anchor="middle" font-size="11" font-weight="600" fill="hsl(212, 22%, 28%)">Nash–Sutcliffe Efficiency (NSE, clipped at [0, 1])</text>
      </svg>
    `;
  }

  function renderHucBreakdownTable(hucRows) {
    const wrap = document.getElementById("insp-huc-table-wrap");
    if (!wrap) return;

    if (!hucRows || hucRows.length === 0) {
      wrap.innerHTML = `
        <div style="padding: 1.25rem; font-size: 0.83rem; color: var(--text-secondary);">
          Upload a daily hydrograph archive on the <strong>Contribute a Model</strong> tab to compute 18-region USGS HUC-02 diagnostics.
        </div>
      `;
      return;
    }

    let html = `
      <table class="lb-table" style="font-size: 0.79rem;">
        <thead>
          <tr>
            <th>USGS HUC-02</th>
            <th class="num-col">Basins</th>
            <th class="num-col">Median NSE</th>
            <th class="num-col">Mean NSE</th>
            <th class="num-col">Median KGE</th>
            <th class="num-col">NSE &le; 0</th>
          </tr>
        </thead>
        <tbody>
    `;

    for (const r of hucRows) {
      html += `
        <tr>
          <td style="font-family: var(--font-mono); font-weight: 700;">Region ${esc(r.huc_02)}</td>
          <td class="num-col">${esc(r.n_basins)}</td>
          <td class="num-col" style="font-weight: 700;">${fmtNum(r.median_NSE, 3)}</td>
          <td class="num-col">${fmtNum(r.mean_NSE, 3)}</td>
          <td class="num-col">${fmtNum(r.median_KGE, 3)}</td>
          <td class="num-col">${esc(r.n_NSE_le_0)}</td>
        </tr>
      `;
    }

    html += `</tbody></table>`;
    wrap.innerHTML = html;
  }

  // =========================================================================
  // 5. Render Participating Models Table (#view-models)
  // =========================================================================
  function classifyModelType(paradigmStr, famObj) {
    const p = String(paradigmStr || "").toLowerCase();
    const cat = String(
      (famObj && (famObj.category || famObj.model_category)) || ""
    ).toLowerCase();
    const fid = String((famObj && famObj.family_id) || "").toLowerCase();
    if (
      cat.includes("differentiable") ||
      cat.includes("hybrid") ||
      p.includes("differentiable") ||
      p.includes("physics") ||
      p.includes("emulator") ||
      p.includes("multi-model hybrid") ||
      p.includes("multi-structure hybrid") ||
      p === "hybrid"
    ) {
      return { key: "hybrid", label: "Hybrid (ML + Physics)" };
    }
    if (
      cat.includes("deep learning") ||
      cat.includes("ml") ||
      cat.includes("machine learning") ||
      p.includes("lstm") ||
      p.includes("transformer") ||
      p.includes("neural") ||
      p.includes("xgboost") ||
      p.includes("gradient-boosted") ||
      p.includes("diffusion") ||
      p.includes("state-space") ||
      p.includes("state space") ||
      p.includes("mlp") ||
      p.includes("deep learning") ||
      p.includes("machine learning")
    ) {
      return { key: "ml", label: "Machine learning" };
    }
    if (
      p.includes("conceptual") ||
      ["sacsma", "sac-sma", "hbv", "fuse", "marrmot", "gr4j"].some((t) =>
        fid.includes(t)
      )
    ) {
      return { key: "conceptual", label: "Conceptual" };
    }
    if (
      cat.includes("process") ||
      p.includes("process") ||
      p.includes("land-surface") ||
      p.includes("land surface") ||
      p.includes("distributed") ||
      p.includes("mesoscale hydrologic") ||
      p.includes("wrf-hydro") ||
      p.includes("national water model") ||
      ["vic", "mhm", "nwm", "summa"].some((t) => fid.includes(t))
    ) {
      return { key: "process", label: "Process-based" };
    }
    if (p.includes("statistical") || p.includes("donor")) {
      return { key: "statistical", label: "Statistical" };
    }
    return { key: "ml", label: "Machine learning" };
  }

  function extractModelYear(fam) {
    if (fam && typeof fam.year === "number" && !Number.isNaN(fam.year)) {
      return fam.year;
    }
    const s = String((fam && fam.citation) || "");
    const matches = s.match(/\b(19\d\d|20\d\d)\b/g);
    if (matches && matches.length > 0) {
      return Math.min(...matches.map(Number));
    }
    return 2026;
  }

  function renderFamiliesGrid() {
    const grid = document.getElementById("families-grid-container");
    if (!grid) return;

    const q = STATE.familySearchQuery.toLowerCase();
    const typeFilter = STATE.familyTypeFilter;

    const filtered = STATE.families
      .filter((f) => {
        const typeInfo = classifyModelType(f.paradigm, f);
        if (typeFilter && typeInfo.key !== typeFilter) {
          return false;
        }
        if (!q) return true;
        const hay = [
          f.name,
          f.family_id,
          f.paradigm,
          f.provenance_category,
          formatFamilySourceCategories(f.provenance_category).join(" "),
          f.citation,
          f.github_url,
          f.zenodo_doi,
          f.description,
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();
        return hay.includes(q);
      })
      .sort((a, b) => extractModelYear(a) - extractModelYear(b));

    let html = `
      <div class="leaderboard-section-card" style="padding: 0;">
        <div class="table-scroll-wrap">
          <table class="lb-table models-registry-table">
            <thead>
              <tr>
                <th style="min-width: 210px;">Model</th>
                <th style="min-width: 170px;">Reference</th>
                <th>Source</th>
                <th>Archive &amp; Code</th>
                <th style="min-width: 180px;">Leaderboards</th>
              </tr>
            </thead>
            <tbody>
    `;

    if (filtered.length === 0) {
      html += `
        <tr>
          <td colspan="5" class="empty-table-cell">
            <div class="empty-table-inner">
              <strong>No matching models</strong>
            </div>
          </td>
        </tr>
      `;
    } else {
      for (const fam of filtered) {
        const typeInfo = classifyModelType(fam.paradigm, fam);
        const famAsRow = {
          data_archive_url:
            fam.zenodo_url ||
            fam.hydroshare_url ||
            (fam.zenodo_doi ? `https://doi.org/${fam.zenodo_doi}` : ""),
          zenodo_doi: fam.zenodo_doi,
          zenodo_url: fam.zenodo_url,
          hydroshare_url: fam.hydroshare_url,
          code_url: fam.github_url,
          github_url: fam.github_url,
        };

        const coveredSet = new Set(Array.isArray(fam.leaderboards_covered) ? fam.leaderboards_covered : []);
        for (const lb of STATE.allLeaderboards) {
          for (const r of lb.rows || []) {
            if (r.family_id === fam.family_id || r.model_name === fam.name) {
              coveredSet.add(lb.id);
            }
          }
        }
        const coveredBoards = STATE.allLeaderboards.filter((lb) => coveredSet.has(lb.id));

        const boardsHtml =
          coveredBoards.length > 0
            ? `<div class="participating-boards-list">` +
              coveredBoards
                .map(
                  (lb) => `
                  <button
                    type="button"
                    class="board-jump-pill"
                    data-jump-lb="${esc(lb.id)}"
                    title="${esc(lb.title)}"
                  >
                    ${esc(formatLeaderboardShortLabel(lb))}
                  </button>
                `
                )
                .join("") +
              `</div>`
            : `<span class="muted-dash">&mdash;</span>`;

        const namedCats = formatFamilySourceCategories(fam.provenance_category);
        const catsHtml = `<div class="cat-badges-wrap">${namedCats
          .map((c) => `<span class="cat-badge">${esc(c)}</span>`)
          .join("")}</div>`;

        const safePaperUrl = sanitizeExternalUrl(fam.paper_url);
        html += `
          <tr class="model-row-colored row-type-${esc(typeInfo.key)}" id="family-row-${esc(fam.family_id)}" title="${esc(typeInfo.label)}${fam.description ? " — " + esc(fam.description) : ""}">
            <td style="font-weight: 600; color: var(--text-primary);">
              ${esc(fam.name)}
            </td>
            <td style="font-size: 0.83rem;">
              ${
                safePaperUrl
                  ? `<a href="${esc(safePaperUrl)}" target="_blank" rel="noopener noreferrer" style="font-weight: 500; color: var(--accent-primary); text-decoration: none;">${esc(fam.citation || "Reference")} &#8599;</a>`
                  : `<span style="color: var(--text-secondary);">${esc(fam.citation || "—")}</span>`
              }
            </td>
            <td>
              ${catsHtml}
            </td>
            <td>
              <div style="display: flex; align-items: center; gap: 0.35rem;">
                ${renderDataArchiveButton(famAsRow)}
                ${renderCodebaseButton(famAsRow)}
              </div>
            </td>
            <td>
              ${boardsHtml}
            </td>
          </tr>
        `;
      }
    }

    const famFootnotes = filtered.filter((f) => f.asterisk || f.asterisk_reason);
    const famFootnotesHtml =
      famFootnotes.length > 0
        ? `
        <div class="lb-footnotes-box" style="margin: 0.75rem 1rem 1rem; padding: 0.75rem 1rem; border-radius: 8px; background: var(--surface-subtle, rgba(15, 23, 42, 0.03)); border: 1px solid var(--border-subtle, rgba(15, 23, 42, 0.09)); font-size: 0.8rem; line-height: 1.45; color: var(--text-secondary);">
          <div style="font-weight: 700; color: var(--text-primary); margin-bottom: 0.35rem;">Model-Specific Footnotes (<code>*</code>)</div>
          <ul style="margin: 0; padding-left: 1.15rem; display: flex; flex-direction: column; gap: 0.22rem;">
            ${famFootnotes
              .map(
                (f) =>
                  `<li><strong>${esc(f.name)}</strong>${f.citation ? ` (${esc(f.citation)})` : ""}: ${esc(f.asterisk_reason || "Non-standard evaluation subset or period.")}</li>`
              )
              .join("")}
          </ul>
        </div>`
        : "";

    html += `
            </tbody>
          </table>
        </div>
        ${famFootnotesHtml}
      </div>
    `;

    grid.innerHTML = html;

    grid.querySelectorAll("[data-jump-lb]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetBoard = btn.getAttribute("data-jump-lb");
        switchMainTab("leaderboards");
        selectLeaderboardTab(targetBoard);
      });
    });
  }

  // =========================================================================
  // 6. Option C Community Contributions (Preview + Official GitHub PR)
  // =========================================================================
  function inferForcingCategoryFromInputs(forcingStr) {
    const s = String(forcingStr || "").toLowerCase();
    if (!s || s.includes("freeform") || s.includes("none of the above")) {
      return "freeform";
    }
    if (
      s.includes("lagged") ||
      s.includes("streamflow") ||
      s.includes("discharge") ||
      s.includes("autoregress")
    ) {
      return "lagged_streamflow";
    }
    let count = 0;
    for (const prod of ["daymet", "maurer", "nldas"]) {
      if (s.includes(prod)) count += 1;
    }
    if (count === 1) return "single_camels";
    if (count >= 2) return "multi_camels";
    return "freeform";
  }

  const MIP_BENCHMARK_CONFIGS = {
    camels_us: {
      title: "CAMELS-US",
      defaultForcing: "Daymet",
      spatialOptions: [
        { value: "gauged", label: "Gauged Simulation" },
        { value: "pub", label: "12-Fold PUB (Ungauged Basins)" },
        { value: "pur", label: "18-Fold HUC-02 PUR (Ungauged Regions)" },
      ],
      forcingGroups: [
        {
          label: "Single CAMELS Forcing",
          options: ["Daymet", "Maurer", "NLDAS"],
        },
        {
          label: "Multiple CAMELS Forcings",
          options: [
            "Daymet + Maurer",
            "Daymet + NLDAS",
            "Maurer + NLDAS",
            "Daymet + Maurer + NLDAS",
          ],
        },
        {
          label: "With Lagged Streamflow (Gauged Only)",
          options: [
            "Daymet + Lagged Streamflow",
            "Maurer + Lagged Streamflow",
            "NLDAS + Lagged Streamflow",
            "Daymet + Maurer + Lagged Streamflow",
            "Daymet + NLDAS + Lagged Streamflow",
            "Maurer + NLDAS + Lagged Streamflow",
            "Daymet + Maurer + NLDAS + Lagged Streamflow",
          ],
        },
        {
          label: "Freeform (Non-CAMELS / External)",
          options: ["Freeform (None of the above)"],
        },
      ],
      rulesHtml: `
        <li>
          <strong>Blind Evaluation Window</strong>
          <span>Select architectures, hyperparameters, and early stopping strictly on <code>1999-10-01</code> to <code>2008-09-30</code>; evaluate blindly on <code>1989-10-01</code> to <code>1999-09-30</code>.</span>
        </li>
        <li>
          <strong>Canonical Spatial Folds</strong>
          <span>Use the canonical <a href="/api/splits/pub_12fold_random_splits.json" target="_blank" rel="noopener noreferrer">12-fold PUB</a> and <a href="/api/splits/pur_18fold_huc02_splits.json" target="_blank" rel="noopener noreferrer">18-fold HUC-02 PUR</a> basin splits (<a href="/api/splits/basins_531.txt" target="_blank" rel="noopener noreferrer">531 basins</a>) with zero target-basin leakage.</span>
        </li>
        <li>
          <strong>Forcing Inputs &amp; 14 Canonical Metrics</strong>
          <span>Declare all meteorological and lagged streamflow inputs used. Models are scored across the 14 canonical Gauch et al. (2023) deterministic metrics including <code>logNSE</code> and <code>logKGE</code> (<code>mm/day</code> specific discharge).</span>
        </li>
        <li>
          <strong>Instant Preview + Open GitHub PR</strong>
          <span>Step 1&ndash;3 previews remain strictly local in your browser (<code>localStorage</code>). Permanent leaderboard publication requires a DOI-minted prediction archive, codebase URL (or Closed-Source declaration), and automated CI verification via GitHub.</span>
        </li>
      `,
      chk1Html: `Blind <code>1989-10-01</code> to <code>1999-09-30</code> evaluation holdout (trained on <code>1999-10-01</code> to <code>2008-09-30</code>)`,
      chk2Html: `Canonical <a href="/api/splits/pub_12fold_random_splits.json" target="_blank" rel="noopener noreferrer">12-fold PUB</a> and <a href="/api/splits/pur_18fold_huc02_splits.json" target="_blank" rel="noopener noreferrer">18-fold HUC-02 PUR</a> splits (<a href="/api/splits/basins_531.txt" target="_blank" rel="noopener noreferrer">531 basins</a>) with zero target leakage`,
      chk3Html: `Selected forcing inputs include all meteorological and lagged streamflow inputs used`,
      chk4Html: `Deterministic daily predictions in mm/day across the 531 CAMELS-US basins`,
    },
  };

  function populateMipBenchmarkFields(mipId) {
    const cfg = MIP_BENCHMARK_CONFIGS[mipId] || MIP_BENCHMARK_CONFIGS.camels_us;
    const spEl = document.getElementById("sub-spatial-regime");
    const foEl = document.getElementById("sub-forcing");
    const rulesHeading = document.getElementById("contrib-rules-heading");
    const rulesList = document.getElementById("contrib-rules-list");

    if (spEl) {
      const prevSp = spEl.value || "gauged";
      spEl.innerHTML = cfg.spatialOptions
        .map(
          (o) =>
            `<option value="${esc(o.value)}"${o.value === prevSp ? " selected" : ""}>${esc(o.label)}</option>`
        )
        .join("");
    }

    if (foEl) {
      foEl.innerHTML = cfg.forcingGroups
        .map(
          (g) =>
            `<optgroup label="${esc(g.label)}">` +
            g.options
              .map(
                (opt) =>
                  `<option value="${esc(opt)}"${opt === cfg.defaultForcing ? " selected" : ""}>${esc(opt)}</option>`
              )
              .join("") +
            `</optgroup>`
        )
        .join("");
    }

    if (rulesHeading) {
      rulesHeading.innerHTML = `Submission Rules &mdash; ${esc(cfg.title)}`;
    }
    if (rulesList) {
      rulesList.innerHTML = cfg.rulesHtml;
    }

    const chk1 = document.getElementById("chk-rule-1-text");
    const chk2 = document.getElementById("chk-rule-2-text");
    const chk3 = document.getElementById("chk-rule-3-text");
    const chk4 = document.getElementById("chk-rule-4-text");
    if (chk1) chk1.innerHTML = cfg.chk1Html;
    if (chk2) chk2.innerHTML = cfg.chk2Html;
    if (chk3) chk3.innerHTML = cfg.chk3Html;
    if (chk4) chk4.innerHTML = cfg.chk4Html;

    syncContributeTargetLeaderboard();
  }

  function syncContributeTargetLeaderboard() {
    const mipEl = document.getElementById("sub-mip-benchmark");
    const mipId = (mipEl && mipEl.value) || "camels_us";
    const cfg = MIP_BENCHMARK_CONFIGS[mipId] || MIP_BENCHMARK_CONFIGS.camels_us;
    const spEl = document.getElementById("sub-spatial-regime");
    const foEl = document.getElementById("sub-forcing");
    const lbInput = document.getElementById("sub-leaderboard-id");
    const sp = spEl ? spEl.value : "gauged";

    if (foEl) {
      const isUngauged = sp === "pub" || sp === "pur";
      foEl.querySelectorAll("option").forEach((opt) => {
        if (opt.value.toLowerCase().includes("lagged streamflow")) {
          opt.disabled = isUngauged;
        }
      });
      if (isUngauged && foEl.value.toLowerCase().includes("lagged streamflow")) {
        foEl.value = cfg.defaultForcing;
      }
    }

    const foVal = foEl ? foEl.value : cfg.defaultForcing;
    const foCat = inferForcingCategoryFromInputs(foVal);
    const resolvedLbId = `${sp}__${foCat}`;
    if (lbInput) {
      lbInput.value = resolvedLbId;
    }
    return resolvedLbId;
  }

  function syncNewFamilyNameVisibility() {
    const famSelect = document.getElementById("sub-family-id");
    const wrap = document.getElementById("sub-new-family-wrap");
    if (!famSelect || !wrap) return;
    wrap.style.display = famSelect.value ? "none" : "";
  }

  function populateContributeSelects() {
    syncContributeTargetLeaderboard();

    const famSelect = document.getElementById("sub-family-id");
    if (famSelect) {
      const prevVal = famSelect.value || "";
      famSelect.innerHTML =
        `<option value="">+ Register as New Model Family</option>` +
        STATE.families
          .map((f) => `<option value="${esc(f.family_id)}">${esc(f.name)} (${esc(f.citation)})</option>`)
          .join("");
      if (prevVal && STATE.families.some((f) => f.family_id === prevVal)) {
        famSelect.value = prevVal;
      }
      if (!famSelect.dataset.boundFamilyToggle) {
        famSelect.addEventListener("change", syncNewFamilyNameVisibility);
        famSelect.dataset.boundFamilyToggle = "1";
      }
      syncNewFamilyNameVisibility();
    }
  }

  function normalizeClientUrl(raw, kind) {
    const s = String(raw || "").trim();
    if (!s) return "";
    if (/^https?:\/\//i.test(s)) return sanitizeExternalUrl(s);
    if (kind === "zenodo") {
      if (s.startsWith("10.")) return sanitizeExternalUrl(`https://doi.org/${s}`);
      if (/^doi:/i.test(s)) return sanitizeExternalUrl(`https://doi.org/${s.replace(/^doi:\s*/i, "")}`);
      if (/^\d{5,}$/.test(s)) return sanitizeExternalUrl(`https://doi.org/10.5281/zenodo.${s}`);
    }
    if (kind === "github") {
      if (s.startsWith("github.com/")) return sanitizeExternalUrl(`https://${s}`);
      if (/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+/.test(s)) {
        return sanitizeExternalUrl(`https://github.com/${s.replace(/^@/, "")}`);
      }
    }
    return sanitizeExternalUrl(s);
  }

  function isClosedSourceSelected() {
    const s1 = document.getElementById("sub-closed-source");
    const s4 = document.getElementById("step4-closed-source");
    return Boolean((s1 && s1.checked) || (s4 && s4.checked));
  }

  function setClosedSourceSelection(checked) {
    const val = Boolean(checked);
    ["sub-closed-source", "step4-closed-source"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.checked = val;
    });
    ["sub-github-url", "step4-code-url"].forEach((id) => {
      const inp = document.getElementById(id);
      if (!inp) return;
      inp.disabled = val;
      if (val) {
        inp.placeholder = "Closed-Source / Proprietary (no public code URL)";
      } else {
        inp.placeholder = "https://github.com/owner/repo";
      }
    });
  }

  function updateLiveRepoBadgePreview() {
    const gh = ((document.getElementById("sub-github-url") || {}).value || "").trim();
    const zn = ((document.getElementById("sub-zenodo-doi") || {}).value || "").trim();
    const closedSource = isClosedSourceSelected();
    const previewEl = document.getElementById("live-repo-badge-preview");
    if (!previewEl) return;

    const safeCode = closedSource ? "" : normalizeClientUrl(gh, "github");
    const safeArch = normalizeClientUrl(zn, "zenodo");
    if (!safeCode && !safeArch && !closedSource) {
      previewEl.style.display = "none";
      previewEl.innerHTML = "";
      return;
    }

    const rowPreview = {
      data_archive_url: safeArch,
      zenodo_doi: zn.startsWith("10.") ? zn : "",
      code_url: safeCode,
      closed_source: closedSource,
      is_closed_source: closedSource,
    };
    previewEl.style.display = "flex";
    previewEl.innerHTML =
      `<span style="font-weight: 600; color: var(--text-secondary);">Validated Link Badges:</span>` +
      renderDataArchiveButton(rowPreview) +
      renderCodebaseButton(rowPreview);
  }

  function buildHydromipPackageCliSnippet() {
    const forcing = ((document.getElementById("sub-forcing") || {}).value || "Daymet").trim();
    const regime = ((document.getElementById("sub-spatial-regime") || {}).value || "gauged").trim();
    return (
      `pip install git+https://github.com/grey-nearing/camels-us-mip.git\n` +
      `hydromip package --input ./your_predictions.nc --output ./camels_submission.zarr.zip \\\n` +
      `  --forcing "${forcing}" --regime ${regime}`
    );
  }

  function renderDiagnosticCard(containerId, title, detailLines, isError = false, showCliSnippet = true) {
    const el = document.getElementById(containerId);
    if (!el) return;
    if (!title && (!detailLines || detailLines.length === 0)) {
      el.hidden = true;
      el.innerHTML = "";
      return;
    }
    const snippet = showCliSnippet ? buildHydromipPackageCliSnippet() : "";
    const itemsHtml = (detailLines || [])
      .map((line) => `<li>${esc(line)}</li>`)
      .join("");
    el.hidden = false;
    el.className = `diagnostic-card ${isError ? "is-error" : !showCliSnippet ? "is-info" : ""}`;
    el.innerHTML = `
      <div style="display:flex; align-items:center; justify-content:space-between; gap:0.5rem; flex-wrap:wrap;">
        <strong>${esc(title)}</strong>
        ${showCliSnippet ? '<button type="button" class="btn-row-link" data-copy-cli-snippet="1">Copy CLI Snippet</button>' : ""}
      </div>
      ${itemsHtml ? `<ul style="margin:0.4rem 0 0; padding-left:1.2rem; line-height:1.45;">${itemsHtml}</ul>` : ""}
      ${
        showCliSnippet
          ? `<div style="font-size:0.76rem; color:var(--color-ink-2); margin-top:0.5rem;">
               Standardize predictions locally before re-uploading or attaching to a GitHub PR:
             </div>
             <pre class="cli-snippet-box">${esc(snippet)}</pre>`
          : ""
      }
    `;
    const copyBtn = el.querySelector("[data-copy-cli-snippet]");
    if (copyBtn) {
      copyBtn.addEventListener("click", async () => {
        try {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(snippet);
          }
          copyBtn.textContent = "Copied!";
          setTimeout(() => {
            copyBtn.textContent = "Copy CLI Snippet";
          }, 1800);
        } catch (_e) {
          // ignore clipboard denial
        }
      });
    }
  }

  function setContribStep(stepNum) {
    const s = Math.max(1, Math.min(4, Number(stepNum) || 1));
    STATE.contribStep = s;
    for (let i = 1; i <= 4; i++) {
      const sec = document.getElementById(`contrib-step-${i}`);
      const pill = document.getElementById(`contrib-pill-${i}`);
      if (sec) sec.hidden = i !== s;
      if (pill) {
        pill.classList.toggle("is-active", i === s);
        pill.classList.toggle("is-done", i < s);
      }
    }
    if (s === 4) {
      syncStep4UrlInputsAndManifest();
    }
  }

  function populateOverrideDropdown(selectId, options, selectedVal, allowNone = false) {
    const sel = document.getElementById(selectId);
    if (!sel) return;
    const opts = [];
    if (allowNone) opts.push("");
    for (const o of Array.isArray(options) ? options : []) {
      const str = String(o || "").trim();
      if (str && !opts.includes(str)) opts.push(str);
    }
    const chosen = selectedVal ? String(selectedVal).trim() : "";
    if (chosen && !opts.includes(chosen)) opts.push(chosen);
    sel.innerHTML = opts
      .map((val) => {
        const label = val || "(none — single deterministic run)";
        return `<option value="${esc(val)}"${val === chosen ? " selected" : ""}>${esc(label)}</option>`;
      })
      .join("");
  }

  function renderSanityHydrographSvgGrid(sampleBasins) {
    const container = document.getElementById("step2-sanity-plots");
    if (!container) return;
    const basins = Array.isArray(sampleBasins) ? sampleBasins.slice(0, 3) : [];
    if (basins.length === 0) {
      container.innerHTML = "";
      return;
    }

    const cardsHtml = basins
      .map((b) => {
        const qobs = Array.isArray(b.qobs) ? b.qobs : [];
        const qsim = Array.isArray(b.qsim) ? b.qsim : [];
        const dates = Array.isArray(b.dates) ? b.dates : [];
        const n = Math.min(qobs.length, qsim.length);
        if (n < 2) {
          return `
            <div class="sanity-basin-card">
              <div class="sanity-basin-card-head">
                <strong>USGS ${esc(b.basin_id || "?")}</strong>
                <span>Insufficient series</span>
              </div>
            </div>`;
        }
        // Downsample to <= 180 points for crisp lightweight inline SVG
        const step = Math.max(1, Math.floor(n / 180));
        const obsPts = [];
        const simPts = [];
        let maxV = 1e-3;
        for (let i = 0; i < n; i += step) {
          const vo = Number(qobs[i]);
          const vs = Number(qsim[i]);
          if (Number.isFinite(vo) && vo > maxV) maxV = vo;
          if (Number.isFinite(vs) && vs > maxV) maxV = vs;
          obsPts.push([i, vo]);
          simPts.push([i, vs]);
        }
        const W = 250;
        const H = 92;
        const padL = 6;
        const padR = 6;
        const padT = 8;
        const padB = 10;
        const xMap = (idx) => padL + (idx / Math.max(1, n - 1)) * (W - padL - padR);
        const yMap = (v) =>
          padT + (1 - Math.max(0, Math.min(1, v / maxV))) * (H - padT - padB);
        const toPolyline = (pairs) =>
          pairs
            .filter(([, v]) => Number.isFinite(v))
            .map(([idx, v]) => `${xMap(idx).toFixed(1)},${yMap(v).toFixed(1)}`)
            .join(" ");

        const nseStr =
          b.nse !== null && b.nse !== undefined && Number.isFinite(Number(b.nse))
            ? `NSE ${Number(b.nse).toFixed(2)}`
            : "";
        const dateSpan =
          dates.length >= 2 ? `${dates[0]} to ${dates[dates.length - 1]}` : `${n} steps`;

        return `
          <div class="sanity-basin-card">
            <div class="sanity-basin-card-head">
              <strong>USGS ${esc(b.basin_id)}</strong>
              <span style="color: var(--color-accent); font-weight: 600;">${esc(nseStr)}</span>
            </div>
            <div style="font-size:0.68rem; color:var(--color-muted); margin-bottom:0.25rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="${esc(b.gauge_name || "")}">
              ${esc(b.gauge_name || dateSpan)}
            </div>
            <svg viewBox="0 0 ${W} ${H}" width="100%" height="92" role="img" aria-label="Sanity plot for basin ${esc(b.basin_id)}" style="display:block; background:#ffffff; border:1px solid var(--color-rule); border-radius:4px;">
              <polyline fill="none" stroke="#16191d" stroke-width="1.3" points="${toPolyline(obsPts)}" />
              <polyline fill="none" stroke="#0284c7" stroke-width="1.2" opacity="0.9" points="${toPolyline(simPts)}" />
            </svg>
            <div style="display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:0.64rem; color:var(--color-muted); margin-top:0.2rem;">
              <span><span style="color:#16191d;">&#9644;</span> Obs &nbsp;<span style="color:#0284c7;">&#9644;</span> Pred</span>
              <span>max ${maxV.toFixed(1)} mm/d</span>
            </div>
          </div>`;
      })
      .join("");

    container.innerHTML = `
      <div style="font-size:0.78rem; font-weight:600; color:var(--color-ink); margin-top:0.75rem;">
        3-Basin Observed vs. Predicted Sanity Hydrographs
      </div>
      <div class="sanity-plot-grid">${cardsHtml}</div>
    `;
  }

  function renderInspectResponseStep2(inspectData) {
    STATE.previewInspectData = inspectData;
    const schema = inspectData.schema || {};
    const sanity = inspectData.sanity || {};
    const fmt = inspectData.format || "unknown";
    const recipeSrc = inspectData.recipe_source || "deterministic_tier1";
    const tierBadge = document.getElementById("step2-tier-badge");
    const summaryEl = document.getElementById("step2-schema-summary");
    const badgesEl = document.getElementById("step2-sanity-badges");

    const tierLabel =
      recipeSrc === "llm_tier2"
        ? "Tier 2 Schema-Recipe"
        : recipeSrc === "user_override"
        ? "Manual Override"
        : "Tier 1 Deterministic";

    if (tierBadge) tierBadge.textContent = `${tierLabel} · ${fmt}`;
    if (summaryEl) {
      summaryEl.innerHTML =
        `Detected <code>${esc(fmt)}</code> schema via <strong>${esc(tierLabel)}</strong>. ` +
        `Verify variable/dimension mappings below before scoring.`;
    }

    const availVars = schema.available_vars || (schema.var ? [schema.var] : ["qsim"]);
    const availDims =
      schema.available_dims ||
      [schema.basin_dim, schema.time_dim, schema.member_dim].filter(Boolean);
    populateOverrideDropdown("override-var", availVars, schema.var || availVars[0] || "qsim", false);
    populateOverrideDropdown("override-basin-dim", availDims, schema.basin_dim || "basin", false);
    populateOverrideDropdown("override-time-dim", availDims, schema.time_dim || "date", false);
    populateOverrideDropdown("override-member-dim", availDims, schema.member_dim || "", true);

    const nMatched = Number(sanity.n_basins_matched ?? 0);
    const nExpected = Number(sanity.n_basins_expected ?? 531);
    const nanCount = Number(sanity.nan_count ?? 0);
    const dStart = sanity.date_start || "";
    const dEnd = sanity.date_end || "";
    const ratio = sanity.median_sim_obs_ratio;
    const unitWarn = sanity.unit_warning || "";

    const basinOk = nMatched === nExpected && nExpected === 531;
    const nanOk = nanCount === 0;
    const winOk = dStart === "1989-10-01" && dEnd === "1999-09-30";
    const ratioNum = ratio !== null && ratio !== undefined ? Number(ratio) : null;
    const unitBad =
      Boolean(unitWarn) ||
      (ratioNum !== null && Number.isFinite(ratioNum) && (ratioNum > 10 || ratioNum < 0.1));

    const badgeList = [
      {
        cls: basinOk ? "is-ok" : "is-fail",
        text: `${nMatched}/${nExpected} basins`,
      },
      {
        cls: nanOk ? "is-ok" : "is-warn",
        text: `${nanCount} NaNs`,
      },
      {
        cls: winOk ? "is-ok" : "is-warn",
        text: dStart && dEnd ? `${dStart} to ${dEnd}` : "Date window unverified",
      },
      {
        cls: unitBad ? "is-warn" : "is-ok",
        text: unitWarn
          ? `Unit Warning: ${unitWarn}`
          : ratioNum !== null && Number.isFinite(ratioNum)
          ? `Sim/Obs ratio ${ratioNum.toFixed(2)}x (mm/day OK)`
          : "Units checked",
      },
    ];
    if (badgesEl) {
      badgesEl.innerHTML = badgeList
        .map((b) => `<span class="sanity-badge ${b.cls}">${esc(b.text)}</span>`)
        .join("");
    }

    const diagnostics = Array.isArray(sanity.diagnostics) ? [...sanity.diagnostics] : [];
    if (unitBad && !unitWarn) {
      diagnostics.push(
        `Median simulated/observed ratio is ${ratioNum.toFixed(2)}x — predictions may be in cfs or m³/s instead of specific discharge (mm/day).`
      );
    }
    if (!basinOk || !winOk || sanity.checks_passed === false || diagnostics.length > 0) {
      const checksPassed = sanity.checks_passed !== false && basinOk && winOk && !unitBad;
      renderDiagnosticCard(
        "step2-diagnostic-card",
        checksPassed ? "Sanity Diagnostics" : "Physical Sanity Check Requires Attention",
        diagnostics,
        sanity.checks_passed === false,
        !checksPassed
      );
    } else {
      renderDiagnosticCard("step2-diagnostic-card", "", [], false, false);
    }

    renderSanityHydrographSvgGrid(inspectData.sample_basins || []);
    setContribStep(2);
  }

  function getConfirmedRecipeFromStep2() {
    return {
      var: ((document.getElementById("override-var") || {}).value || "").trim(),
      basin_dim: ((document.getElementById("override-basin-dim") || {}).value || "").trim(),
      time_dim: ((document.getElementById("override-time-dim") || {}).value || "").trim(),
      member_dim: ((document.getElementById("override-member-dim") || {}).value || "").trim() || null,
    };
  }

  async function inspectPreviewSubmission(isOverrideReinspect = false) {
    const statusEl = document.getElementById("inspect-status-msg");
    const step1Diag = document.getElementById("step1-diagnostic-card");
    if (step1Diag) step1Diag.hidden = true;

    const lbId = syncContributeTargetLeaderboard();
    const spatialRegime = ((document.getElementById("sub-spatial-regime") || {}).value || "gauged").trim();
    const forcing = ((document.getElementById("sub-forcing") || {}).value || "Daymet").trim();
    const modelName = ((document.getElementById("sub-model-name") || {}).value || "").trim();
    const rawArchive = ((document.getElementById("sub-zenodo-doi") || {}).value || "").trim();
    const archiveUrl = normalizeClientUrl(rawArchive, "zenodo");
    const file = STATE.uploadedFile;

    if (!file && !archiveUrl) {
      if (statusEl) {
        statusEl.textContent = "Please attach a hydrograph file (.nc, .zarr.zip, .csv, .parquet, .npz) OR paste a valid archive DOI/URL.";
        statusEl.style.color = "var(--status-danger)";
      }
      return;
    }
    if (!modelName) {
      if (statusEl) {
        statusEl.textContent = "Please enter a Model Name before inspecting.";
        statusEl.style.color = "var(--status-danger)";
      }
      return;
    }

    for (const chkId of ["chk-rule-1", "chk-rule-2", "chk-rule-3", "chk-rule-4"]) {
      const chk = document.getElementById(chkId);
      if (chk && !chk.checked) {
        if (statusEl) {
          statusEl.textContent = "Please confirm all benchmark compliance checkboxes before inspecting.";
          statusEl.style.color = "var(--status-danger)";
        }
        return;
      }
    }

    const btn = document.getElementById("btn-preview-inspect");
    if (btn) btn.disabled = true;
    if (statusEl) {
      statusEl.textContent = file
        ? `Inspecting '${file.name}' schema and 531-basin physical sanity...`
        : "Fetching remote archive metadata and inspecting hydrograph schema...";
      statusEl.style.color = "var(--color-ink-2)";
    }

    try {
      const recipeOverride = isOverrideReinspect ? getConfirmedRecipeFromStep2() : null;
      let res;
      if (file) {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("leaderboard_id", lbId);
        fd.append("spatial_regime", spatialRegime);
        fd.append("forcing", forcing);
        fd.append("model_name", modelName);
        if (archiveUrl) fd.append("archive_url", archiveUrl);
        if (recipeOverride) fd.append("recipe_override", JSON.stringify(recipeOverride));
        res = await fetch("/api/preview/inspect", { method: "POST", body: fd });
      } else {
        res = await fetch("/api/preview/inspect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            archive_url: archiveUrl,
            leaderboard_id: lbId,
            spatial_regime: spatialRegime,
            forcing: forcing,
            model_name: modelName,
            ...(recipeOverride ? { recipe_override: recipeOverride } : {}),
          }),
        });
      }

      const payload = await parseApiJson(res);
      if (!res.ok) {
        const msg = payload.error || payload.detail || `HTTP ${res.status}`;
        if (payload.code === "SCORER_UNCONFIGURED" || res.status === 503) {
          if (statusEl) {
            statusEl.textContent = msg;
            statusEl.style.color = "var(--status-warning)";
          }
          renderDiagnosticCard(
            "step1-diagnostic-card",
            "Live Scoring Backend Unavailable on This Deployment — Use GitHub PR Submission",
            [
              msg,
              "You can package & validate locally with `hydromip package` or jump straight to Step 4 to submit your official GitHub PR / Issue with your DOI.",
            ],
            false
          );
          return;
        }
        throw new Error(msg);
      }

      if (statusEl) {
        statusEl.textContent = "Inspection complete — review Step 2 mappings and sanity plots below.";
        statusEl.style.color = "var(--status-success)";
      }
      renderInspectResponseStep2(payload);
    } catch (err) {
      const msg = err && err.message ? err.message : String(err);
      if (statusEl) {
        statusEl.textContent = `Inspection failed: ${msg}`;
        statusEl.style.color = "var(--status-danger)";
      }
      renderDiagnosticCard(
        "step1-diagnostic-card",
        "Unable to Parse or Validate File",
        [msg],
        true
      );
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  async function scorePreviewSubmission() {
    const scoreStatusEl = document.getElementById("score-status-msg");
    if (!STATE.previewInspectData || !STATE.previewInspectData.preview_token) {
      if (scoreStatusEl) {
        scoreStatusEl.textContent = "Please complete Step 1 inspection first.";
        scoreStatusEl.style.color = "var(--status-danger)";
      }
      return;
    }

    const lbId = syncContributeTargetLeaderboard();
    const spatialRegime = ((document.getElementById("sub-spatial-regime") || {}).value || "gauged").trim();
    const forcing = ((document.getElementById("sub-forcing") || {}).value || "Daymet").trim();
    const modelName = ((document.getElementById("sub-model-name") || {}).value || "").trim();
    const familyId = ((document.getElementById("sub-family-id") || {}).value || "").trim();
    const newFamilyName = ((document.getElementById("sub-new-family-name") || {}).value || "").trim();
    const paradigm = ((document.getElementById("sub-paradigm") || {}).value || "Machine learning").trim();
    const category = ((document.getElementById("sub-category") || {}).value || "Contribution").trim();
    const ensembleInfo = ((document.getElementById("sub-ensemble-info") || {}).value || "").trim();
    const closedSource = isClosedSourceSelected();
    const codeUrl = closedSource
      ? ""
      : normalizeClientUrl(((document.getElementById("sub-github-url") || {}).value || "").trim(), "github");
    const archiveUrl = normalizeClientUrl(((document.getElementById("sub-zenodo-doi") || {}).value || "").trim(), "zenodo");
    const citation = ((document.getElementById("sub-citation") || {}).value || "").trim();

    const recipe = getConfirmedRecipeFromStep2();
    const btn = document.getElementById("btn-preview-score");
    if (btn) btn.disabled = true;
    if (scoreStatusEl) {
      scoreStatusEl.textContent = "Computing 14 canonical Gauch et al. (2023) metrics (including logNSE & logKGE) across 531 basins...";
      scoreStatusEl.style.color = "var(--color-ink-2)";
    }

    try {
      const res = await fetch("/api/preview/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          preview_token: STATE.previewInspectData.preview_token,
          recipe,
          metadata: {
            leaderboard_id: lbId,
            spatial_regime: spatialRegime,
            forcing,
            model_name: modelName,
            family_id: familyId || null,
            new_family_name: familyId ? null : newFamilyName || modelName,
            paradigm,
            category,
            ensemble_info: ensembleInfo,
            code_url: codeUrl,
            archive_url: archiveUrl,
            citation: citation || "Community Contribution (2026)",
          },
        }),
      });
      const data = await parseApiJson(res);
      if (!res.ok) {
        throw new Error(data.error || data.detail || `HTTP ${res.status}`);
      }

      const rawEntry = data.entry || data;
      const entryId =
        rawEntry.entry_id ||
        `local_preview_${Date.now().toString(36)}`;
      const entry = ensureFlatGauchMetrics({
        ...rawEntry,
        entry_id: entryId,
        leaderboard_id: rawEntry.leaderboard_id || lbId,
        model_name: rawEntry.model_name || modelName,
        family_id: rawEntry.family_id || familyId || `preview_${entryId}`,
        family_name: rawEntry.family_name || newFamilyName || modelName,
        paradigm: rawEntry.paradigm || paradigm,
        category: rawEntry.category || category,
        forcing: rawEntry.forcing || forcing,
        ensemble_info: rawEntry.ensemble_info || ensembleInfo,
        code_url: sanitizeExternalUrl(rawEntry.code_url || codeUrl),
        github_url: sanitizeExternalUrl(rawEntry.github_url || codeUrl),
        closed_source: closedSource || !codeUrl,
        is_closed_source: closedSource || !codeUrl,
        data_archive_url: sanitizeExternalUrl(rawEntry.data_archive_url || archiveUrl),
        citation: rawEntry.citation || citation || "Community Contribution (2026)",
        is_local_preview: true,
        has_preview_hydrographs: Boolean(
          rawEntry.hydrograph_samples || (STATE.previewInspectData && STATE.previewInspectData.sample_basins)
        ),
        hydrograph_samples: rawEntry.hydrograph_samples || null,
      });

      STATE.activePreviewEntryId = entry.entry_id;
      const manifest = buildSubmissionManifest(entry, recipe);
      const previewRecord = {
        entry,
        manifest,
        recipe,
        sanity: STATE.previewInspectData.sanity || null,
        sample_basins: STATE.previewInspectData.sample_basins || null,
        created_at: new Date().toISOString(),
      };
      upsertLocalPreview(previewRecord);
      await loadHydrographIndex();
      populateStep3Summary(previewRecord);
      renderLeaderboardTabs();
      renderActiveLeaderboardPanel();
      renderFamiliesGrid();

      // Switch directly to target leaderboard and scroll to highlighted preview row
      setContribStep(3);
      switchMainTab("leaderboards");
      selectLeaderboardTab(entry.leaderboard_id);
      requestAnimationFrame(() => {
        const targetTr = document.getElementById(`row-${entry.entry_id}`);
        if (targetTr) {
          targetTr.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      });
    } catch (err) {
      const msg = err && err.message ? err.message : String(err);
      if (scoreStatusEl) {
        scoreStatusEl.textContent = `Scoring failed: ${msg}`;
        scoreStatusEl.style.color = "var(--status-danger)";
      }
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  function populateStep3Summary(record) {
    const entry = (record && record.entry) || {};
    const noteEl = document.getElementById("step3-summary-note");
    const stripEl = document.getElementById("step3-metrics-strip");
    if (noteEl) {
      noteEl.innerHTML =
        `Saved <strong>${esc(entry.model_name || "Preview Model")}</strong> locally under key ` +
        `<code>${esc(LOCAL_PREVIEWS_STORAGE_KEY)}</code> on <code>${esc(entry.leaderboard_id || "")}</code> ` +
        `(#${esc(entry.rank || "?")} by Median NSE). Only visible in this browser until published via GitHub PR.`;
    }
    if (stripEl) {
      const boxes = [
        ["Rank", entry.rank ? `#${entry.rank}` : "—"],
        ["Median NSE", isValidMetric(entry.primary_ranking_nse) ? Number(entry.primary_ranking_nse).toFixed(3) : "N/A"],
        ["Mean NSE", isValidMetric(entry.mean_nse) ? Number(entry.mean_nse).toFixed(3) : "N/A"],
        ["Median KGE", isValidMetric(entry.med_kge) ? Number(entry.med_kge).toFixed(3) : "N/A"],
        ["Mean KGE", isValidMetric(entry.mean_kge) ? Number(entry.mean_kge).toFixed(3) : "N/A"],
        ["logNSE", isValidMetric(entry.gauch_log_nse) ? Number(entry.gauch_log_nse).toFixed(3) : "N/A"],
        ["logKGE", isValidMetric(entry.gauch_log_kge) ? Number(entry.gauch_log_kge).toFixed(3) : "N/A"],
        ["FHV (%)", isValidMetric(entry.gauch_fhv) ? `${Number(entry.gauch_fhv).toFixed(1)}%` : "N/A"],
        ["NSE ≤ 0", entry.nse_le_0 !== undefined && entry.nse_le_0 !== null ? String(entry.nse_le_0) : "N/A"],
      ];
      stripEl.innerHTML = boxes
        .map(
          ([lbl, val]) =>
            `<div class="score-metric-box"><span class="m-label">${esc(lbl)}</span><span class="m-val">${esc(val)}</span></div>`
        )
        .join("");
    }
  }

  function loadPreviewRecordIntoContribute(record) {
    if (!record || !record.entry) return;
    const e = record.entry;
    STATE.activePreviewEntryId = e.entry_id;
    const setVal = (id, v) => {
      const el = document.getElementById(id);
      if (el && v !== undefined && v !== null) el.value = String(v);
    };
    setVal("sub-spatial-regime", (e.leaderboard_id || "").split("__")[0] || "gauged");
    setVal("sub-forcing", e.forcing || "Daymet");
    setVal("sub-model-name", e.model_name || "");
    setVal("sub-paradigm", e.paradigm || "Machine learning");
    setVal("sub-category", e.category || "Contribution");
    setVal("sub-ensemble-info", e.ensemble_info || "");
    setVal("sub-github-url", e.code_url || e.github_url || "");
    setClosedSourceSelection(Boolean(e.closed_source || e.is_closed_source));
    setVal("sub-zenodo-doi", e.data_archive_url || e.zenodo_doi || "");
    setVal("sub-citation", e.citation || "");
    syncContributeTargetLeaderboard();
    updateLiveRepoBadgePreview();
    populateStep3Summary(record);
    syncStep4UrlInputsAndManifest();
  }

  function buildSubmissionManifest(scoredEntry = null, recipeOverride = null) {
    const lbId = syncContributeTargetLeaderboard();
    const spatialRegime = ((document.getElementById("sub-spatial-regime") || {}).value || "gauged").trim();
    const modelName = ((document.getElementById("sub-model-name") || {}).value || "").trim();
    const forcing = ((document.getElementById("sub-forcing") || {}).value || "Daymet").trim();
    const paradigm = ((document.getElementById("sub-paradigm") || {}).value || "Machine learning").trim();
    const category = ((document.getElementById("sub-category") || {}).value || "Contribution").trim();
    const ensembleInfo = ((document.getElementById("sub-ensemble-info") || {}).value || "").trim();

    const step4CodeEl = document.getElementById("step4-code-url");
    const step4ArchEl = document.getElementById("step4-archive-url");
    const closedSource = isClosedSourceSelected();
    const rawCode = closedSource
      ? ""
      : (step4CodeEl && step4CodeEl.value.trim()) || ((document.getElementById("sub-github-url") || {}).value || "").trim();
    const rawArch = (step4ArchEl && step4ArchEl.value.trim()) || ((document.getElementById("sub-zenodo-doi") || {}).value || "").trim();

    const codeUrl = closedSource ? "" : normalizeClientUrl(rawCode, "github");
    const archiveUrl = normalizeClientUrl(rawArch, "zenodo");
    const cit = ((document.getElementById("sub-citation") || {}).value || "").trim();
    const selectedFamId = ((document.getElementById("sub-family-id") || {}).value || "").trim();
    const newFamName = ((document.getElementById("sub-new-family-name") || {}).value || "").trim();
    const recipe = recipeOverride || (STATE.previewInspectData ? getConfirmedRecipeFromStep2() : null);

    const activePreviewRec =
      scoredEntry ||
      (STATE.activePreviewEntryId
        ? (loadLocalPreviews().find((p) => p.entry && p.entry.entry_id === STATE.activePreviewEntryId) || {}).entry
        : null);

    return {
      schema_version: "1.1",
      benchmark: "camels_us_daily",
      evaluation_window: "1989-10-01 to 1999-09-30",
      spatial_regime: spatialRegime,
      leaderboard_id: lbId,
      model_name: modelName || "Unspecified_Model",
      family_id: selectedFamId || null,
      new_family_name: selectedFamId ? null : newFamName || modelName || null,
      category: category,
      paradigm: paradigm,
      forcing: forcing,
      ensemble_info: ensembleInfo || null,
      citation: cit || "Community Contribution (2026)",
      archive_url: archiveUrl || rawArch,
      code_url: closedSource ? "" : codeUrl || rawCode,
      closed_source: closedSource,
      confirmed_recipe: recipe || null,
      preview_summary: activePreviewRec
        ? {
            median_nse: activePreviewRec.primary_ranking_nse ?? null,
            mean_nse: activePreviewRec.mean_nse ?? null,
            median_kge: activePreviewRec.med_kge ?? null,
            nse_le_0: activePreviewRec.nse_le_0 ?? null,
          }
        : null,
      compliance_attestation: {
        blind_holdout_window: Boolean((document.getElementById("chk-rule-1") || {}).checked),
        strict_spatial_holdout_canonical_folds: Boolean((document.getElementById("chk-rule-2") || {}).checked),
        all_forcing_inputs_declared: Boolean((document.getElementById("chk-rule-3") || {}).checked),
        deterministic_daily_predictions_mm_per_day: Boolean((document.getElementById("chk-rule-4") || {}).checked),
      },
      generated_at: new Date().toISOString(),
    };
  }

  function syncStep4UrlInputsAndManifest() {
    const s1Arch = ((document.getElementById("sub-zenodo-doi") || {}).value || "").trim();
    const s1Code = ((document.getElementById("sub-github-url") || {}).value || "").trim();
    const s4Arch = document.getElementById("step4-archive-url");
    const s4Code = document.getElementById("step4-code-url");
    if (s4Arch && !s4Arch.value.trim() && s1Arch) s4Arch.value = normalizeClientUrl(s1Arch, "zenodo") || s1Arch;
    if (s4Code && !s4Code.value.trim() && s1Code) s4Code.value = normalizeClientUrl(s1Code, "github") || s1Code;

    const manifest = buildSubmissionManifest();
    const manifestEl = document.getElementById("submission-manifest-preview");
    const sublineEl = document.getElementById("preview-subline");
    if (manifestEl) manifestEl.textContent = JSON.stringify(manifest, null, 2);
    if (sublineEl) {
      sublineEl.textContent =
        `Target leaderboard: ${manifest.leaderboard_id} · Verified automatically via GitHub Actions upon issue/PR submission.`;
    }
    return manifest;
  }

  function validateStep4RequiredUrls() {
    const msgEl = document.getElementById("step4-validation-msg");
    const s4ArchEl = document.getElementById("step4-archive-url");
    const s4CodeEl = document.getElementById("step4-code-url");
    const modelName = ((document.getElementById("sub-model-name") || {}).value || "").trim();

    const archUrl = normalizeClientUrl(s4ArchEl ? s4ArchEl.value : "", "zenodo");
    const closedSource = isClosedSourceSelected();
    const codeUrl = closedSource ? "" : normalizeClientUrl(s4CodeEl ? s4CodeEl.value : "", "github");

    if (!modelName) {
      if (msgEl) {
        msgEl.textContent = "Please enter a Model Name in Step 1 before generating an official submission.";
        msgEl.style.color = "var(--status-danger)";
      }
      return null;
    }
    if (!archUrl) {
      if (msgEl) {
        msgEl.textContent = "A valid permanent Archive DOI/URL (https:// or http://, e.g., Zenodo or HydroShare) is required.";
        msgEl.style.color = "var(--status-danger)";
      }
      return null;
    }
    if (!closedSource && !codeUrl) {
      if (msgEl) {
        msgEl.textContent = "Enter a valid Code Repository URL (https:// or http://) or check Closed-Source / Proprietary.";
        msgEl.style.color = "var(--status-danger)";
      }
      return null;
    }

    if (s4ArchEl) s4ArchEl.value = archUrl;
    if (s4CodeEl && !closedSource) s4CodeEl.value = codeUrl;
    const s1Arch = document.getElementById("sub-zenodo-doi");
    const s1Code = document.getElementById("sub-github-url");
    if (s1Arch) s1Arch.value = archUrl;
    if (s1Code && !closedSource) s1Code.value = codeUrl;

    if (msgEl) {
      msgEl.textContent = closedSource
        ? "Archive DOI validated (Closed-Source / Proprietary codebase declaration active)."
        : "Archive DOI and open-source code URL validated.";
      msgEl.style.color = "var(--status-success)";
    }
    return syncStep4UrlInputsAndManifest();
  }

  function openGitHubSubmissionIssue() {
    const manifest = validateStep4RequiredUrls();
    if (!manifest) return;
    const jsonText = JSON.stringify(manifest, null, 2);
    const title = encodeURIComponent(
      `[Model Submission] ${manifest.model_name} (${manifest.leaderboard_id})`
    );
    const codeDisplay = manifest.closed_source
      ? "Closed-Source / Proprietary (no public code URL)"
      : manifest.code_url || "None";
    const body = encodeURIComponent(
      `### Model Submission\n` +
        `- **Model:** ${manifest.model_name}\n` +
        `- **Leaderboard:** \`${manifest.leaderboard_id}\`\n` +
        `- **Archive DOI / URL:** ${manifest.archive_url}\n` +
        `- **Code Repository:** ${codeDisplay}\n\n` +
        `### Canonical Submission Manifest\n\`\`\`json\n${jsonText}\n\`\`\`\n`
    );
    const issueUrl =
      `https://github.com/grey-nearing/camels-us-mip/issues/new` +
      `?template=model_submission.yml&title=${title}&body=${body}`;
    window.open(issueUrl, "_blank", "noopener,noreferrer");
  }

  function downloadManifestJsonObject(manifest) {
    const slug = (manifest.model_name || "model").replace(/[^a-z0-9]+/gi, "_").toLowerCase();
    const blob = new Blob([JSON.stringify(manifest, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `hydrologymip_submission_${slug}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function exportPullRequestManifest() {
    const manifest = validateStep4RequiredUrls();
    if (!manifest) return;
    downloadManifestJsonObject(manifest);
    const prGuide = document.getElementById("step4-pr-guide");
    if (prGuide) {
      prGuide.hidden = false;
      prGuide.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  function handleSelectedPredictionFile(file) {
    const statusEl = document.getElementById("inspect-status-msg");
    const dropzone = document.getElementById("sub-dropzone");
    const bar = document.getElementById("sub-selected-file-bar");
    const lbl = document.getElementById("sub-selected-file-label");

    if (!file) {
      STATE.uploadedFile = null;
      if (dropzone) dropzone.classList.remove("has-file");
      if (bar) bar.hidden = true;
      if (lbl) lbl.textContent = "";
      return;
    }

    const nameLow = String(file.name || "").toLowerCase();
    if (nameLow.endsWith(".pkl") || nameLow.endsWith(".pickle")) {
      STATE.uploadedFile = null;
      if (dropzone) dropzone.classList.remove("has-file");
      if (bar) bar.hidden = true;
      if (statusEl) {
        statusEl.textContent =
          "Pickle files (.pkl / .pickle) are rejected for security. Please export predictions as .nc, .zarr.zip, .csv, .parquet, or .npz.";
        statusEl.style.color = "var(--status-danger)";
      }
      return;
    }

    const okExt = [".nc", ".nc4", ".cdf", ".zarr.zip", ".zip", ".csv", ".parquet", ".pq", ".npz"].some(
      (ext) => nameLow.endsWith(ext)
    );
    if (!okExt) {
      STATE.uploadedFile = null;
      if (statusEl) {
        statusEl.textContent =
          "Unsupported file extension. Allowed formats: .nc, .zarr.zip, .csv, .parquet, .npz.";
        statusEl.style.color = "var(--status-danger)";
      }
      return;
    }

    STATE.uploadedFile = file;
    const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
    if (dropzone) dropzone.classList.add("has-file");
    if (bar) bar.hidden = false;
    if (lbl) lbl.textContent = `${file.name} (${sizeMb} MB)`;
    if (statusEl) {
      statusEl.textContent = `Selected '${file.name}'. Click "Inspect & Validate File" to verify schema and units.`;
      statusEl.style.color = "var(--color-ink-2)";
    }
  }

  // =========================================================================
  // 8A. Hydrograph, Hyetograph & Temperature Comparison Explorer
  // =========================================================================
  let HYDRO_AVAILABLE_KEYS_SET = new Set();
  let HYDRO_MODELS_BY_KEY = new Map();
  let HYDRO_CANVAS_GEOM = null;
  let HYDRO_DRAG_STATE = null;

  async function loadHydrographIndex() {
    try {
      const res = await fetch("/api/hydrographs/index");
      if (!res.ok) return;
      const data = await parseApiJson(res);
      STATE.hydroIndex = data;
      HYDRO_AVAILABLE_KEYS_SET = new Set();
      HYDRO_MODELS_BY_KEY = new Map();
      for (const m of data.models || []) {
        if (m && m.model_key) {
          HYDRO_AVAILABLE_KEYS_SET.add(m.model_key);
          HYDRO_MODELS_BY_KEY.set(m.model_key, m);
        }
      }
      // Also register any local preview or submitted entries that carry hydrographs
      for (const lb of STATE.allLeaderboards || []) {
        for (const r of lb.rows || []) {
          if (r && (r.has_hydrograph || (r.is_local_preview && r.has_preview_hydrographs))) {
            const mk = r.hydrograph_model_key || r.model_key || `${lb.id}__${r.entry_id}`;
            HYDRO_AVAILABLE_KEYS_SET.add(mk);
            if (!HYDRO_MODELS_BY_KEY.has(mk)) {
              const mMeta = {
                model_key: mk,
                entry_id: r.entry_id,
                leaderboard_id: lb.id,
                model_name: r.model_name,
                family_id: r.family_id || "",
                forcing: r.forcing || "",
                category: r.category || "Contribution",
                paradigm: r.paradigm || "Machine learning",
                is_ensemble: Boolean(r.is_ensemble),
                primary_ranking_nse: r.primary_ranking_nse,
                med_kge: r.med_kge,
              };
              HYDRO_MODELS_BY_KEY.set(mk, mMeta);
              if (Array.isArray(data.models)) {
                data.models.push(mMeta);
              }
            }
            if (
              Array.isArray(r.per_basin_nse) &&
              r.per_basin_nse.length === 531 &&
              data.model_basin_nse
            ) {
              data.model_basin_nse[mk] = r.per_basin_nse;
            }
          }
        }
      }
      populateHydroExplorerControls();
    } catch (err) {
      console.warn("Hydrograph index not available yet:", err);
    }
  }

  function isModelHydrographAvailable(row, lbId) {
    if (!row) return false;
    if (row.is_local_preview && row.has_preview_hydrographs) return true;
    if (row.has_hydrograph === true) return true;
    const mk = row.model_key || `${lbId || row.leaderboard_id || ""}__${row.entry_id}`;
    return HYDRO_AVAILABLE_KEYS_SET.has(mk);
  }

  function findModelMetadataByKey(modelKey) {
    if (HYDRO_MODELS_BY_KEY.has(modelKey)) {
      return HYDRO_MODELS_BY_KEY.get(modelKey);
    }
    for (const lb of STATE.allLeaderboards) {
      for (const r of lb.rows || []) {
        const mk = r.model_key || `${lb.id}__${r.entry_id}`;
        if (mk === modelKey) {
          return {
            model_key: mk,
            leaderboard_id: lb.id,
            leaderboard_title: lb.short_title || lb.title,
            entry_id: r.entry_id,
            model_name: r.model_name,
            family_id: r.family_id,
            paradigm: r.paradigm,
            category: r.category,
            forcing: r.forcing,
            rank: r.rank,
            primary_ranking_nse: r.primary_ranking_nse,
            med_kge: r.med_kge,
            citation: r.citation,
          };
        }
      }
    }
    return null;
  }

  function ensureDefaultComparedModels() {
    if (STATE.comparedModels.length > 0) return;
    const activeLbId = getActiveLeaderboardId();
    const candidates = [];

    // Prefer top 2 available hydrograph models from the currently active leaderboard, plus a contrasting model
    const activeLb = STATE.allLeaderboards.find((b) => b.id === activeLbId);
    if (activeLb && activeLb.rows) {
      for (const r of activeLb.rows) {
        const mk = r.model_key || `${activeLb.id}__${r.entry_id}`;
        if (isModelHydrographAvailable(r, activeLb.id) && !candidates.includes(mk)) {
          candidates.push(mk);
          if (candidates.length >= 2) break;
        }
      }
    }

    const fallbackPreferred = [
      "gauged__single_camels__t3_02",
      "gauged__single_camels__t3_18",
      "gauged__single_camels__sacsma_daymet",
    ];
    for (const fk of fallbackPreferred) {
      if (HYDRO_AVAILABLE_KEYS_SET.has(fk) && !candidates.includes(fk) && candidates.length < 3) {
        candidates.push(fk);
      }
    }

    if (candidates.length === 0 && STATE.hydroIndex && (STATE.hydroIndex.models || []).length > 0) {
      candidates.push(STATE.hydroIndex.models[0].model_key);
    }
    STATE.comparedModels = candidates;
  }

  function toggleModelInHydroComparison(modelKey) {
    if (!modelKey) return;
    const idx = STATE.comparedModels.indexOf(modelKey);
    if (idx >= 0) {
      STATE.comparedModels.splice(idx, 1);
    } else {
      if (STATE.comparedModels.length >= 8) {
        STATE.comparedModels.shift();
      }
      STATE.comparedModels.push(modelKey);
    }
    updateCompareUiState();
    if (STATE.activeView === "leaderboards") {
      renderActiveLeaderboardPanel();
    } else if (STATE.activeView === "hydrographs") {
      renderHydroSelectedModelChips();
      populateHydroModelPickerDropdown();
      fetchAndRenderHydroBasin();
    }
  }

  function updateCompareUiState() {
    const count = STATE.comparedModels.length;
    const badge = document.getElementById("lb-compare-count-badge");
    if (badge) badge.textContent = String(count);

    const tray = document.getElementById("hydrograph-compare-tray");
    const trayCount = document.getElementById("tray-compare-count");
    const trayChips = document.getElementById("tray-compare-chips");
    if (!tray) return;

    const showTray = STATE.activeView === "leaderboards" && count > 0;
    tray.hidden = !showTray;
    if (trayCount) trayCount.textContent = String(count);

    if (trayChips && showTray) {
      trayChips.innerHTML = STATE.comparedModels
        .map((mk, idx) => {
          const meta = findModelMetadataByKey(mk) || { model_name: mk, leaderboard_id: "" };
          const lbObj = STATE.allLeaderboards.find((b) => b.id === meta.leaderboard_id);
          const lbShort = lbObj ? formatLeaderboardShortLabel(lbObj) : meta.leaderboard_id;
          const color = HYDRO_MODEL_COLORS[idx % HYDRO_MODEL_COLORS.length];
          return `
            <span class="tray-model-chip">
              <span class="chip-swatch" style="background: ${color};"></span>
              <span>${esc(formatShortLeaderName(meta))}</span>
              <span class="chip-lb-tag">${esc(lbShort)}</span>
              <button type="button" data-tray-remove-model="${esc(mk)}" title="Remove model">&times;</button>
            </span>
          `;
        })
        .join("");

      trayChips.querySelectorAll("[data-tray-remove-model]").forEach((btn) => {
        btn.addEventListener("click", () => {
          toggleModelInHydroComparison(btn.getAttribute("data-tray-remove-model"));
        });
      });
    }
  }

  function populateHydroExplorerControls() {
    if (!STATE.hydroIndex) return;

    // 1. Populate Leaderboard filter in Model Picker
    const lbFilter = document.getElementById("hydro-add-lb-filter");
    if (lbFilter) {
      const lbs = STATE.allLeaderboards || [];
      lbFilter.innerHTML =
        `<option value="">All Leaderboards (${(STATE.hydroIndex.models || []).length} models)</option>` +
        lbs
          .map(
            (lb) =>
              `<option value="${esc(lb.id)}" ${
                STATE.hydroAddLbFilter === lb.id ? "selected" : ""
              }>${esc(formatLeaderboardShortLabel(lb))}</option>`
          )
          .join("");
    }

    populateHydroModelPickerDropdown();

    // 2. Populate HUC-02 Region filter
    const hucSel = document.getElementById("hydro-huc-filter");
    if (hucSel) {
      const basins = STATE.hydroIndex.basins || [];
      const hucCounts = new Map();
      const hucNames = new Map();
      for (const b of basins) {
        hucCounts.set(b.huc_02, (hucCounts.get(b.huc_02) || 0) + 1);
        hucNames.set(b.huc_02, b.huc_name || `Region ${b.huc_02}`);
      }
      const hucs = Array.from(hucCounts.keys()).sort();
      hucSel.innerHTML =
        `<option value="">All 18 HUC-02 Regions (${basins.length} Basins)</option>` +
        hucs
          .map(
            (h) =>
              `<option value="${esc(h)}" ${
                STATE.hydroHucFilter === h ? "selected" : ""
              }>HUC ${esc(h)} &mdash; ${esc(hucNames.get(h))} (${hucCounts.get(h)})</option>`
          )
          .join("");
    }

    populateHydroBasinDropdown();
  }

  function populateHydroModelPickerDropdown() {
    const sel = document.getElementById("hydro-add-model-select");
    if (!sel || !STATE.hydroIndex) return;
    const allModels = STATE.hydroIndex.models || [];
    const lbFilter = STATE.hydroAddLbFilter;
    const filtered = lbFilter
      ? allModels.filter((m) => m.leaderboard_id === lbFilter)
      : allModels;

    const byLb = new Map();
    for (const m of filtered) {
      const list = byLb.get(m.leaderboard_id) || [];
      list.push(m);
      byLb.set(m.leaderboard_id, list);
    }

    let html = `<option value="">+ Add Model Across Leaderboards (${filtered.length} available)...</option>`;
    for (const [lbId, models] of byLb.entries()) {
      const lbObj = STATE.allLeaderboards.find((b) => b.id === lbId);
      const groupLabel = lbObj ? lbObj.title : lbId;
      html += `<optgroup label="${esc(groupLabel)}">`;
      for (const m of models) {
        const already = STATE.comparedModels.includes(m.model_key);
        const nseStr =
          m.primary_ranking_nse != null && m.primary_ranking_nse > -100
            ? Number(m.primary_ranking_nse).toFixed(3)
            : "N/A";
        html += `<option value="${esc(m.model_key)}" ${already ? "disabled" : ""}>${
          already ? "✓ " : ""
        }#${m.rank} ${esc(m.model_name)} [${esc(m.forcing)}] (Med NSE ${nseStr})</option>`;
      }
      html += `</optgroup>`;
    }
    sel.innerHTML = html;
  }

  function getFilteredHydroBasins() {
    if (!STATE.hydroIndex || !STATE.hydroIndex.basins) return [];
    const huc = STATE.hydroHucFilter;
    const q = (STATE.hydroBasinSearch || "").trim().toLowerCase();
    return STATE.hydroIndex.basins.filter((b) => {
      if (huc && b.huc_02 !== huc) return false;
      if (q) {
        const hay = `${b.basin_id} ${b.gauge_name || ""} ${b.huc_02} ${b.huc_name || ""}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }

  function populateHydroBasinDropdown() {
    const sel = document.getElementById("hydro-basin-select");
    if (!sel || !STATE.hydroIndex) return;
    const list = getFilteredHydroBasins();
    if (list.length === 0) {
      sel.innerHTML = `<option value="">No basins match filter</option>`;
      return;
    }
    if (!list.some((b) => b.basin_id === STATE.hydroBasinId)) {
      STATE.hydroBasinId = list[0].basin_id;
    }
    sel.innerHTML = list
      .map(
        (b) =>
          `<option value="${esc(b.basin_id)}" ${
            b.basin_id === STATE.hydroBasinId ? "selected" : ""
          }>${esc(b.basin_id)} &mdash; ${esc(b.gauge_name || "")} [HUC ${esc(
            b.huc_02
          )}, ${Number(b.area_km2 || 0).toFixed(0)} km&sup2;]</option>`
      )
      .join("");
  }

  function renderHydroSelectedModelChips() {
    const wrap = document.getElementById("hydro-selected-chips-row") || document.getElementById("hydro-selected-models-chips");
    if (!wrap) return;

    const seriesMap = new Map();
    if (STATE.hydroBasinPayload && STATE.hydroBasinPayload.series) {
      for (const s of STATE.hydroBasinPayload.series) {
        seriesMap.set(s.model_key, s);
      }
    }

    let html = `
      <span class="hydro-model-chip chip-qobs" title="USGS Observed Daily Streamflow (mm/day)">
        <span class="chip-color-bar" style="background: #0f172a; border: 1px solid #94a3b8;"></span>
        <span>Observed Streamflow (USGS Q<sub>obs</sub>)</span>
      </span>
    `;

    STATE.comparedModels.forEach((mk, idx) => {
      const meta = findModelMetadataByKey(mk) || { model_name: mk, leaderboard_id: "" };
      const lbObj = STATE.allLeaderboards.find((b) => b.id === meta.leaderboard_id);
      const lbShort = lbObj ? formatLeaderboardShortLabel(lbObj) : meta.leaderboard_id;
      const color = HYDRO_MODEL_COLORS[idx % HYDRO_MODEL_COLORS.length];
      const sData = seriesMap.get(mk);
      const basinNse =
        sData && sData.basin_metrics && sData.basin_metrics.NSE != null
          ? Number(sData.basin_metrics.NSE).toFixed(3)
          : null;

      html += `
        <span class="hydro-model-chip" style="border-left: 4px solid ${color};">
          <span class="chip-color-bar" style="background: ${color};"></span>
          <span>${esc(meta.model_name)}</span>
          <span class="chip-meta">${esc(lbShort)} &middot; ${esc(meta.forcing || "")}</span>
          ${
            basinNse !== null
              ? `<span class="chip-nse-pill" title="Catchment NSE on Basin ${esc(
                  STATE.hydroBasinId
                )}">Basin NSE ${esc(basinNse)}</span>`
              : ""
          }
          <button
            type="button"
            class="chip-remove-btn"
            data-hydro-remove-chip="${esc(mk)}"
            title="Remove model from comparison"
          >&times;</button>
        </span>
      `;
    });

    wrap.innerHTML = html;
    wrap.querySelectorAll("[data-hydro-remove-chip]").forEach((btn) => {
      btn.addEventListener("click", () => {
        toggleModelInHydroComparison(btn.getAttribute("data-hydro-remove-chip"));
      });
    });
  }

  function renderHydroBasinMetaStrip(basin) {
    const strip = document.getElementById("hydro-basin-meta-strip");
    if (!strip || !basin) return;
    strip.innerHTML = `
      <span class="hydro-meta-item"><span class="meta-k">USGS Gauge:</span><span class="meta-v">${esc(
        basin.basin_id
      )}</span></span>
      <span class="hydro-meta-item"><span class="meta-k">Station:</span><span class="meta-v">${esc(
        basin.gauge_name || basin.basin_id
      )}</span></span>
      <span class="hydro-meta-item"><span class="meta-k">Region:</span><span class="meta-v">HUC ${esc(
        basin.huc_02
      )} (${esc(basin.huc_name || "")})</span></span>
      <span class="hydro-meta-item"><span class="meta-k">Drainage Area:</span><span class="meta-v">${Number(
        basin.area_km2 || 0
      ).toFixed(1)} km&sup2;</span></span>
      <span class="hydro-meta-item"><span class="meta-k">Elevation:</span><span class="meta-v">${Number(
        basin.elev_m || 0
      ).toFixed(0)} m</span></span>
      <span class="hydro-meta-item"><span class="meta-k">Coords:</span><span class="meta-v">${Number(
        basin.lat || 0
      ).toFixed(2)}&deg;N, ${Number(basin.lon || 0).toFixed(2)}&deg;E</span></span>
      <span class="hydro-meta-item"><span class="meta-k">Mean / Peak Q<sub>obs</sub>:</span><span class="meta-v">${Number(
        basin.mean_qobs || 0
      ).toFixed(2)} / ${Number(basin.max_qobs || 0).toFixed(1)} mm/d</span></span>
    `;
  }

  function applyHydroZoomPreset(presetKey) {
    const payload = STATE.hydroBasinPayload;
    if (!payload || !payload.dates) return;
    const dates = payload.dates;
    const nDays = dates.length;
    STATE.hydroActivePreset = presetKey;

    if (presetKey === "full") {
      STATE.hydroZoomStart = 0;
      STATE.hydroZoomEnd = nDays - 1;
    } else if (presetKey === "peak1" || presetKey === "peak2") {
      const qobs = payload.qobs || [];
      let peak1Idx = Math.floor(nDays / 2);
      let peak1Val = -Infinity;
      for (let i = 0; i < nDays; i++) {
        const v = qobs[i];
        if (v !== null && v !== undefined && v > peak1Val) {
          peak1Val = v;
          peak1Idx = i;
        }
      }
      let centerIdx = peak1Idx;
      if (presetKey === "peak2") {
        let peak2Idx = peak1Idx;
        let peak2Val = -Infinity;
        for (let i = 0; i < nDays; i++) {
          if (Math.abs(i - peak1Idx) < 60) continue;
          const v = qobs[i];
          if (v !== null && v !== undefined && v > peak2Val) {
            peak2Val = v;
            peak2Idx = i;
          }
        }
        centerIdx = peak2Idx;
      }
      STATE.hydroZoomStart = Math.max(0, centerIdx - 45);
      STATE.hydroZoomEnd = Math.min(nDays - 1, centerIdx + 45);
    } else if (presetKey.startsWith("wy")) {
      const wyYear = Number(presetKey.slice(2));
      const startStr = `${wyYear - 1}-10-01`;
      const endStr = `${wyYear}-09-30`;
      const sIdx = dates.indexOf(startStr);
      const eIdx = dates.indexOf(endStr);
      STATE.hydroZoomStart = sIdx >= 0 ? sIdx : 0;
      STATE.hydroZoomEnd = eIdx >= 0 ? eIdx : nDays - 1;
    }

    syncHydroZoomButtonsAndDates();
    renderHydrographCanvas();
  }

  function syncHydroZoomButtonsAndDates() {
    document.querySelectorAll("[data-hydro-zoom]").forEach((btn) => {
      btn.classList.toggle(
        "active",
        btn.getAttribute("data-hydro-zoom") === STATE.hydroActivePreset
      );
    });
    const payload = STATE.hydroBasinPayload;
    if (!payload || !payload.dates) return;
    const dates = payload.dates;
    const sEl = document.getElementById("hydro-start-date");
    const eEl = document.getElementById("hydro-end-date");
    if (sEl && dates[STATE.hydroZoomStart]) sEl.value = dates[STATE.hydroZoomStart];
    if (eEl && dates[STATE.hydroZoomEnd]) eEl.value = dates[STATE.hydroZoomEnd];
  }

  function renderHydroBasinMetricsTable() {
    const wrap = document.getElementById("hydro-basin-metrics-table-wrap");
    const sub = document.getElementById("hydro-metrics-table-subtitle");
    const payload = STATE.hydroBasinPayload;
    if (!wrap || !payload) return;

    const basin = payload.basin || {};
    if (sub) {
      sub.textContent = `Basin ${basin.basin_id} (${basin.gauge_name || ""}) vs. 531-Basin Benchmark Medians (1989-10-01 to 1999-09-30)`;
    }

    const series = payload.series || [];
    if (series.length === 0) {
      wrap.innerHTML = `<div style="padding: 1rem; color: var(--color-muted); font-size: 0.82rem;">Select one or more models above to compare catchment performance metrics.</div>`;
      return;
    }

    let html = `
      <table class="lb-table" style="font-size: 0.8rem;">
        <thead>
          <tr>
            <th>Model</th>
            <th>Leaderboard Regime</th>
            <th>Forcing</th>
            <th class="num-col" title="Single-basin NSE on this catchment">Basin NSE</th>
            <th class="num-col" title="531-Basin Benchmark Median NSE">531-Basin Med NSE</th>
            <th class="num-col" title="Single-basin KGE on this catchment">Basin KGE</th>
            <th class="num-col" title="Single-basin logNSE on log(max(Q, 0) + 1e-5)">Basin logNSE</th>
            <th class="num-col" title="Single-basin logKGE on log(max(Q, 0) + 1e-5)">Basin logKGE</th>
            <th class="num-col" title="Single-basin Pearson r on this catchment">Basin r</th>
            <th class="num-col" title="Single-basin Peak-Flow Bias FHV (%)">Basin FHV (%)</th>
            <th class="num-col" title="Single-basin Low-Flow Bias FLV (%)">Basin FLV (%)</th>
            <th class="num-col" title="Single-basin RMSE (mm/day)">Basin RMSE (mm/d)</th>
            <th class="num-col" title="Mean Peak Timing Error (days)">Peak Timing (d)</th>
          </tr>
        </thead>
        <tbody>
    `;

    series.forEach((s, idx) => {
      const color = HYDRO_MODEL_COLORS[idx % HYDRO_MODEL_COLORS.length];
      const lbObj = STATE.allLeaderboards.find((b) => b.id === s.leaderboard_id);
      const lbShort = lbObj ? formatLeaderboardShortLabel(lbObj) : s.leaderboard_id;
      const bm = s.basin_metrics || {};
      html += `
        <tr>
          <td style="font-weight: 600;">
            <span style="display: inline-flex; align-items: center; gap: 0.45rem;">
              <span style="width: 10px; height: 10px; border-radius: 3px; background: ${color}; display: inline-block; flex-shrink: 0;"></span>
              <span>${esc(s.model_name)}</span>
            </span>
          </td>
          <td><span class="cat-badge">${esc(lbShort)}</span></td>
          <td class="forcing-cell">${esc(s.forcing || "")}</td>
          <td class="num-col" style="font-weight: 700;">${fmtNum(bm.NSE, 3)}</td>
          <td class="num-col">${fmtNum(s.primary_ranking_nse, 3)}</td>
          <td class="num-col">${fmtNum(bm.KGE, 3)}</td>
          <td class="num-col">${fmtNum(bm.logNSE ?? bm.log_nse, 3)}</td>
          <td class="num-col">${fmtNum(bm.logKGE ?? bm.log_kge, 3)}</td>
          <td class="num-col">${fmtNum(bm.Pearson_r, 3)}</td>
          <td class="num-col">${bm.FHV != null ? `${Number(bm.FHV).toFixed(1)}%` : "&mdash;"}</td>
          <td class="num-col">${bm.FLV != null ? `${Number(bm.FLV).toFixed(1)}%` : "&mdash;"}</td>
          <td class="num-col">${fmtNum(bm.RMSE, 2)}</td>
          <td class="num-col">${bm.Peak_Timing != null ? `${Number(bm.Peak_Timing).toFixed(2)} d` : "&mdash;"}</td>
        </tr>
      `;
    });

    html += `</tbody></table>`;
    wrap.innerHTML = html;
  }

  async function fetchAndRenderHydroBasin(opts = {}) {
    ensureDefaultComparedModels();
    if (!STATE.hydroIndex) {
      await loadHydrographIndex();
    }
    const basinId = STATE.hydroBasinId || "01022500";
    const modelsParam = STATE.comparedModels.join(",");
    try {
      const url = `/api/hydrographs/basin/${encodeURIComponent(
        basinId
      )}?models=${encodeURIComponent(modelsParam)}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const payload = await res.json();

      // Overlay any local preview hydrographs for this basin if compared
      if (Array.isArray(payload.series)) {
        const previewItems = loadLocalPreviews();
        for (const item of previewItems) {
          const r = item && item.entry;
          if (!r) continue;
          const mk = r.model_key || `${r.leaderboard_id}__${r.entry_id}`;
          if (!STATE.comparedModels.includes(mk)) continue;
          if (payload.series.some((s) => s.model_key === mk)) continue;
          const sample =
            (r.hydrograph_samples && r.hydrograph_samples[basinId]) ||
            (Array.isArray(item.sample_basins)
              ? item.sample_basins.find((sb) => sb.basin_id === basinId)
              : null);
          if (sample && Array.isArray(sample.qsim)) {
            payload.series.push({
              model_key: mk,
              entry_id: r.entry_id,
              leaderboard_id: r.leaderboard_id,
              model_name: `${r.model_name} (Local Preview)`,
              forcing: r.forcing,
              category: r.category || "Contribution",
              primary_ranking_nse: r.primary_ranking_nse,
              qsim: sample.qsim,
              basin_metrics: sample.basin_metrics || { NSE: sample.nse ?? null },
            });
          }
        }
      }

      STATE.hydroBasinPayload = payload;

      renderHydroBasinMetaStrip(payload.basin);
      renderHydroSelectedModelChips();
      renderHydroBasinMetricsTable();
      renderHydroConusMap();

      if (
        opts.recenterPeak ||
        STATE.hydroActivePreset === "peak1" ||
        STATE.hydroActivePreset === "peak2"
      ) {
        applyHydroZoomPreset(STATE.hydroActivePreset || "peak1");
      } else {
        syncHydroZoomButtonsAndDates();
        renderHydrographCanvas();
      }
    } catch (err) {
      console.error("Failed to load basin hydrographs:", err);
    }
  }

  function openHydrographExplorer(opts = {}) {
    ensureDefaultComparedModels();
    updateCompareUiState();
    switchMainTab("hydrographs", opts);
    populateHydroExplorerControls();
    renderHydroConusMap();
    fetchAndRenderHydroBasin({ recenterPeak: false });
  }

  const HUC02_PALETTE = {
    "01": "#2563eb",
    "02": "#0284c7",
    "03": "#0d9488",
    "04": "#059669",
    "05": "#16a34a",
    "06": "#65a30d",
    "07": "#ca8a04",
    "08": "#d97706",
    "09": "#ea580c",
    "10": "#dc2626",
    "11": "#e11d48",
    "12": "#db2777",
    "13": "#9333ea",
    "14": "#7c3aed",
    "15": "#4f46e5",
    "16": "#0891b2",
    "17": "#1d4ed8",
    "18": "#0f766e",
  };

  function colorForNse(nse) {
    if (nse === null || nse === undefined || Number.isNaN(Number(nse))) return "#94a3b8";
    const v = Number(nse);
    if (v <= 0.0) return "#dc2626";
    if (v < 0.35) return "#ea580c";
    if (v < 0.55) return "#eab308";
    if (v < 0.70) return "#10b981";
    if (v < 0.82) return "#0284c7";
    return "#1e3a8a";
  }

  function colorForNseDiff(diff) {
    if (diff === null || diff === undefined || Number.isNaN(Number(diff))) return "#94a3b8";
    const d = Number(diff);
    if (d <= -0.15) return "#b91c1c";
    if (d <= -0.05) return "#f97316";
    if (d < 0.05) return "#cbd5e1";
    if (d < 0.15) return "#38bdf8";
    return "#1d4ed8";
  }

  function colorForMeanQ(q) {
    const v = Number(q) || 0;
    if (v < 0.3) return "#d97706";
    if (v < 0.8) return "#eab308";
    if (v < 1.8) return "#10b981";
    if (v < 3.5) return "#0284c7";
    return "#1e3a8a";
  }

  function applyConusMapViewBox() {
    const svg = document.getElementById("hydro-conus-svg");
    if (!svg) return;
    const vb = STATE.hydroMapViewBox || { x: 0, y: 0, w: 960, h: 580 };
    svg.setAttribute(
      "viewBox",
      `${vb.x.toFixed(1)} ${vb.y.toFixed(1)} ${vb.w.toFixed(1)} ${vb.h.toFixed(1)}`
    );
  }

  function renderHydroConusMap() {
    const svg = document.getElementById("hydro-conus-svg");
    const legend = document.getElementById("hydro-map-legend");
    if (!svg || !STATE.hydroIndex || !STATE.hydroIndex.conus_map) return;

    const conus = STATE.hydroIndex.conus_map;
    const basins = STATE.hydroIndex.basins || [];
    const nseMap = STATE.hydroIndex.model_basin_nse || {};
    const m1Key = STATE.comparedModels[0] || "";
    const m2Key = STATE.comparedModels[1] || "";
    const m1Meta = m1Key ? findModelMetadataByKey(m1Key) : null;
    const m2Meta = m2Key ? findModelMetadataByKey(m2Key) : null;
    const m1NseArr = m1Key && nseMap[m1Key] ? nseMap[m1Key] : null;
    const m2NseArr = m2Key && nseMap[m2Key] ? nseMap[m2Key] : null;

    let colorMode = STATE.hydroMapColorBy || "nse_primary";
    if (colorMode === "nse_diff" && (!m1NseArr || !m2NseArr)) {
      colorMode = "nse_primary";
    }

    const filteredSet = new Set(getFilteredHydroBasins().map((b) => b.basin_id));
    const hasFilter = Boolean(STATE.hydroHucFilter || (STATE.hydroBasinSearch || "").trim());

    const statesSvg = (conus.states || [])
      .map(
        (st) =>
          `<path d="${st.d}" class="conus-state-fill"><title>${esc(st.name)}</title></path>`
      )
      .join("");

    const nationSvg = conus.nation_path
      ? `<path d="${conus.nation_path}" class="conus-nation-outline" />`
      : "";

    // Sort so selected basin is rendered last (on top)
    const indexedBasins = basins.map((b, i) => ({ b, i }));
    indexedBasins.sort((a, z) => {
      const aSel = a.b.basin_id === STATE.hydroBasinId ? 1 : 0;
      const zSel = z.b.basin_id === STATE.hydroBasinId ? 1 : 0;
      if (aSel !== zSel) return aSel - zSel;
      return (z.b.area_km2 || 0) - (a.b.area_km2 || 0);
    });

    const basinsSvg = indexedBasins
      .map(({ b, i }) => {
        const isSel = b.basin_id === STATE.hydroBasinId;
        const isDimmed = hasFilter && !filteredSet.has(b.basin_id) && !isSel;
        let fill = "#0284c7";
        if (colorMode === "nse_primary" && m1NseArr) {
          fill = colorForNse(m1NseArr[i]);
        } else if (colorMode === "nse_diff" && m1NseArr && m2NseArr) {
          const v1 = m1NseArr[i];
          const v2 = m2NseArr[i];
          fill = v1 != null && v2 != null ? colorForNseDiff(v1 - v2) : "#94a3b8";
        } else if (colorMode === "huc02") {
          fill = HUC02_PALETTE[b.huc_02] || "#0284c7";
        } else if (colorMode === "mean_qobs") {
          fill = colorForMeanQ(b.mean_qobs);
        }

        const r = isSel ? 6.8 : 4.2;
        const polyHtml = b.poly_d
          ? `<path d="${b.poly_d}" fill="${fill}" fill-opacity="${
              isSel ? "0.72" : "0.34"
            }" stroke="${isSel ? "#0f172a" : fill}" class="camels-basin-poly" />`
          : "";
        const selHalo = isSel
          ? `<circle cx="${b.map_x}" cy="${b.map_y}" r="12" fill="none" stroke="#0f172a" stroke-width="2.2" opacity="0.85" />
             <circle cx="${b.map_x}" cy="${b.map_y}" r="16.5" fill="none" stroke="#dc2626" stroke-width="1.6" stroke-dasharray="3,2" />`
          : "";

        return `
          <g
            class="camels-basin-group ${isSel ? "is-selected" : ""} ${
          isDimmed ? "is-dimmed" : ""
        }"
            data-map-basin-id="${esc(b.basin_id)}"
            data-map-basin-idx="${i}"
          >
            ${polyHtml}
            ${selHalo}
            <circle
              cx="${b.map_x}"
              cy="${b.map_y}"
              r="${r}"
              fill="${fill}"
              stroke="${isSel ? "#0f172a" : "#ffffff"}"
              stroke-width="${isSel ? "2.4" : "1.1"}"
              class="camels-basin-marker"
            />
          </g>
        `;
      })
      .join("");

    svg.innerHTML = `
      <g id="conus-states-layer">${statesSvg}${nationSvg}</g>
      <g id="conus-basins-layer">${basinsSvg}</g>
    `;
    applyConusMapViewBox();

    // Update Legend
    if (legend) {
      if (colorMode === "nse_primary" && m1Meta) {
        legend.innerHTML = `
          <span><strong>${esc(formatShortLeaderName(m1Meta))}</strong> Basin NSE:</span>
          <span><span class="tt-dot" style="background:#dc2626;display:inline-block;"></span> &le;0</span>
          <span><span class="tt-dot" style="background:#ea580c;display:inline-block;"></span> 0.2</span>
          <span><span class="tt-dot" style="background:#eab308;display:inline-block;"></span> 0.45</span>
          <span><span class="tt-dot" style="background:#10b981;display:inline-block;"></span> 0.65</span>
          <span><span class="tt-dot" style="background:#0284c7;display:inline-block;"></span> 0.75</span>
          <span><span class="tt-dot" style="background:#1e3a8a;display:inline-block;"></span> &ge;0.85</span>
          <span style="margin-left:0.4rem;color:#0f172a;">&#9678; Selected: <strong>${esc(
            STATE.hydroBasinId
          )}</strong></span>
        `;
      } else if (colorMode === "nse_diff" && m1Meta && m2Meta) {
        legend.innerHTML = `
          <span><strong>&Delta;NSE</strong> (${esc(formatShortLeaderName(m1Meta))} &minus; ${esc(
          formatShortLeaderName(m2Meta)
        )}):</span>
          <span><span class="tt-dot" style="background:#b91c1c;display:inline-block;"></span> &le;&minus;0.15</span>
          <span><span class="tt-dot" style="background:#f97316;display:inline-block;"></span> &minus;0.08</span>
          <span><span class="tt-dot" style="background:#cbd5e1;display:inline-block;"></span> &plusmn;0.05</span>
          <span><span class="tt-dot" style="background:#38bdf8;display:inline-block;"></span> +0.08</span>
          <span><span class="tt-dot" style="background:#1d4ed8;display:inline-block;"></span> &ge;+0.15</span>
        `;
      } else if (colorMode === "mean_qobs") {
        legend.innerHTML = `
          <span><strong>Mean Runoff Q<sub>obs</sub> (mm/d):</strong></span>
          <span><span class="tt-dot" style="background:#d97706;display:inline-block;"></span> &lt;0.3</span>
          <span><span class="tt-dot" style="background:#eab308;display:inline-block;"></span> 0.5</span>
          <span><span class="tt-dot" style="background:#10b981;display:inline-block;"></span> 1.2</span>
          <span><span class="tt-dot" style="background:#0284c7;display:inline-block;"></span> 2.5</span>
          <span><span class="tt-dot" style="background:#1e3a8a;display:inline-block;"></span> &ge;3.5</span>
        `;
      } else {
        legend.innerHTML = `
          <span><strong>18 USGS HUC-02 Regions</strong> &middot; 531 CAMELS-US Watersheds (Polygons + Gauges)</span>
          <span style="color:#0f172a;">&#9678; Selected: <strong>${esc(
            STATE.hydroBasinId
          )}</strong></span>
        `;
      }
    }

    // Bind click & hover on each basin group
    const tt = document.getElementById("hydro-map-tooltip");
    const stage = document.getElementById("hydro-conus-map-stage");
    svg.querySelectorAll("[data-map-basin-id]").forEach((gEl) => {
      const bid = gEl.getAttribute("data-map-basin-id");
      const bIdx = Number(gEl.getAttribute("data-map-basin-idx"));
      const bObj = basins[bIdx];

      gEl.addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (bid && bid !== STATE.hydroBasinId) {
          STATE.hydroBasinId = bid;
          const sel = document.getElementById("hydro-basin-select");
          if (sel) sel.value = bid;
          renderHydroConusMap();
          fetchAndRenderHydroBasin();
        }
      });

      gEl.addEventListener("mouseenter", (ev) => {
        if (!tt || !stage || !bObj) return;
        const modelRows = STATE.comparedModels
          .map((mk, mIdx) => {
            const meta = findModelMetadataByKey(mk);
            const arr = nseMap[mk];
            const nseVal = arr && arr[bIdx] != null ? Number(arr[bIdx]).toFixed(3) : "N/A";
            const color = HYDRO_MODEL_COLORS[mIdx % HYDRO_MODEL_COLORS.length];
            return `
              <div class="hydro-tt-row">
                <span class="tt-label"><span class="tt-dot" style="background:${color};"></span>${esc(
              formatShortLeaderName(meta || { model_name: mk })
            )}</span>
                <span class="tt-val">NSE ${esc(nseVal)}</span>
              </div>
            `;
          })
          .join("");

        tt.innerHTML = `
          <div class="hydro-tt-date">
            <span>USGS ${esc(bObj.basin_id)}</span>
            <span style="color:#94a3b8;font-weight:500;">HUC ${esc(bObj.huc_02)} (${esc(
          bObj.huc_name || ""
        )})</span>
          </div>
          <div style="font-weight:600;color:#f8fafc;margin-bottom:0.25rem;font-size:0.74rem;">
            ${esc(bObj.gauge_name || bObj.basin_id)}
          </div>
          <div class="hydro-tt-row">
            <span class="tt-label">Area / Elev</span>
            <span class="tt-val">${Number(bObj.area_km2 || 0).toFixed(0)} km&sup2; &middot; ${Number(
          bObj.elev_m || 0
        ).toFixed(0)} m</span>
          </div>
          <div class="hydro-tt-row">
            <span class="tt-label">Mean / Peak Q<sub>obs</sub></span>
            <span class="tt-val">${Number(bObj.mean_qobs || 0).toFixed(2)} / ${Number(
          bObj.max_qobs || 0
        ).toFixed(1)} mm/d</span>
          </div>
          ${modelRows ? `<div class="hydro-tt-section">${modelRows}</div>` : ""}
          <div style="margin-top:0.28rem;font-size:0.66rem;color:#38bdf8;font-family:var(--font-mono);">
            Click watershed to load hydrograph &darr;
          </div>
        `;
        tt.hidden = false;
      });

      gEl.addEventListener("mousemove", (ev) => {
        if (!tt || !stage || tt.hidden) return;
        const rect = stage.getBoundingClientRect();
        const relX = ev.clientX - rect.left;
        const relY = ev.clientY - rect.top;
        const ttW = tt.offsetWidth || 250;
        const ttH = tt.offsetHeight || 130;
        let left = relX + 14;
        if (left + ttW > rect.width - 8) left = Math.max(8, relX - ttW - 14);
        let top = relY - ttH / 2;
        top = Math.max(8, Math.min(rect.height - ttH - 8, top));
        tt.style.left = `${left}px`;
        tt.style.top = `${top}px`;
      });

      gEl.addEventListener("mouseleave", () => {
        if (tt) tt.hidden = true;
      });
    });
  }

  function initConusMapInteractions() {
    const svg = document.getElementById("hydro-conus-svg");
    if (!svg) return;

    const zoomMapBy = (factor, centerFracX = 0.5, centerFracY = 0.5) => {
      const vb = STATE.hydroMapViewBox || { x: 0, y: 0, w: 960, h: 580 };
      const newW = Math.max(140, Math.min(960, vb.w * factor));
      const newH = Math.max(85, Math.min(580, vb.h * factor));
      let newX = vb.x + (vb.w - newW) * centerFracX;
      let newY = vb.y + (vb.h - newH) * centerFracY;
      newX = Math.max(0, Math.min(960 - newW, newX));
      newY = Math.max(0, Math.min(580 - newH, newY));
      STATE.hydroMapViewBox = { x: newX, y: newY, w: newW, h: newH };
      applyConusMapViewBox();
    };

    svg.addEventListener(
      "wheel",
      (ev) => {
        ev.preventDefault();
        const rect = svg.getBoundingClientRect();
        const fx = Math.max(0, Math.min(1, (ev.clientX - rect.left) / Math.max(1, rect.width)));
        const fy = Math.max(0, Math.min(1, (ev.clientY - rect.top) / Math.max(1, rect.height)));
        zoomMapBy(ev.deltaY < 0 ? 0.8 : 1.25, fx, fy);
      },
      { passive: false }
    );

    let mapDrag = null;
    svg.addEventListener("mousedown", (ev) => {
      if (ev.target && ev.target.closest && ev.target.closest("[data-map-basin-id]")) return;
      const vb = STATE.hydroMapViewBox || { x: 0, y: 0, w: 960, h: 580 };
      mapDrag = {
        startX: ev.clientX,
        startY: ev.clientY,
        origX: vb.x,
        origY: vb.y,
        w: vb.w,
        h: vb.h,
      };
    });

    window.addEventListener("mousemove", (ev) => {
      if (!mapDrag) return;
      const rect = svg.getBoundingClientRect();
      const dx = ((ev.clientX - mapDrag.startX) / Math.max(1, rect.width)) * mapDrag.w;
      const dy = ((ev.clientY - mapDrag.startY) / Math.max(1, rect.height)) * mapDrag.h;
      const nx = Math.max(0, Math.min(960 - mapDrag.w, mapDrag.origX - dx));
      const ny = Math.max(0, Math.min(580 - mapDrag.h, mapDrag.origY - dy));
      STATE.hydroMapViewBox = { x: nx, y: ny, w: mapDrag.w, h: mapDrag.h };
      applyConusMapViewBox();
    });

    window.addEventListener("mouseup", () => {
      mapDrag = null;
    });

    const btnIn = document.getElementById("btn-map-zoom-in");
    const btnOut = document.getElementById("btn-map-zoom-out");
    const btnReset = document.getElementById("btn-map-zoom-reset");
    if (btnIn) btnIn.addEventListener("click", () => zoomMapBy(0.75));
    if (btnOut) btnOut.addEventListener("click", () => zoomMapBy(1.33));
    if (btnReset) {
      btnReset.addEventListener("click", () => {
        STATE.hydroMapViewBox = { x: 0, y: 0, w: 960, h: 580 };
        applyConusMapViewBox();
      });
    }

    const colorSel = document.getElementById("hydro-map-color-by");
    if (colorSel) {
      colorSel.addEventListener("change", (e) => {
        STATE.hydroMapColorBy = e.target.value || "nse_primary";
        renderHydroConusMap();
      });
    }

    const btnToggleMap = document.getElementById("btn-toggle-hydro-map");
    const mapWrap = document.getElementById("hydro-conus-map-wrap");
    if (btnToggleMap && mapWrap) {
      btnToggleMap.addEventListener("click", () => {
        STATE.hydroMapVisible = !STATE.hydroMapVisible;
        mapWrap.hidden = !STATE.hydroMapVisible;
        btnToggleMap.textContent = STATE.hydroMapVisible ? "Hide Map" : "Show Map";
      });
    }
  }

  function transformQVal(val, scaleMode) {
    const v = Math.max(0, Number(val) || 0);
    if (scaleMode === "sqrt") return Math.sqrt(v);
    if (scaleMode === "log") return Math.log10(v + 0.01);
    return v;
  }

  function renderHydrographCanvas() {
    const canvas = document.getElementById("hydro-multipanel-canvas");
    const payload = STATE.hydroBasinPayload;
    if (!canvas || !payload || !payload.dates) return;

    const rect = canvas.getBoundingClientRect();
    const cssW = Math.max(640, Math.floor(rect.width || 1100));
    const cssH = Math.max(580, Math.floor(rect.height || 700));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    if (canvas.width !== Math.floor(cssW * dpr) || canvas.height !== Math.floor(cssH * dpr)) {
      canvas.width = Math.floor(cssW * dpr);
      canvas.height = Math.floor(cssH * dpr);
    }

    const ctx = canvas.getContext("2d");
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cssW, cssH);

    // Background
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, cssW, cssH);

    const dates = payload.dates;
    const nDays = dates.length;
    const i0 = Math.max(0, Math.min(nDays - 2, Math.floor(STATE.hydroZoomStart)));
    const i1 = Math.max(i0 + 1, Math.min(nDays - 1, Math.ceil(STATE.hydroZoomEnd)));
    const visCount = i1 - i0 + 1;

    // Layout geometry for the 4 synchronized panels:
    // 1. Top Panel: Inverted Hyetographs (Precipitation mm/d for Daymet, Maurer, NLDAS)
    // 2. Middle Panel: Observed & Simulated Hydrographs (mm/d)
    // 3. Bottom Panel: Temperature Time Series (°C for Daymet, Maurer, NLDAS)
    // 4. Overview Brush Navigator (full 10-year 1989–1999)
    const padL = 68;
    const padR = 24;
    const plotW = cssW - padL - padR;

    const p1Top = 26;
    const p1H = 128;
    const p1Bot = p1Top + p1H;

    const p2Top = p1Bot + 24;
    const p2H = 264;
    const p2Bot = p2Top + p2H;

    const p3Top = p2Bot + 26;
    const p3H = 136;
    const p3Bot = p3Top + p3H;

    const navTop = p3Bot + 34;
    const navH = 54;
    const navBot = navTop + navH;

    HYDRO_CANVAS_GEOM = {
      cssW,
      cssH,
      padL,
      padR,
      plotW,
      p1Top,
      p1Bot,
      p2Top,
      p2Bot,
      p3Top,
      p3Bot,
      navTop,
      navBot,
      i0,
      i1,
      nDays,
    };

    const idxToX = (idx) => padL + ((idx - i0) / Math.max(1, i1 - i0)) * plotW;
    const activeForcingKeys = ["daymet", "maurer", "nldas"].filter(
      (k) => STATE.hydroForcingsOn[k] && payload.forcings && payload.forcings[k]
    );

    // -----------------------------------------------------------------------
    // Helper: Draw X-axis date gridlines across panels 1, 2, 3
    // -----------------------------------------------------------------------
    const tickIndices = [];
    const targetXTicks = Math.max(4, Math.min(10, Math.floor(plotW / 115)));
    const stepDays = Math.max(1, Math.round(visCount / targetXTicks));
    for (let idx = i0; idx <= i1; idx += stepDays) {
      tickIndices.push(idx);
    }
    if (tickIndices[tickIndices.length - 1] !== i1 && i1 - tickIndices[tickIndices.length - 1] > stepDays * 0.4) {
      tickIndices.push(i1);
    }

    ctx.strokeStyle = "#f1f5f9";
    ctx.lineWidth = 1;
    for (const tIdx of tickIndices) {
      const x = idxToX(tIdx);
      ctx.beginPath();
      ctx.moveTo(x, p1Top);
      ctx.lineTo(x, p1Bot);
      ctx.moveTo(x, p2Top);
      ctx.lineTo(x, p2Bot);
      ctx.moveTo(x, p3Top);
      ctx.lineTo(x, p3Bot);
      ctx.stroke();
    }

    // -----------------------------------------------------------------------
    // PANEL 1: Inverted Hyetographs (Daymet, Maurer, NLDAS Precipitation)
    // -----------------------------------------------------------------------
    let maxPrcp = 5.0;
    for (const fk of activeForcingKeys) {
      const pArr = payload.forcings[fk].prcp || [];
      for (let i = i0; i <= i1; i++) {
        if (pArr[i] > maxPrcp) maxPrcp = pArr[i];
      }
    }
    const prcpAxisMax = Math.ceil(maxPrcp * 1.12);
    const prcpToY = (v) => p1Top + (Math.max(0, v) / Math.max(1e-6, prcpAxisMax)) * p1H;

    const prcpTicks = computeNiceAxisTicks(0, prcpAxisMax, 4).filter((v) => v >= 0 && v <= prcpAxisMax);
    ctx.font = "500 10.5px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const tv of prcpTicks) {
      const y = prcpToY(tv);
      ctx.strokeStyle = "#f1f5f9";
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();
      ctx.fillStyle = "#64748b";
      ctx.fillText(tv.toFixed(0), padL - 8, y);
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(padL, p1Top, plotW, p1H);
    ctx.clip();

    const dayPxW = plotW / Math.max(1, visCount);
    const nForc = Math.max(1, activeForcingKeys.length);

    activeForcingKeys.forEach((fk, fIdx) => {
      const fMeta = HYDRO_FORCING_COLORS[fk];
      const pArr = payload.forcings[fk].prcp || [];
      if (visCount <= 420) {
        // Grouped / layered hanging bars for clear daily comparison
        const groupW = Math.max(1.5, Math.min(14, dayPxW * 0.84));
        const subW = Math.max(1.0, groupW / nForc);
        ctx.fillStyle = fMeta.fill;
        ctx.strokeStyle = fMeta.stroke;
        ctx.lineWidth = visCount <= 160 ? 1.0 : 0.7;
        for (let i = i0; i <= i1; i++) {
          const val = pArr[i] || 0;
          if (val <= 0.05) continue;
          const xCenter = idxToX(i);
          const xLeft = xCenter - groupW / 2 + fIdx * subW;
          const barH = Math.max(1, prcpToY(val) - p1Top);
          ctx.fillRect(xLeft, p1Top, subW, barH);
          if (visCount <= 220) {
            ctx.strokeRect(xLeft, p1Top, subW, barH);
          }
        }
      } else {
        // High-density hanging impulse stems when viewing multi-year windows
        ctx.strokeStyle = fMeta.stroke;
        ctx.lineWidth = 1.1;
        ctx.beginPath();
        for (let i = i0; i <= i1; i++) {
          const val = pArr[i] || 0;
          if (val <= 0.2) continue;
          const x = idxToX(i);
          ctx.moveTo(x, p1Top);
          ctx.lineTo(x, prcpToY(val));
        }
        ctx.stroke();
      }
    });
    ctx.restore();

    // Panel 1 Frame & Title
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, p1Top, plotW, p1H);

    ctx.fillStyle = "#0f172a";
    ctx.font = "600 11px 'IBM Plex Sans', sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText("1. CAMELS-US Hyetographs — Precipitation (mm/d, inverted)", padL + 2, p1Top - 5);

    // Y-axis label Panel 1
    ctx.save();
    ctx.translate(16, p1Top + p1H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = "#475569";
    ctx.font = "600 10px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText("Prcp (mm/d)", 0, 0);
    ctx.restore();

    // -----------------------------------------------------------------------
    // PANEL 2: Observed & Simulated Hydrographs (mm/d)
    // -----------------------------------------------------------------------
    const qScale = STATE.hydroQScale || "linear";
    const qobs = payload.qobs || [];
    const seriesList = payload.series || [];

    let maxQRaw = 1.0;
    for (let i = i0; i <= i1; i++) {
      if (qobs[i] != null && qobs[i] > maxQRaw) maxQRaw = qobs[i];
    }
    for (const s of seriesList) {
      const qsim = s.qsim || [];
      for (let i = i0; i <= i1; i++) {
        if (qsim[i] != null && qsim[i] > maxQRaw) maxQRaw = qsim[i];
      }
    }

    const qDomMin = qScale === "log" ? transformQVal(0, "log") : 0;
    const qDomMax = transformQVal(maxQRaw * 1.1, qScale);
    const qToY = (rawVal) => {
      const tv = transformQVal(rawVal, qScale);
      return p2Bot - ((tv - qDomMin) / Math.max(1e-6, qDomMax - qDomMin)) * p2H;
    };

    // Y-Axis Ticks for Panel 2
    let qRawTicks = [];
    if (qScale === "log") {
      qRawTicks = [0, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 200].filter((v) => v <= maxQRaw * 1.15);
    } else if (qScale === "sqrt") {
      const nice = computeNiceAxisTicks(0, Math.sqrt(maxQRaw * 1.1), 5);
      qRawTicks = nice.map((v) => Number((v * v).toFixed(2)));
    } else {
      qRawTicks = computeNiceAxisTicks(0, maxQRaw * 1.08, 6).filter((v) => v >= 0);
    }

    ctx.font = "500 10.5px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const tv of qRawTicks) {
      const y = qToY(tv);
      if (y < p2Top - 2 || y > p2Bot + 2) continue;
      ctx.strokeStyle = "#f1f5f9";
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();
      ctx.fillStyle = "#64748b";
      ctx.fillText(tv >= 10 ? tv.toFixed(0) : tv.toFixed(1), padL - 8, y);
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(padL, p2Top, plotW, p2H);
    ctx.clip();

    // Draw Observed Qobs area fill + bold dark trace
    ctx.beginPath();
    let startedObs = false;
    let firstObsX = padL;
    let lastObsX = padL;
    for (let i = i0; i <= i1; i++) {
      const v = qobs[i];
      if (v === null || v === undefined) {
        startedObs = false;
        continue;
      }
      const x = idxToX(i);
      const y = qToY(v);
      if (!startedObs) {
        ctx.moveTo(x, y);
        firstObsX = x;
        startedObs = true;
      } else {
        ctx.lineTo(x, y);
      }
      lastObsX = x;
    }
    ctx.strokeStyle = "#0f172a";
    ctx.lineWidth = visCount <= 400 ? 2.3 : 1.6;
    ctx.stroke();

    // Subtle fill under Qobs
    if (startedObs) {
      ctx.lineTo(lastObsX, p2Bot);
      ctx.lineTo(firstObsX, p2Bot);
      ctx.closePath();
      ctx.fillStyle = "rgba(15, 23, 42, 0.06)";
      ctx.fill();
    }

    // Draw each selected model's simulated hydrograph Qsim
    seriesList.forEach((s, sIdx) => {
      const color = HYDRO_MODEL_COLORS[sIdx % HYDRO_MODEL_COLORS.length];
      const qsim = s.qsim || [];
      ctx.beginPath();
      let started = false;
      for (let i = i0; i <= i1; i++) {
        const v = qsim[i];
        if (v === null || v === undefined) {
          started = false;
          continue;
        }
        const x = idxToX(i);
        const y = qToY(v);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = visCount <= 400 ? 1.95 : 1.35;
      ctx.stroke();
    });

    ctx.restore();

    // Panel 2 Frame & Title
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, p2Top, plotW, p2H);

    ctx.fillStyle = "#0f172a";
    ctx.font = "600 11px 'IBM Plex Sans', sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    const scaleLabel =
      qScale === "sqrt" ? "√mm/d scale" : qScale === "log" ? "log10 scale" : "mm/d";
    ctx.fillText(
      `2. Observed & Simulated Daily Streamflow Hydrographs (${scaleLabel})`,
      padL + 2,
      p2Top - 5
    );

    ctx.save();
    ctx.translate(16, p2Top + p2H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = "#475569";
    ctx.font = "600 10px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText("Q (mm/d)", 0, 0);
    ctx.restore();

    // -----------------------------------------------------------------------
    // PANEL 3: Temperature Time Series (°C) for All 3 CAMELS-US Forcings
    // -----------------------------------------------------------------------
    let tMinVal = -5.0;
    let tMaxVal = 25.0;
    if (activeForcingKeys.length > 0) {
      tMinVal = Infinity;
      tMaxVal = -Infinity;
      for (const fk of activeForcingKeys) {
        const tminArr = payload.forcings[fk].tmin || [];
        const tmaxArr = payload.forcings[fk].tmax || [];
        for (let i = i0; i <= i1; i++) {
          if (tminArr[i] < tMinVal) tMinVal = tminArr[i];
          if (tmaxArr[i] > tMaxVal) tMaxVal = tmaxArr[i];
        }
      }
      if (!Number.isFinite(tMinVal) || !Number.isFinite(tMaxVal)) {
        tMinVal = -10;
        tMaxVal = 30;
      }
    }
    const tPad = Math.max(2.0, (tMaxVal - tMinVal) * 0.1);
    const tDomMin = Math.floor(tMinVal - tPad);
    const tDomMax = Math.ceil(tMaxVal + tPad);
    const tempToY = (t) =>
      p3Bot - ((t - tDomMin) / Math.max(1e-6, tDomMax - tDomMin)) * p3H;

    const tempTicks = computeNiceAxisTicks(tDomMin, tDomMax, 5);
    ctx.font = "500 10.5px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const tv of tempTicks) {
      const y = tempToY(tv);
      if (y < p3Top - 2 || y > p3Bot + 2) continue;
      ctx.strokeStyle = "#f1f5f9";
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();
      ctx.fillStyle = "#64748b";
      ctx.fillText(`${tv.toFixed(0)}°`, padL - 8, y);
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(padL, p3Top, plotW, p3H);
    ctx.clip();

    // 0°C Freezing Threshold Reference Line
    if (tDomMin <= 0 && tDomMax >= 0) {
      const yZero = tempToY(0);
      ctx.strokeStyle = "#94a3b8";
      ctx.lineWidth = 1.1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(padL, yZero);
      ctx.lineTo(padL + plotW, yZero);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = "#64748b";
      ctx.font = "600 9.5px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText("0°C freezing", padL + plotW - 6, yZero - 6);
    }

    const tempMode = STATE.hydroTempMode || "tmean_band";
    for (const fk of activeForcingKeys) {
      const fMeta = HYDRO_FORCING_COLORS[fk];
      const fObj = payload.forcings[fk];
      const tminArr = fObj.tmin || [];
      const tmaxArr = fObj.tmax || [];
      const tmeanArr = fObj.tmean || [];

      if (tempMode === "tmean_band" && visCount <= 900) {
        // Shaded diurnal range [Tmin, Tmax]
        ctx.beginPath();
        for (let i = i0; i <= i1; i++) {
          const x = idxToX(i);
          const y = tempToY(tmaxArr[i]);
          if (i === i0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        for (let i = i1; i >= i0; i--) {
          const x = idxToX(i);
          const y = tempToY(tminArr[i]);
          ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fillStyle = fMeta.band;
        ctx.fill();
      }

      if (tempMode === "tmax_tmin") {
        // Solid Tmax line + dashed Tmin line
        ctx.strokeStyle = fMeta.stroke;
        ctx.lineWidth = 1.35;
        ctx.beginPath();
        for (let i = i0; i <= i1; i++) {
          const x = idxToX(i);
          const y = tempToY(tmaxArr[i]);
          if (i === i0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();

        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        for (let i = i0; i <= i1; i++) {
          const x = idxToX(i);
          const y = tempToY(tminArr[i]);
          if (i === i0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      } else {
        // Daily Mean Temperature Tmean line
        ctx.strokeStyle = fMeta.stroke;
        ctx.lineWidth = visCount <= 400 ? 1.65 : 1.2;
        ctx.beginPath();
        for (let i = i0; i <= i1; i++) {
          const x = idxToX(i);
          const y = tempToY(tmeanArr[i]);
          if (i === i0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
    }

    ctx.restore();

    // Panel 3 Frame & Title
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, p3Top, plotW, p3H);

    ctx.fillStyle = "#0f172a";
    ctx.font = "600 11px 'IBM Plex Sans', sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText(
      "3. CAMELS-US Daily Temperature — Daymet, Maurer & NLDAS (°C)",
      padL + 2,
      p3Top - 5
    );

    ctx.save();
    ctx.translate(16, p3Top + p3H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = "#475569";
    ctx.font = "600 10px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText("Temp (°C)", 0, 0);
    ctx.restore();

    // Date labels below Panel 3
    ctx.fillStyle = "#334155";
    ctx.font = "500 10.5px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const tIdx of tickIndices) {
      const x = idxToX(tIdx);
      const dStr = dates[tIdx] || "";
      ctx.fillText(visCount <= 500 ? dStr : dStr.slice(0, 7), x, p3Bot + 6);
    }

    // -----------------------------------------------------------------------
    // PANEL 4: 10-Year Overview & Interactive Zoom Brush Navigator
    // -----------------------------------------------------------------------
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(padL, navTop, plotW, navH);

    let fullMaxQ = 1.0;
    for (let i = 0; i < nDays; i++) {
      if (qobs[i] != null && qobs[i] > fullMaxQ) fullMaxQ = qobs[i];
    }
    const navX = (idx) => padL + (idx / Math.max(1, nDays - 1)) * plotW;
    const navY = (v) =>
      navBot - 4 - (Math.max(0, v || 0) / Math.max(1e-6, fullMaxQ * 1.08)) * (navH - 10);

    // Full 10-year Qobs mini trace
    ctx.beginPath();
    for (let i = 0; i < nDays; i++) {
      const x = navX(i);
      const y = navY(qobs[i] || 0);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 1.0;
    ctx.stroke();

    // Year dividers inside Navigator
    ctx.font = "500 9.5px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#64748b";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    for (let yr = 1990; yr <= 1999; yr++) {
      const dIdx = dates.indexOf(`${yr}-01-01`);
      if (dIdx >= 0) {
        const x = navX(dIdx);
        ctx.strokeStyle = "#e2e8f0";
        ctx.beginPath();
        ctx.moveTo(x, navTop);
        ctx.lineTo(x, navBot);
        ctx.stroke();
        ctx.fillText(String(yr), x + 3, navTop + 2);
      }
    }

    // Active Zoom Brush Window
    const bx0 = navX(i0);
    const bx1 = Math.max(bx0 + 6, navX(i1));

    // Dim outside active brush
    ctx.fillStyle = "rgba(148, 163, 184, 0.25)";
    ctx.fillRect(padL, navTop, Math.max(0, bx0 - padL), navH);
    ctx.fillRect(bx1, navTop, Math.max(0, padL + plotW - bx1), navH);

    // Highlight active brush box
    const isPanActive =
      STATE.hydroNavHoverZone === "pan" ||
      (HYDRO_DRAG_STATE && HYDRO_DRAG_STATE.mode === "nav-pan");
    ctx.fillStyle = isPanActive ? "rgba(15, 118, 110, 0.22)" : "rgba(15, 118, 110, 0.14)";
    ctx.fillRect(bx0, navTop, bx1 - bx0, navH);
    ctx.strokeStyle = "#0f766e";
    ctx.lineWidth = isPanActive ? 2.0 : 1.6;
    ctx.strokeRect(bx0, navTop, bx1 - bx0, navH);

    // Left & right tactile grab handles with grip lines and hover highlight
    const drawNavGrabHandle = (cx, isActive) => {
      const w = isActive ? 10 : 8;
      const h = isActive ? 32 : 28;
      const rx = cx - w / 2;
      const ry = navTop + (navH - h) / 2;

      if (isActive) {
        ctx.fillStyle = "rgba(15, 118, 110, 0.22)";
        ctx.beginPath();
        ctx.roundRect(rx - 3, ry - 3, w + 6, h + 6, 5);
        ctx.fill();
      }

      ctx.fillStyle = isActive ? "#0d9488" : "#0f766e";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.roundRect(rx, ry, w, h, 3.5);
      ctx.fill();
      ctx.stroke();

      // Grip lines inside handle
      ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx - 1.3, ry + 7);
      ctx.lineTo(cx - 1.3, ry + h - 7);
      ctx.moveTo(cx + 1.3, ry + 7);
      ctx.lineTo(cx + 1.3, ry + h - 7);
      ctx.stroke();
    };

    const leftActive =
      STATE.hydroNavHoverZone === "left" ||
      (HYDRO_DRAG_STATE && HYDRO_DRAG_STATE.mode === "nav-left");
    const rightActive =
      STATE.hydroNavHoverZone === "right" ||
      (HYDRO_DRAG_STATE && HYDRO_DRAG_STATE.mode === "nav-right");
    drawNavGrabHandle(bx0, leftActive);
    drawNavGrabHandle(bx1, rightActive);

    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, navTop, plotW, navH);

    // -----------------------------------------------------------------------
    // Drag-Zoom Selection Box Overlay (when dragging on Panels 1–3)
    // -----------------------------------------------------------------------
    if (HYDRO_DRAG_STATE && HYDRO_DRAG_STATE.mode === "box") {
      const sx = Math.max(padL, Math.min(padL + plotW, HYDRO_DRAG_STATE.startX));
      const cx = Math.max(padL, Math.min(padL + plotW, HYDRO_DRAG_STATE.currX));
      const left = Math.min(sx, cx);
      const w = Math.abs(cx - sx);
      if (w > 3) {
        ctx.fillStyle = "rgba(37, 99, 235, 0.14)";
        ctx.strokeStyle = "#2563eb";
        ctx.lineWidth = 1.3;
        for (const [top, h] of [
          [p1Top, p1H],
          [p2Top, p2H],
          [p3Top, p3H],
        ]) {
          ctx.fillRect(left, top, w, h);
          ctx.strokeRect(left, top, w, h);
        }
      }
    }

    // -----------------------------------------------------------------------
    // Synchronized Hover Crosshair across Panels 1, 2, 3
    // -----------------------------------------------------------------------
    if (
      STATE.hydroHoverDayIdx !== null &&
      STATE.hydroHoverDayIdx >= i0 &&
      STATE.hydroHoverDayIdx <= i1
    ) {
      const hIdx = STATE.hydroHoverDayIdx;
      const hx = idxToX(hIdx);
      ctx.strokeStyle = "rgba(15, 23, 42, 0.55)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(hx, p1Top);
      ctx.lineTo(hx, p1Bot);
      ctx.moveTo(hx, p2Top);
      ctx.lineTo(hx, p2Bot);
      ctx.moveTo(hx, p3Top);
      ctx.lineTo(hx, p3Bot);
      ctx.stroke();
      ctx.setLineDash([]);

      // Highlight dots on Panel 2 (Qobs & Qsim)
      if (qobs[hIdx] != null) {
        ctx.fillStyle = "#0f172a";
        ctx.beginPath();
        ctx.arc(hx, qToY(qobs[hIdx]), 4, 0, Math.PI * 2);
        ctx.fill();
      }
      seriesList.forEach((s, sIdx) => {
        const v = s.qsim ? s.qsim[hIdx] : null;
        if (v != null) {
          ctx.fillStyle = HYDRO_MODEL_COLORS[sIdx % HYDRO_MODEL_COLORS.length];
          ctx.beginPath();
          ctx.arc(hx, qToY(v), 3.6, 0, Math.PI * 2);
          ctx.fill();
        }
      });
    }

    ctx.restore();
  }

  function updateHydroHoverTooltip(dayIdx, clientX, clientY) {
    const tt = document.getElementById("hydro-tooltip");
    const stage = document.getElementById("hydro-canvas-stage");
    const payload = STATE.hydroBasinPayload;
    if (!tt || !stage || !payload || dayIdx === null || dayIdx < 0) {
      if (tt) tt.hidden = true;
      return;
    }

    const dateStr = payload.dates[dayIdx] || "";
    const qobsVal = payload.qobs ? payload.qobs[dayIdx] : null;
    const qobsStr = qobsVal != null ? `${Number(qobsVal).toFixed(2)} mm/d` : "N/A";

    const forcingRows = ["daymet", "maurer", "nldas"]
      .filter((fk) => STATE.hydroForcingsOn[fk] && payload.forcings && payload.forcings[fk])
      .map((fk) => {
        const fMeta = HYDRO_FORCING_COLORS[fk];
        const fObj = payload.forcings[fk];
        const prcp = fObj.prcp[dayIdx] != null ? Number(fObj.prcp[dayIdx]).toFixed(1) : "—";
        const tmean = fObj.tmean[dayIdx] != null ? Number(fObj.tmean[dayIdx]).toFixed(1) : "—";
        const tmin = fObj.tmin[dayIdx] != null ? Number(fObj.tmin[dayIdx]).toFixed(1) : "—";
        const tmax = fObj.tmax[dayIdx] != null ? Number(fObj.tmax[dayIdx]).toFixed(1) : "—";
        return `
          <div class="hydro-tt-row">
            <span class="tt-label"><span class="tt-dot" style="background:${fMeta.stroke};"></span>${esc(
          fMeta.label
        )}</span>
            <span class="tt-val">${prcp} mm &middot; ${tmean}&deg;C (${tmin}..${tmax}&deg;)</span>
          </div>
        `;
      })
      .join("");

    const modelRows = (payload.series || [])
      .map((s, idx) => {
        const color = HYDRO_MODEL_COLORS[idx % HYDRO_MODEL_COLORS.length];
        const v = s.qsim ? s.qsim[dayIdx] : null;
        const vStr = v != null ? `${Number(v).toFixed(2)} mm/d` : "N/A";
        return `
          <div class="hydro-tt-row">
            <span class="tt-label"><span class="tt-dot" style="background:${color};"></span>${esc(
          formatShortLeaderName(s)
        )}</span>
            <span class="tt-val">${vStr}</span>
          </div>
        `;
      })
      .join("");

    tt.innerHTML = `
      <div class="hydro-tt-date">
        <span>${esc(dateStr)}</span>
        <span style="color:#94a3b8; font-weight:500;">Basin ${esc(payload.basin.basin_id)}</span>
      </div>
      <div class="hydro-tt-row">
        <span class="tt-label"><span class="tt-dot" style="background:#f8fafc;"></span>Observed Q<sub>obs</sub></span>
        <span class="tt-val">${qobsStr}</span>
      </div>
      ${modelRows}
      ${
        forcingRows
          ? `<div class="hydro-tt-section">${forcingRows}</div>`
          : ""
      }
    `;
    tt.hidden = false;

    const rect = stage.getBoundingClientRect();
    const relX = clientX - rect.left;
    const relY = clientY - rect.top;
    const ttW = tt.offsetWidth || 260;
    const ttH = tt.offsetHeight || 160;
    let left = relX + 16;
    if (left + ttW > rect.width - 10) {
      left = Math.max(10, relX - ttW - 16);
    }
    let top = relY - ttH / 2;
    top = Math.max(10, Math.min(rect.height - ttH - 10, top));
    tt.style.left = `${left}px`;
    tt.style.top = `${top}px`;
  }

  function initHydrographCanvasInteractions() {
    const canvas = document.getElementById("hydro-multipanel-canvas");
    if (!canvas) return;

    const getRelPos = (ev) => {
      const r = canvas.getBoundingClientRect();
      return { x: ev.clientX - r.left, y: ev.clientY - r.top };
    };

    canvas.addEventListener("mousedown", (ev) => {
      if (!HYDRO_CANVAS_GEOM) return;
      const { x, y } = getRelPos(ev);
      const g = HYDRO_CANVAS_GEOM;
      if (x < g.padL || x > g.padL + g.plotW) return;

      if (y >= g.navTop && y <= g.navBot) {
        const bx0 = g.padL + (g.i0 / Math.max(1, g.nDays - 1)) * g.plotW;
        const bx1 = Math.max(bx0 + 6, g.padL + (g.i1 / Math.max(1, g.nDays - 1)) * g.plotW);
        if (Math.abs(x - bx0) <= 9) {
          HYDRO_DRAG_STATE = { mode: "nav-left", startX: x, origI0: g.i0, origI1: g.i1 };
          canvas.style.cursor = "ew-resize";
        } else if (Math.abs(x - bx1) <= 9) {
          HYDRO_DRAG_STATE = { mode: "nav-right", startX: x, origI0: g.i0, origI1: g.i1 };
          canvas.style.cursor = "ew-resize";
        } else if (x >= bx0 && x <= bx1) {
          HYDRO_DRAG_STATE = { mode: "nav-pan", startX: x, origI0: g.i0, origI1: g.i1 };
          canvas.style.cursor = "grabbing";
        } else {
          // Click outside brush jumps center of brush to clicked day
          const clickDay = Math.round(((x - g.padL) / g.plotW) * (g.nDays - 1));
          const halfSpan = Math.round((g.i1 - g.i0) / 2);
          let newI0 = Math.max(0, clickDay - halfSpan);
          let newI1 = Math.min(g.nDays - 1, newI0 + (g.i1 - g.i0));
          newI0 = Math.max(0, newI1 - (g.i1 - g.i0));
          STATE.hydroZoomStart = newI0;
          STATE.hydroZoomEnd = newI1;
          STATE.hydroActivePreset = "custom";
          syncHydroZoomButtonsAndDates();
          renderHydrographCanvas();
          HYDRO_DRAG_STATE = { mode: "nav-pan", startX: x, origI0: newI0, origI1: newI1 };
          canvas.style.cursor = "grabbing";
        }
      } else if (y >= g.p1Top && y <= g.p3Bot) {
        HYDRO_DRAG_STATE = { mode: "box", startX: x, currX: x };
        canvas.style.cursor = "crosshair";
      }
    });

    window.addEventListener("mousemove", (ev) => {
      if (!HYDRO_CANVAS_GEOM) return;
      const g = HYDRO_CANVAS_GEOM;
      const { x, y } = getRelPos(ev);

      if (HYDRO_DRAG_STATE) {
        if (HYDRO_DRAG_STATE.mode === "box") {
          canvas.style.cursor = "crosshair";
          HYDRO_DRAG_STATE.currX = x;
          renderHydrographCanvas();
        } else {
          if (HYDRO_DRAG_STATE.mode === "nav-pan") {
            canvas.style.cursor = "grabbing";
          } else {
            canvas.style.cursor = "ew-resize";
          }
          const deltaDays = Math.round(((x - HYDRO_DRAG_STATE.startX) / g.plotW) * (g.nDays - 1));
          const minSpan = 14;
          if (HYDRO_DRAG_STATE.mode === "nav-pan") {
            const span = HYDRO_DRAG_STATE.origI1 - HYDRO_DRAG_STATE.origI0;
            let s = Math.max(0, Math.min(g.nDays - 1 - span, HYDRO_DRAG_STATE.origI0 + deltaDays));
            STATE.hydroZoomStart = s;
            STATE.hydroZoomEnd = s + span;
          } else if (HYDRO_DRAG_STATE.mode === "nav-left") {
            STATE.hydroZoomStart = Math.max(
              0,
              Math.min(HYDRO_DRAG_STATE.origI1 - minSpan, HYDRO_DRAG_STATE.origI0 + deltaDays)
            );
          } else if (HYDRO_DRAG_STATE.mode === "nav-right") {
            STATE.hydroZoomEnd = Math.min(
              g.nDays - 1,
              Math.max(HYDRO_DRAG_STATE.origI0 + minSpan, HYDRO_DRAG_STATE.origI1 + deltaDays)
            );
          }
          STATE.hydroActivePreset = "custom";
          syncHydroZoomButtonsAndDates();
          renderHydrographCanvas();
        }
        return;
      }

      // Check hover zone over Panel 4 Navigator grab handles & brush window
      let nextNavZone = null;
      if (x >= g.padL && x <= g.padL + g.plotW && y >= g.navTop - 4 && y <= g.navBot + 4) {
        const bx0 = g.padL + (g.i0 / Math.max(1, g.nDays - 1)) * g.plotW;
        const bx1 = Math.max(bx0 + 6, g.padL + (g.i1 / Math.max(1, g.nDays - 1)) * g.plotW);
        if (Math.abs(x - bx0) <= 9) {
          nextNavZone = "left";
          canvas.style.cursor = "ew-resize";
        } else if (Math.abs(x - bx1) <= 9) {
          nextNavZone = "right";
          canvas.style.cursor = "ew-resize";
        } else if (x > bx0 && x < bx1) {
          nextNavZone = "pan";
          canvas.style.cursor = "grab";
        } else {
          canvas.style.cursor = "pointer";
        }
      } else {
        canvas.style.cursor = "crosshair";
      }

      const navZoneChanged = nextNavZone !== STATE.hydroNavHoverZone;
      STATE.hydroNavHoverZone = nextNavZone;

      // Hover crosshair inside Panels 1–3
      if (x >= g.padL && x <= g.padL + g.plotW && y >= g.p1Top && y <= g.p3Bot) {
        const frac = (x - g.padL) / Math.max(1, g.plotW);
        const dayIdx = Math.max(g.i0, Math.min(g.i1, Math.round(g.i0 + frac * (g.i1 - g.i0))));
        STATE.hydroHoverDayIdx = dayIdx;
        renderHydrographCanvas();
        updateHydroHoverTooltip(dayIdx, ev.clientX, ev.clientY);
      } else if (STATE.hydroHoverDayIdx !== null || navZoneChanged) {
        STATE.hydroHoverDayIdx = null;
        renderHydrographCanvas();
        updateHydroHoverTooltip(null, 0, 0);
      }
    });

    window.addEventListener("mouseup", () => {
      if (!HYDRO_DRAG_STATE || !HYDRO_CANVAS_GEOM) return;
      const g = HYDRO_CANVAS_GEOM;
      if (HYDRO_DRAG_STATE.mode === "box") {
        const sx = Math.max(g.padL, Math.min(g.padL + g.plotW, HYDRO_DRAG_STATE.startX));
        const cx = Math.max(g.padL, Math.min(g.padL + g.plotW, HYDRO_DRAG_STATE.currX));
        if (Math.abs(cx - sx) > 6) {
          const leftFrac = (Math.min(sx, cx) - g.padL) / g.plotW;
          const rightFrac = (Math.max(sx, cx) - g.padL) / g.plotW;
          const newI0 = Math.round(g.i0 + leftFrac * (g.i1 - g.i0));
          const newI1 = Math.max(newI0 + 7, Math.round(g.i0 + rightFrac * (g.i1 - g.i0)));
          STATE.hydroZoomStart = Math.max(0, newI0);
          STATE.hydroZoomEnd = Math.min(g.nDays - 1, newI1);
          STATE.hydroActivePreset = "custom";
          syncHydroZoomButtonsAndDates();
        }
      }
      HYDRO_DRAG_STATE = null;
      if (STATE.hydroNavHoverZone === "left" || STATE.hydroNavHoverZone === "right") {
        canvas.style.cursor = "ew-resize";
      } else if (STATE.hydroNavHoverZone === "pan") {
        canvas.style.cursor = "grab";
      } else {
        canvas.style.cursor = "crosshair";
      }
      renderHydrographCanvas();
    });

    canvas.addEventListener("mouseleave", () => {
      if (!HYDRO_DRAG_STATE) {
        STATE.hydroHoverDayIdx = null;
        STATE.hydroNavHoverZone = null;
        canvas.style.cursor = "crosshair";
        renderHydrographCanvas();
        updateHydroHoverTooltip(null, 0, 0);
      }
    });

    // Mouse-wheel zoom centered on cursor day
    canvas.addEventListener(
      "wheel",
      (ev) => {
        if (!HYDRO_CANVAS_GEOM) return;
        const g = HYDRO_CANVAS_GEOM;
        const { x, y } = getRelPos(ev);
        if (x < g.padL || x > g.padL + g.plotW || y < g.p1Top || y > g.p3Bot) return;
        ev.preventDefault();

        const frac = (x - g.padL) / g.plotW;
        const anchorDay = g.i0 + frac * (g.i1 - g.i0);
        const curSpan = g.i1 - g.i0;
        const zoomFactor = ev.deltaY < 0 ? 0.78 : 1.28;
        const newSpan = Math.max(14, Math.min(g.nDays - 1, Math.round(curSpan * zoomFactor)));

        let newI0 = Math.round(anchorDay - frac * newSpan);
        let newI1 = newI0 + newSpan;
        if (newI0 < 0) {
          newI0 = 0;
          newI1 = Math.min(g.nDays - 1, newSpan);
        }
        if (newI1 > g.nDays - 1) {
          newI1 = g.nDays - 1;
          newI0 = Math.max(0, newI1 - newSpan);
        }
        STATE.hydroZoomStart = newI0;
        STATE.hydroZoomEnd = newI1;
        STATE.hydroActivePreset = newSpan >= g.nDays - 2 ? "full" : "custom";
        syncHydroZoomButtonsAndDates();
        renderHydrographCanvas();
      },
      { passive: false }
    );

    // Double-click toggles between Full 10 Years and Peak Flood (±45d)
    canvas.addEventListener("dblclick", () => {
      if (STATE.hydroActivePreset === "full") {
        applyHydroZoomPreset("peak1");
      } else {
        applyHydroZoomPreset("full");
      }
    });

    window.addEventListener("resize", () => {
      if (STATE.activeView === "hydrographs") {
        renderHydrographCanvas();
      }
    });
  }

  // =========================================================================
  // 8. Main Page Tab Switching & Event Bindings
  // =========================================================================
  const VALID_VIEWS = new Set([
    "landing",
    "about",
    "about-protocol",
    "about-camels-us",
    "contribute",
    "leaderboards-hub",
    "leaderboards",
    "hydrographs",
    "models",
    "publications",
  ]);

  function persistNavState() {
    const hashTarget =
      STATE.activeView === "leaderboards"
        ? `#leaderboards/${STATE.activeSpatialTab}__${STATE.activeForcingTab}`
        : `#${STATE.activeView}`;
    if (window.location.hash !== hashTarget) {
      history.replaceState(null, "", hashTarget);
    }
    try {
      sessionStorage.setItem(
        "hmip_nav_state",
        JSON.stringify({
          activeView: STATE.activeView,
          activeSpatialTab: STATE.activeSpatialTab,
          activeForcingTab: STATE.activeForcingTab,
          metricPreset: STATE.metricPreset,
        })
      );
    } catch (_) {
      // Ignore storage quota errors
    }
  }

  function restoreNavStateFromUrlOrStorage() {
    let view = null;
    let sp = null;
    let fo = null;

    const rawHash = (window.location.hash || "").replace(/^#/, "").trim();
    if (rawHash) {
      const [hView, hSub] = rawHash.split("/");
      if (VALID_VIEWS.has(hView)) {
        view = hView;
        if (hSub && hSub.includes("__")) {
          const [hSp, hFo] = hSub.split("__");
          if (hSp && hFo) {
            sp = hSp;
            fo = hFo;
          }
        }
      }
    }

    if (!view) {
      try {
        const saved = JSON.parse(sessionStorage.getItem("hmip_nav_state") || "null");
        if (saved && VALID_VIEWS.has(saved.activeView)) {
          view = saved.activeView;
          sp = saved.activeSpatialTab || null;
          fo = saved.activeForcingTab || null;
          if (saved.metricPreset && METRIC_PRESETS[saved.metricPreset]) {
            STATE.metricPreset = saved.metricPreset;
            STATE.visibleMetricCols = new Set(METRIC_PRESETS[saved.metricPreset].cols);
          }
        }
      } catch (_) {
        // Ignore parse errors
      }
    }

    if (sp) STATE.activeSpatialTab = sp;
    if (fo) STATE.activeForcingTab = fo;
    if (view) {
      switchMainTab(view, { skipScroll: true });
    }
  }

  function switchMainTab(viewName, opts = {}) {
    if (!VALID_VIEWS.has(viewName)) viewName = "landing";
    STATE.activeView = viewName;
    const views = Array.from(VALID_VIEWS);
    for (const v of views) {
      const sec = document.getElementById(`view-${v}`);
      if (sec) sec.style.display = v === viewName ? "block" : "none";
    }

    const navMap = {
      landing: "nav-tab-landing",
      about: "nav-tab-about",
      "about-protocol": "nav-tab-about",
      "about-camels-us": "nav-tab-about",
      contribute: "nav-tab-contribute",
      "leaderboards-hub": "nav-tab-leaderboards-hub",
      leaderboards: "nav-tab-leaderboards-hub",
      hydrographs: "nav-tab-leaderboards-hub",
      models: "nav-tab-leaderboards-hub",
      publications: "nav-tab-publications",
    };
    const activeNavId = navMap[viewName] || "nav-tab-landing";
    document.querySelectorAll(".nav-tab-btn").forEach((btn) => {
      const isAct = btn.id === activeNavId;
      btn.classList.toggle("active", isAct);
      btn.setAttribute("aria-selected", isAct ? "true" : "false");
    });

    updateCompareUiState();
    persistNavState();
    if (!opts.skipScroll) {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }

  function initEventListeners() {
    // Top Header Page Tabs
    document.querySelectorAll(".nav-tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        switchMainTab(btn.getAttribute("data-view"));
      });
    });

    const brandHome = document.getElementById("brand-home-link");
    if (brandHome) {
      brandHome.addEventListener("click", (e) => {
        e.preventDefault();
        switchMainTab("landing");
      });
    }

    // Social Media Copy Link button
    const btnCopyShare = document.getElementById("btn-copy-share-link");
    if (btnCopyShare) {
      btnCopyShare.addEventListener("click", async () => {
        const shareUrl = window.location.href || "https://hydrologymip.org/";
        const lbl = document.getElementById("copy-share-link-label");
        try {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(shareUrl);
          }
          if (lbl) lbl.textContent = "Link Copied!";
          setTimeout(() => {
            if (lbl) lbl.textContent = "Copy Link";
          }, 2200);
        } catch (e) {
          console.error(e);
        }
      });
    }

    // CTA buttons across Landing, Header, Footer, and CAMELS-US Leaderboards pages
    const bindViewButton = (id, viewName) => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener("click", (e) => {
          if (el.tagName === "A") e.preventDefault();
          switchMainTab(viewName);
        });
      }
    };

    bindViewButton("btn-go-leaderboards-cta", "leaderboards-hub");
    bindViewButton("hero-btn-open-camels-us", "leaderboards-hub");
    bindViewButton("hero-btn-open-models", "models");
    bindViewButton("card-btn-open-camels-us", "leaderboards");
    bindViewButton("hero-btn-open-contribute", "contribute");
    bindViewButton("leaderboards-top-contribute-btn", "contribute");
    bindViewButton("resource-card-leaderboards", "leaderboards-hub");
    bindViewButton("resource-card-protocol", "about");
    bindViewButton("resource-card-publications", "publications");
    bindViewButton("footer-link-contribute", "contribute");
    bindViewButton("footer-link-protocol", "about");
    bindViewButton("footer-link-publications", "publications");
    bindViewButton("btn-back-to-leaderboards-hub", "leaderboards-hub");
    bindViewButton("btn-open-models-from-lb", "models");
    bindViewButton("btn-back-to-camels-lb", "leaderboards");
    bindViewButton("btn-back-to-lb-from-hydro", "leaderboards");
    bindViewButton("btn-back-to-about-from-camels", "about");
    bindViewButton("link-open-general-protocol", "about-protocol");
    bindViewButton("btn-back-to-about-from-protocol", "about");
    bindViewButton("btn-open-camels-spec-from-protocol", "about-camels-us");
    bindViewButton("btn-open-contribute-from-protocol", "contribute");
    bindViewButton("btn-open-camels-lb-from-about", "leaderboards");

    // Hydrograph Comparison Explorer buttons & controls
    const btnOpenHydroLb = document.getElementById("btn-open-hydrographs-from-lb");
    if (btnOpenHydroLb) {
      btnOpenHydroLb.addEventListener("click", () => openHydrographExplorer());
    }
    const btnTrayOpenHydro = document.getElementById("btn-tray-open-hydrographs");
    if (btnTrayOpenHydro) {
      btnTrayOpenHydro.addEventListener("click", () => openHydrographExplorer());
    }
    const btnTrayClear = document.getElementById("btn-tray-clear-compare");
    if (btnTrayClear) {
      btnTrayClear.addEventListener("click", () => {
        STATE.comparedModels = [];
        updateCompareUiState();
        renderActiveLeaderboardPanel();
      });
    }
    const btnInspCompareHydro = document.getElementById("btn-insp-compare-hydrograph");
    if (btnInspCompareHydro) {
      btnInspCompareHydro.addEventListener("click", () => {
        if (STATE.inspectedEntryId) {
          const activeLbId = getActiveLeaderboardId();
          const mk = `${activeLbId}__${STATE.inspectedEntryId}`;
          if (!STATE.comparedModels.includes(mk)) {
            STATE.comparedModels.push(mk);
          }
        }
        openHydrographExplorer();
      });
    }

    const hydroAddLbFilter = document.getElementById("hydro-add-lb-filter");
    if (hydroAddLbFilter) {
      hydroAddLbFilter.addEventListener("change", (e) => {
        STATE.hydroAddLbFilter = e.target.value;
        populateHydroModelPickerDropdown();
      });
    }

    const hydroAddModelSel = document.getElementById("hydro-add-model-select");
    if (hydroAddModelSel) {
      hydroAddModelSel.addEventListener("change", (e) => {
        const mk = e.target.value;
        if (mk) {
          if (!STATE.comparedModels.includes(mk)) {
            if (STATE.comparedModels.length >= 8) {
              STATE.comparedModels.shift();
            }
            STATE.comparedModels.push(mk);
          }
          e.target.value = "";
          updateCompareUiState();
          populateHydroModelPickerDropdown();
          fetchAndRenderHydroBasin();
        }
      });
    }

    const btnHydroClearModels = document.getElementById("btn-hydro-clear-models");
    if (btnHydroClearModels) {
      btnHydroClearModels.addEventListener("click", () => {
        STATE.comparedModels = [];
        ensureDefaultComparedModels();
        updateCompareUiState();
        populateHydroModelPickerDropdown();
        fetchAndRenderHydroBasin();
      });
    }

    const hydroHucFilter = document.getElementById("hydro-huc-filter");
    if (hydroHucFilter) {
      hydroHucFilter.addEventListener("change", (e) => {
        STATE.hydroHucFilter = e.target.value;
        populateHydroBasinDropdown();
        fetchAndRenderHydroBasin();
      });
    }

    const hydroBasinSelect = document.getElementById("hydro-basin-select");
    if (hydroBasinSelect) {
      hydroBasinSelect.addEventListener("change", (e) => {
        if (e.target.value) {
          STATE.hydroBasinId = e.target.value;
          fetchAndRenderHydroBasin();
        }
      });
    }

    const hydroBasinSearch = document.getElementById("hydro-basin-search");
    if (hydroBasinSearch) {
      hydroBasinSearch.addEventListener("input", (e) => {
        STATE.hydroBasinSearch = e.target.value;
        const prevBasin = STATE.hydroBasinId;
        populateHydroBasinDropdown();
        renderHydroConusMap();
        if (STATE.hydroBasinId !== prevBasin) {
          fetchAndRenderHydroBasin();
        }
      });
    }

    const stepHydroBasin = (delta) => {
      const list = getFilteredHydroBasins();
      if (list.length === 0) return;
      const curIdx = list.findIndex((b) => b.basin_id === STATE.hydroBasinId);
      const nextIdx = (curIdx + delta + list.length) % list.length;
      STATE.hydroBasinId = list[nextIdx].basin_id;
      const sel = document.getElementById("hydro-basin-select");
      if (sel) sel.value = STATE.hydroBasinId;
      fetchAndRenderHydroBasin();
    };

    const btnPrevBasin = document.getElementById("btn-hydro-prev-basin");
    if (btnPrevBasin) btnPrevBasin.addEventListener("click", () => stepHydroBasin(-1));
    const btnNextBasin = document.getElementById("btn-hydro-next-basin");
    if (btnNextBasin) btnNextBasin.addEventListener("click", () => stepHydroBasin(1));

    document.querySelectorAll("[data-hydro-zoom]").forEach((btn) => {
      btn.addEventListener("click", () => {
        applyHydroZoomPreset(btn.getAttribute("data-hydro-zoom"));
      });
    });

    const onDateInputChanged = () => {
      const payload = STATE.hydroBasinPayload;
      if (!payload || !payload.dates) return;
      const dates = payload.dates;
      const sVal = document.getElementById("hydro-start-date").value;
      const eVal = document.getElementById("hydro-end-date").value;
      let sIdx = dates.findIndex((d) => d >= sVal);
      let eIdx = dates.findIndex((d) => d >= eVal);
      if (sIdx < 0) sIdx = 0;
      if (eIdx < 0) eIdx = dates.length - 1;
      if (eIdx <= sIdx + 5) eIdx = Math.min(dates.length - 1, sIdx + 14);
      STATE.hydroZoomStart = sIdx;
      STATE.hydroZoomEnd = eIdx;
      STATE.hydroActivePreset = "custom";
      syncHydroZoomButtonsAndDates();
      renderHydrographCanvas();
    };
    const startInp = document.getElementById("hydro-start-date");
    const endInp = document.getElementById("hydro-end-date");
    if (startInp) startInp.addEventListener("change", onDateInputChanged);
    if (endInp) endInp.addEventListener("change", onDateInputChanged);

    ["daymet", "maurer", "nldas"].forEach((fk) => {
      const chk = document.getElementById(`chk-forcing-${fk}`);
      if (chk) {
        chk.addEventListener("change", (e) => {
          STATE.hydroForcingsOn[fk] = Boolean(e.target.checked);
          renderHydrographCanvas();
        });
      }
    });

    const qScaleSel = document.getElementById("hydro-q-scale");
    if (qScaleSel) {
      qScaleSel.addEventListener("change", (e) => {
        STATE.hydroQScale = e.target.value || "linear";
        renderHydrographCanvas();
      });
    }

    const tempModeSel = document.getElementById("hydro-temp-mode");
    if (tempModeSel) {
      tempModeSel.addEventListener("change", (e) => {
        STATE.hydroTempMode = e.target.value || "tmean_band";
        renderHydrographCanvas();
      });
    }

    initHydrographCanvasInteractions();
    initConusMapInteractions();

    const bindAboutCard = (cardId, targetView) => {
      const card = document.getElementById(cardId);
      if (card) {
        card.addEventListener("click", () => switchMainTab(targetView));
        card.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            switchMainTab(targetView);
          }
        });
      }
    };
    bindAboutCard("card-about-camels-us", "about-camels-us");

    const btnContribCamelsAbout = document.getElementById("btn-contribute-camels-from-about");
    if (btnContribCamelsAbout) {
      btnContribCamelsAbout.addEventListener("click", () => {
        const mipSel = document.getElementById("sub-mip-benchmark");
        if (mipSel) {
          mipSel.value = "camels_us";
          populateMipBenchmarkFields("camels_us");
        }
        switchMainTab("contribute");
      });
    }

    const camelsSuiteCard = document.getElementById("card-camels-us-daily-deterministic");
    if (camelsSuiteCard) {
      camelsSuiteCard.addEventListener("click", () => switchMainTab("leaderboards"));
      camelsSuiteCard.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          switchMainTab("leaderboards");
        }
      });
    }

    const pubResCard = document.getElementById("resource-card-publications");
    if (pubResCard) {
      pubResCard.addEventListener("click", (e) => {
        e.preventDefault();
        switchMainTab("publications");
      });
    }

    document.querySelectorAll("[data-open-lb]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const lbId = btn.getAttribute("data-open-lb");
        switchMainTab("leaderboards");
        selectLeaderboardTab(lbId);
      });
    });

    // Tier 1 Spatial Regime Tabs
    document.querySelectorAll("#spatial-tabs-bar .spatial-tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        STATE.activeSpatialTab = btn.getAttribute("data-spatial");
        persistNavState();
        renderLeaderboardTabs();
        renderActiveLeaderboardPanel();
      });
    });

    // Close 531-Basin Inspector
    const btnCloseInsp = document.getElementById("btn-close-inspector");
    if (btnCloseInsp) {
      btnCloseInsp.addEventListener("click", () => {
        STATE.inspectedEntryId = null;
        document.getElementById("basin-inspector-drawer").classList.remove("open");
        renderActiveLeaderboardPanel();
      });
    }

    // Active leaderboard search & filter controls
    document.getElementById("filter-search-input").addEventListener("input", (e) => {
      STATE.searchQuery = e.target.value.trim();
      renderActiveLeaderboardPanel();
    });

    document.getElementById("filter-category-select").addEventListener("change", (e) => {
      STATE.categoryFilter = e.target.value;
      renderActiveLeaderboardPanel();
    });

    // Table 2 Registry search, type filter & modal
    const famTypeFilter = document.getElementById("family-type-filter");
    if (famTypeFilter) {
      famTypeFilter.addEventListener("change", (e) => {
        STATE.familyTypeFilter = e.target.value;
        renderFamiliesGrid();
      });
    }
    const famSearchInput = document.getElementById("family-search-input");
    if (famSearchInput) {
      famSearchInput.addEventListener("input", (e) => {
        STATE.familySearchQuery = e.target.value.trim();
        renderFamiliesGrid();
      });
    }

    const subFamSelect = document.getElementById("sub-family-id");
    if (subFamSelect) {
      subFamSelect.addEventListener("change", (e) => {
        const fid = e.target.value;
        const fam = STATE.families.find((f) => f.family_id === fid);
        if (fam) {
          document.getElementById("sub-github-url").value = fam.github_url || "";
          if (fam.github_url) setClosedSourceSelection(false);
          document.getElementById("sub-zenodo-doi").value = fam.zenodo_doi || fam.zenodo_url || "";
          document.getElementById("sub-citation").value = fam.citation || "";
          updateLiveRepoBadgePreview();
        }
      });
    }

    ["sub-github-url", "sub-zenodo-doi"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.addEventListener("input", updateLiveRepoBadgePreview);
    });

    ["sub-closed-source", "step4-closed-source"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener("change", (ev) => {
          setClosedSourceSelection(Boolean(ev.target.checked));
          updateLiveRepoBadgePreview();
          syncStep4UrlInputsAndManifest();
        });
      }
    });

    const mipBenchSelect = document.getElementById("sub-mip-benchmark");
    if (mipBenchSelect) {
      mipBenchSelect.addEventListener("change", (e) => {
        populateMipBenchmarkFields(e.target.value);
      });
    }

    ["sub-spatial-regime", "sub-forcing"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener("change", syncContributeTargetLeaderboard);
      }
    });

    const formSubmit = document.getElementById("form-score-submit");
    if (formSubmit) {
      formSubmit.addEventListener("submit", (ev) => {
        ev.preventDefault();
        if (STATE.contribStep === 1) inspectPreviewSubmission(false);
        else if (STATE.contribStep === 2) scorePreviewSubmission();
      });
    }

    // Option C Contribute Stepper, Dropzone, Preview, and GitHub PR bindings
    document.querySelectorAll("[data-contrib-step-btn]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const stepNum = Number(btn.getAttribute("data-contrib-step-btn")) || 1;
        if (stepNum === 2 && !STATE.previewInspectData) return;
        if (stepNum === 3 && !STATE.activePreviewEntryId) return;
        setContribStep(stepNum);
      });
    });

    const dropzone = document.getElementById("sub-dropzone");
    const fileInput = document.getElementById("sub-file-input");
    if (dropzone && fileInput) {
      dropzone.addEventListener("click", () => fileInput.click());
      dropzone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          fileInput.click();
        }
      });
      dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("is-dragover");
      });
      dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("is-dragover");
      });
      dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("is-dragover");
        const dtFile = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (dtFile) handleSelectedPredictionFile(dtFile);
      });
      fileInput.addEventListener("change", (e) => {
        const picked = e.target.files && e.target.files[0];
        if (picked) handleSelectedPredictionFile(picked);
      });
    }

    const btnClearFile = document.getElementById("btn-clear-selected-file");
    if (btnClearFile) {
      btnClearFile.addEventListener("click", () => {
        if (fileInput) fileInput.value = "";
        handleSelectedPredictionFile(null);
      });
    }

    const btnInspect = document.getElementById("btn-preview-inspect");
    if (btnInspect) btnInspect.addEventListener("click", () => inspectPreviewSubmission(false));

    const btnReinspect = document.getElementById("btn-reinspect-overrides");
    if (btnReinspect) btnReinspect.addEventListener("click", () => inspectPreviewSubmission(true));

    const btnJumpStep4Direct = document.getElementById("btn-jump-step4-direct");
    if (btnJumpStep4Direct) btnJumpStep4Direct.addEventListener("click", () => setContribStep(4));

    const btnScore = document.getElementById("btn-preview-score");
    if (btnScore) btnScore.addEventListener("click", scorePreviewSubmission);

    const btnBackStep1 = document.getElementById("btn-back-to-step1");
    if (btnBackStep1) btnBackStep1.addEventListener("click", () => setContribStep(1));

    const btnViewPreviewOnLb = document.getElementById("btn-view-preview-on-lb");
    if (btnViewPreviewOnLb) {
      btnViewPreviewOnLb.addEventListener("click", () => {
        const list = loadLocalPreviews();
        const rec =
          list.find((p) => p.entry && p.entry.entry_id === STATE.activePreviewEntryId) ||
          list[list.length - 1];
        if (rec && rec.entry) {
          switchMainTab("leaderboards");
          selectLeaderboardTab(rec.entry.leaderboard_id);
          requestAnimationFrame(() => {
            const tr = document.getElementById(`row-${rec.entry.entry_id}`);
            if (tr) tr.scrollIntoView({ behavior: "smooth", block: "center" });
          });
        }
      });
    }

    const btnGotoStep4 = document.getElementById("btn-goto-step4");
    if (btnGotoStep4) btnGotoStep4.addEventListener("click", () => setContribStep(4));

    const btnClearCurrentPreview = document.getElementById("btn-clear-current-preview");
    if (btnClearCurrentPreview) {
      btnClearCurrentPreview.addEventListener("click", () => {
        removeLocalPreview(STATE.activePreviewEntryId);
        setContribStep(1);
      });
    }

    const btnBackFromStep4 = document.getElementById("btn-back-from-step4");
    if (btnBackFromStep4) btnBackFromStep4.addEventListener("click", () => setContribStep(1));

    ["step4-archive-url", "step4-code-url"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.addEventListener("input", syncStep4UrlInputsAndManifest);
    });

    const btnOpenGhIssue = document.getElementById("btn-open-github-issue");
    if (btnOpenGhIssue) btnOpenGhIssue.addEventListener("click", openGitHubSubmissionIssue);

    const btnExport = document.getElementById("btn-export-pr-manifest");
    if (btnExport) btnExport.addEventListener("click", exportPullRequestManifest);

    const btnCopyManifest = document.getElementById("btn-copy-manifest-json");
    if (btnCopyManifest) {
      btnCopyManifest.addEventListener("click", async () => {
        const manifest = syncStep4UrlInputsAndManifest();
        try {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(JSON.stringify(manifest, null, 2));
          }
          btnCopyManifest.textContent = "Copied JSON!";
          setTimeout(() => {
            btnCopyManifest.textContent = "Copy Manifest JSON";
          }, 1800);
        } catch (_e) {
          // ignore
        }
      });
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    initEventListeners();
    restoreNavStateFromUrlOrStorage();
    window.addEventListener("hashchange", () => {
      restoreNavStateFromUrlOrStorage();
      renderLeaderboardTabs();
      renderActiveLeaderboardPanel();
    });
    loadAllData();
  });
})();
