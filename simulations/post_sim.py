"""
post_sim.py
===========
Post-processing for needle simulation results: load ``SimResult`` objects
(from memory or ``.npz`` on disk) and render the plots / GIFs.

All heavy rendering (matplotlib animations, GeoPandas cell maps) lives here so
that ``needle.py`` stays a pure compute module — the parallel march produces
plain-array ``SimResult`` objects, and every figure is a decoupled
post-processing step run one result at a time.

Entry points
------------
``concentration_gif(res, path)``   transient cell concentration field (GIF)
``flow_network_plot(res, path, quantity)``  per-link arrows: 'solute' | 'water' | 'peclet'
``flow_network_gif(res, path, quantity)``   same, animated over the stored snapshots
``mass_balance_plot(res, path)``   interior mass / mass-to-sink vs. time
``diurnal_flux_plot(res, path)``   j_w, j_s, j_s/j_w (+ starch) time series
``starch_comparison_plot(results, path)``  k_f × k_r sweep overlay
``load_result(path)`` / ``load_results(dir_or_glob)``  load saved runs
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np

from needle import SimResult, g_photosynthesis


# ══════════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════════
def load_result(path: str) -> SimResult:
    """Load a single saved ``SimResult`` from a ``.npz`` path."""
    return SimResult.load(path)


def load_results(paths) -> list[SimResult]:
    """Load several results.

    ``paths`` may be a list of ``.npz`` paths, a directory (all ``*.npz`` in it),
    or a glob pattern.  Sorted by path for reproducible ordering.
    """
    if isinstance(paths, str):
        if os.path.isdir(paths):
            files = sorted(glob.glob(os.path.join(paths, '*.npz')))
        else:
            files = sorted(glob.glob(paths))
    else:
        files = list(paths)
    return [SimResult.load(p) for p in files]


# ══════════════════════════════════════════════════════════════════════════════
# Matplotlib backend (Agg for headless rendering)
# ══════════════════════════════════════════════════════════════════════════════
def _import_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _cell_gdf(res: SimResult):
    """Rebuild a GeoDataFrame of cell polygons from a result's stored geometry."""
    import geopandas as gpd
    from shapely import wkb
    if res.cell_polygons is None:
        raise ValueError("No stored cell geometry in this SimResult.")
    geoms = [wkb.loads(b) for b in res.cell_polygons]
    return gpd.GeoDataFrame({'id_cell': res.cell_ids}, geometry=geoms)


# ══════════════════════════════════════════════════════════════════════════════
# GRANAR anatomy map (tissue types)
# ══════════════════════════════════════════════════════════════════════════════
def anatomy_plot(res: SimResult, path: str) -> str:
    """Render the GRANAR-generated needle anatomy, coloured by tissue type.

    Reads the tissue cell-id arrays stored in ``res.extras`` (``tissue_meso``,
    ``tissue_strasburger``, ``tissue_phloem``, ``tissue_xylem``,
    ``tissue_airspace``) and paints each cell polygon by its class.
    """
    plt = _import_mpl()
    from matplotlib.patches import Patch

    gdf = _cell_gdf(res).copy()

    # tissue name → (colour, cell-id set) in the extras key order.
    tissue_spec = [
        ('mesophyll',       'tissue_meso',        '#4daf4a'),
        ('Strasburger',     'tissue_strasburger', '#984ea3'),
        ('phloem',          'tissue_phloem',      '#377eb8'),
        ('xylem',           'tissue_xylem',       '#e41a1c'),
        ('airspace',        'tissue_airspace',    '#bdbdbd'),
    ]
    other_color = '#f5f5dc'   # epidermis / cortex / unclassified

    id_to_tissue = {}
    color_of = {'other': other_color}
    for name, key, color in tissue_spec:
        ids = res.get_extra(key)
        color_of[name] = color
        if ids is None:
            continue
        for cid in np.asarray(ids).ravel():
            id_to_tissue[int(cid)] = name

    gdf['tissue'] = [id_to_tissue.get(int(cid), 'other') for cid in gdf['id_cell']]
    gdf['color']  = gdf['tissue'].map(color_of)

    fig, ax = plt.subplots(figsize=(9, 8))
    gdf.plot(ax=ax, color=gdf['color'], edgecolor='black', linewidth=0.3)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)')
    ax.set_title('GRANAR-generated needle anatomy (tissue types)')

    present = ['other'] + [n for n, k, _ in tissue_spec if res.get_extra(k) is not None
                           and np.asarray(res.get_extra(k)).size]
    handles = [Patch(facecolor=color_of[n], edgecolor='black',
                     label=('other' if n == 'other' else n)) for n in present]
    ax.legend(handles=handles, loc='upper right', fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Field-map GIF (concentration)
# ══════════════════════════════════════════════════════════════════════════════
def _render_cell_field_gif(gdf, concentration, node_of_cell, times, series,
                           path, max_frames=150, fps=12, cmap='plasma',
                           title='', cbar_label='', scale=1.0,
                           series_label='', series_unit='', series_scale=1.0):
    plt = _import_mpl()
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = concentration if concentration.ndim == 2 else concentration[None, :]
    n = frames.shape[0]
    stride = max(1, n // max_frames)
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    vmax = float(np.nanpercentile(frames[-1][node_of_cell] * scale, 99)) or 1.0

    gdf = gdf.copy()
    fig, ax = plt.subplots(figsize=(9, 7))

    def _draw(k):
        fi = idx[k]
        ax.clear()
        gdf['value'] = [float(frames[fi][n_i]) * scale for n_i in node_of_cell]
        gdf.plot(ax=ax, column='value', cmap=cmap, edgecolor='black',
                 linewidth=0.3, vmin=0.0, vmax=vmax,
                 missing_kwds={'color': 'lightgray'})
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)')
        t = '' if times is None else f't = {times[fi]:.3f} d'
        s = ('' if series is None
             else f'   |   {series_label}: {series[fi]*series_scale:.4g} {series_unit}'.rstrip())
        ax.set_title(f'{title}\n{t}{s}')
        return ax.collections

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, vmax))
    fig.colorbar(sm, ax=ax, label=cbar_label)
    anim = FuncAnimation(fig, _draw, frames=len(idx), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Public render entry points (take a SimResult)
# ══════════════════════════════════════════════════════════════════════════════
def concentration_gif(res: SimResult, path: str, max_frames: int = 150,
                      fps: int = 12, cmap: str = 'plasma') -> str:
    """Render the transient sucrose concentration field of a dynamic run as a GIF."""
    if res.mode != 'dynamic':
        raise ValueError("concentration_gif is only defined for dynamic runs.")
    return _render_cell_field_gif(
        _cell_gdf(res), res.concentration, res.node_of_cell,
        res.times, res.frac_to_sink, path,
        max_frames=max_frames, fps=fps, cmap=cmap,
        title='Needle sucrose transport', cbar_label='c (µM sucrose)', scale=1e6,
        series_label=res.sink_label, series_unit=res.sink_unit,
        series_scale=res.sink_scale,
    )


def starch_concentration_gif(res: SimResult, path: str, max_frames: int = 150,
                              fps: int = 12, cmap: str = 'YlOrBr') -> str:
    """Render the transient starch concentration field of a dynamic run as a GIF."""
    if res.mode != 'dynamic':
        raise ValueError("starch_concentration_gif is only defined for dynamic runs.")
    if res.concentration_starch is None:
        raise ValueError("This SimResult has no starch field (run with starch=True).")
    return _render_cell_field_gif(
        _cell_gdf(res), res.concentration_starch, res.node_of_cell,
        res.times, None, path,
        max_frames=max_frames, fps=fps, cmap=cmap,
        title='Needle starch concentration', cbar_label='c (µM starch)', scale=1e6,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Per-link network arrows (solute flux / water flux / Péclet)
# ══════════════════════════════════════════════════════════════════════════════
# quantity → (extras key, label, unit)
_FLOW_QUANTITIES = {
    'solute': ('edge_solute', 'Sucrose flux', 'mol d⁻¹'),
    'water':  ('edge_water',  'Water flux',   'cm³ d⁻¹'),
    'peclet': ('edge_peclet', 'Local Péclet (1−σ)F/D_e', '–'),
}
_BG = '#0d0d1a'


def _flow_series(res: SimResult, quantity: str):
    """(segments, path codes, snapshot times, (n_snap,E) signed values) for a quantity."""
    if quantity not in _FLOW_QUANTITIES:
        raise ValueError(f"quantity must be one of {tuple(_FLOW_QUANTITIES)}; got {quantity!r}")
    key = _FLOW_QUANTITIES[quantity][0]
    seg, code, vals = (res.get_extra('edge_segments'), res.get_extra('edge_path'),
                       res.get_extra(key))
    if seg is None or code is None or vals is None:
        raise ValueError(f"SimResult has no per-link snapshots ('{key}'); re-run the simulation.")
    vals = np.atleast_2d(np.asarray(vals, dtype=float))
    t = res.get_extra('edge_t')
    return seg, np.asarray(code), (np.zeros(len(vals)) if t is None else np.asarray(t)), vals


def _resolve_scale(filter_links: bool, tol_decades: float, log_decades):
    """(tol, span): drop links below ref_max·10^-tol; log width span (0 → linear)."""
    tol = float(tol_decades) if filter_links else None
    if log_decades is None:
        log_decades = tol if tol else 4.0
    return tol, float(log_decades)


def _keep_mask(vals, ref_max, tol):
    a = np.abs(np.nan_to_num(vals, nan=0.0))
    if ref_max <= 0.0:
        return np.zeros(a.shape, dtype=bool)
    return a > (ref_max * 10.0 ** (-tol) if tol is not None else 0.0)


def _draw_flow_frame(ax, seg, code, vals, ref_max, tol, log_decades, title):
    """Draw one snapshot: width/alpha ∝ |value| / ref_max, arrow = sign (u→v if > 0)."""
    from needle import EDGE_PATHS
    from openalea.mecha.utils.network_export import _draw_edge_vectors

    a = np.abs(np.nan_to_num(vals, nan=0.0))
    keep = _keep_mask(vals, ref_max, tol)
    if ref_max <= 0.0:
        mag = np.zeros_like(a)
    elif log_decades:
        mag = np.clip(1.0 + np.log10(np.maximum(a, 1e-300) / ref_max) / log_decades, 0.0, 1.0)
    else:
        mag = np.clip(a / ref_max, 0.0, 1.0)

    path_data = {pt: [] for pt in EDGE_PATHS}
    # Kept links at the bottom of the window get mag≈0; floor them so they still draw.
    for k in np.nonzero(keep)[0]:
        u, v = seg[k]
        src, dst = (u, v) if vals[k] >= 0 else (v, u)
        path_data[EDGE_PATHS[int(code[k])]].append((src, dst, max(float(mag[k]), 1e-3)))

    ax.set_facecolor(_BG)
    ax.set_aspect('equal')
    _draw_edge_vectors(ax, path_data, seg.reshape(-1, 2))
    ax.set_title(title, color='white', fontsize=11)


def _flow_title(quantity, t, vals, code, ref_max, tol, log_decades):
    from needle import EDGE_PATHS
    _, label, unit = _FLOW_QUANTITIES[quantity]
    a = np.abs(np.nan_to_num(vals, nan=0.0))
    scale = f"log {log_decades:g} dec" if log_decades else "linear"
    shown = int(_keep_mask(vals, ref_max, tol).sum())
    filt = f", |·| > 1e-{tol:g}·max" if tol is not None else ""
    head = (f"{label} [{unit}] – links, t = {t:.2f} d   (ref max = {ref_max:.2e}, {scale})\n"
            f"shown {shown} of {a.size} links{filt}")
    if quantity == 'peclet':
        fin = np.isfinite(vals)
        stats = "   ".join(
            f"{pt}: |Pe|>1 in {100*np.mean(a[(code == i) & fin] > 1):.0f}%"
            for i, pt in enumerate(EDGE_PATHS) if np.any((code == i) & fin))
    else:
        stats = "   ".join(f"Σ|·| {pt}: {a[code == i].sum():.2e}"
                           for i, pt in enumerate(EDGE_PATHS))
    return f"{head}\n{stats}"


def flow_network_plot(res: SimResult, path: Optional[str] = None,
                      quantity: str = 'solute', step: int = -1, ax=None,
                      filter_links: bool = True, tol_decades: float = 2.0,
                      log_decades: Optional[float] = None) -> Optional[str]:
    """Per-link arrows on apo / membrane / PD links, styled like MECHA's ``plot_flow_network``.

    ``quantity``: 'solute' (sucrose flux), 'water' (volume flux) or 'peclet'
    (signed local Péclet). ``step`` indexes the stored snapshots (``extras['edge_t']``).
    ``filter_links`` drops links with |value| ≤ max·10^-``tol_decades``.
    ``log_decades``: log width span; None → ``tol_decades`` (4 if unfiltered), 0 → linear.
    Output format follows the extension of ``path`` ('.pdf' keeps arrows as vectors).
    """
    plt = _import_mpl()
    seg, code, times, series = _flow_series(res, quantity)
    vals = series[step]
    ref_max = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 0.0
    tol, span = _resolve_scale(filter_links, tol_decades, log_decades)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 12), facecolor=_BG)
    _draw_flow_frame(ax, seg, code, vals, ref_max, tol, span,
                     _flow_title(quantity, float(times[step]), vals, code, ref_max, tol, span))
    if own_fig:
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=_BG)
        plt.close(fig)
    return path


def flow_network_gif(res: SimResult, path: str, quantity: str = 'solute',
                     max_frames: int = 60, fps: int = 4,
                     filter_links: bool = True, tol_decades: float = 2.0,
                     log_decades: Optional[float] = None) -> str:
    """Animate the per-link snapshots of ``quantity`` (one frame per stored snapshot).

    Width/alpha and the filter share ONE reference (max over all frames), so frames
    are comparable. Filter/scale options as in ``flow_network_plot``.
    """
    plt = _import_mpl()
    from matplotlib.animation import FuncAnimation, PillowWriter

    seg, code, times, series = _flow_series(res, quantity)
    n = series.shape[0]
    stride = max(1, n // max_frames)
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    finite = np.abs(series[np.isfinite(series)])
    ref_max = float(finite.max()) if finite.size else 0.0
    tol, span = _resolve_scale(filter_links, tol_decades, log_decades)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=_BG)

    def _draw(k):
        fi = idx[k]
        ax.clear()
        _draw_flow_frame(ax, seg, code, series[fi], ref_max, tol, span,
                         _flow_title(quantity, float(times[fi]), series[fi], code,
                                     ref_max, tol, span))
        return []

    anim = FuncAnimation(fig, _draw, frames=len(idx), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), savefig_kwargs={'facecolor': _BG})
    plt.close(fig)
    return path


def mass_balance_plot(res: SimResult, path: str) -> str:
    """Interior solute mass / cumulative mass-to-sink vs. time (dynamic only)."""
    if res.mode != 'dynamic':
        raise ValueError("mass_balance_plot is only defined for dynamic runs.")
    plt = _import_mpl()
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(res.times, np.asarray(res.interior_mass) * 1e12, 'b-',
             label='interior mass')
    ax1.set_xlabel('t (d)'); ax1.set_ylabel('interior solute mass (pmol)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(res.times, np.asarray(res.frac_to_sink) * res.sink_scale, 'r--',
             label=res.sink_label)
    unit = f' ({res.sink_unit})' if res.sink_unit else ''
    ax2.set_ylabel(f'{res.sink_label}{unit}', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    fig.suptitle('Solute mass balance')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Diurnal time-series plot
# ══════════════════════════════════════════════════════════════════════════════
def diurnal_flux_plot(res: SimResult, path: str) -> str:
    """Time evolution of a diurnal run: j_w, j_s, j_s/j_w (+ starch if present).

    Reads the series from ``res.extras`` (via ``get_extra``); a starch panel is
    added automatically when the run produced an ``m_starch`` series.
    """
    plt = _import_mpl()

    t   = np.asarray(res.times)
    j_w = np.asarray(res.get_extra('j_w'))
    j_s = np.asarray(res.get_extra('j_s'))
    m_s = res.get_extra('m_starch')
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(np.abs(j_w) > 0, j_s / j_w, np.nan)

    period  = float(res.param('period', 1.0))
    g_phase = float(res.param('g_phase', 0.0))

    n_panels = 4 if m_s is not None else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3 * n_panels), sharex=True)
    axes = np.atleast_1d(axes)

    def _night_bands(ax):
        # Night = dark period (no photosynthesis); f no longer reaches zero.
        tt = np.linspace(t[0], t[-1], 2000)
        night = g_photosynthesis(tt, period, g_phase) <= 0.0
        ax.fill_between(tt, 0, 1, where=night, transform=ax.get_xaxis_transform(),
                        color='0.85', zorder=0, step='mid')

    ax_jw, ax_js, ax_rat = axes[0], axes[1], axes[-1]

    _night_bands(ax_jw)
    ax_jw.plot(t, j_w, 'b-o', ms=3, lw=1.2)
    ax_jw.set_ylabel('j_w  (cm³/d)', color='b')
    ax_jw.tick_params(axis='y', labelcolor='b')
    ax_jw.set_title('Transpirational water flux j_w')

    _night_bands(ax_js)
    ax_js.plot(t, j_s * 1e12, 'r-o', ms=3, lw=1.2)
    ax_js.set_ylabel('j_s  (pmol/d)', color='r')
    ax_js.tick_params(axis='y', labelcolor='r')
    ax_js.set_title('Transported solute flux j_s (to sink)')

    if m_s is not None:
        ax_st = axes[2]
        _night_bands(ax_st)
        ax_st.plot(t, np.asarray(m_s) * 1e6, 'g-o', ms=3, lw=1.2)
        ax_st.set_ylabel('mean starch  (µM)', color='g')
        ax_st.tick_params(axis='y', labelcolor='g')
        ax_st.set_title('Interior starch concentration (cell mean)')

    _night_bands(ax_rat)
    ax_rat.plot(t, ratio * 1e3, 'k-o', ms=3, lw=1.2)
    ax_rat.set_ylabel('j_s / j_w  (mM)')
    ax_rat.set_xlabel('t  (d)')
    ax_rat.set_title('Export ratio j_s / j_w')

    fig.suptitle('Diurnal needle transport: water, solute, and their quotient')
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Diel (time-dependent rate) starch time series
# ══════════════════════════════════════════════════════════════════════════════
def diel_starch_plot(res: SimResult, path: str) -> str:
    """Five-panel diel run: k_f/k_r, j_w, j_s, mean starch, j_s/j_w vs. time.

    Meant for ``NeedleDiurnalDielStarchSimulation`` results, which store the
    instantaneous ``k_f``/``k_r`` series in ``res.extras``.
    """
    plt = _import_mpl()

    t   = np.asarray(res.times)
    j_w = np.asarray(res.get_extra('j_w'))
    j_s = np.asarray(res.get_extra('j_s'))
    m_s = np.asarray(res.get_extra('m_starch'))
    k_f = res.get_extra('k_f')
    k_r = res.get_extra('k_r')
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(np.abs(j_w) > 0, j_s / j_w, np.nan)

    period  = float(res.param('period', 1.0))
    g_phase = float(res.param('g_phase', 0.0))

    fig, axes = plt.subplots(5, 1, figsize=(9, 14), sharex=True)
    ax_k, ax_jw, ax_js, ax_st, ax_rat = axes

    def _night_bands(ax):
        tt = np.linspace(t[0], t[-1], 2000)
        night = g_photosynthesis(tt, period, g_phase) <= 0.0
        ax.fill_between(tt, 0, 1, where=night, transform=ax.get_xaxis_transform(),
                        color='0.85', zorder=0, step='mid')

    _night_bands(ax_k)
    if k_f is not None:
        ax_k.plot(t, np.asarray(k_f), '-', color='#4daf4a', lw=1.4,
                  label='k_f  sucrose→starch (day)')
    if k_r is not None:
        ax_k.plot(t, np.asarray(k_r), '-', color='#984ea3', lw=1.4,
                  label='k_r  starch→sucrose (night)')
    ax_k.set_ylabel('rate  (1/d)')
    ax_k.set_title('Time-dependent conversion rates (forward by day, backward by night)')
    ax_k.legend(loc='upper right', fontsize=8)

    _night_bands(ax_jw)
    ax_jw.plot(t, j_w, 'b-', lw=1.2)
    ax_jw.set_ylabel('j_w  (cm³/d)', color='b')
    ax_jw.tick_params(axis='y', labelcolor='b')
    ax_jw.set_title('Transpirational water flux j_w')

    _night_bands(ax_js)
    ax_js.plot(t, j_s * 1e12, 'r-', lw=1.2)
    ax_js.set_ylabel('j_s  (pmol/d)', color='r')
    ax_js.tick_params(axis='y', labelcolor='r')
    ax_js.set_title('Transported solute flux j_s (to sink)')

    _night_bands(ax_st)
    ax_st.plot(t, m_s * 1e6, 'g-', lw=1.2)
    ax_st.set_ylabel('mean starch  (µM)', color='g')
    ax_st.tick_params(axis='y', labelcolor='g')
    ax_st.set_title('Interior starch concentration (cell mean)')

    _night_bands(ax_rat)
    ax_rat.plot(t, ratio * 1e3, 'k-', lw=1.2)
    ax_rat.set_ylabel('j_s / j_w  (mM)')
    ax_rat.set_xlabel('t  (d)')
    ax_rat.set_title('Export ratio j_s / j_w')

    fig.suptitle('Diel sucrose ↔ starch: time-dependent conversion')
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Starch sweep comparison
# ══════════════════════════════════════════════════════════════════════════════
def starch_comparison_plot(results: list[SimResult], path: str) -> str:
    """Four-panel comparison: j_w, j_s, mean starch, j_s/j_w for all (k_f, k_r)."""
    plt = _import_mpl()

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    ax_jw, ax_js, ax_st, ax_rat = axes

    period  = float(results[0].param('period', 1.0))
    g_phase = float(results[0].param('g_phase', 0.0))

    def _night_bands(ax, t):
        tt = np.linspace(t[0], t[-1], 2000)
        night = g_photosynthesis(tt, period, g_phase) <= 0.0
        ax.fill_between(tt, 0, 1, where=night, transform=ax.get_xaxis_transform(),
                        color='0.88', zorder=0, step='mid')

    colors = plt.cm.viridis(np.linspace(0, 0.95, len(results)))

    for res, col in zip(results, colors):
        t   = np.asarray(res.times)
        j_w = np.asarray(res.get_extra('j_w'))
        j_s = np.asarray(res.get_extra('j_s'))
        m_s = res.get_extra('m_starch')
        m_s = np.asarray(m_s) if m_s is not None else np.zeros_like(j_s)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(np.abs(j_w) > 0, j_s / j_w, np.nan)
        starch = res.params.get('starch') or {}
        kf  = starch.get('k_syn', res.params.get('k_starch_syn', 0.0))
        kr  = starch.get('k_deg', res.params.get('k_starch_deg', 0.0))
        lbl = f"k_f={kf:.1f}, k_r={kr:.1f}"

        for ax in axes:
            _night_bands(ax, t)

        ax_jw.plot(t, j_w,          color=col, lw=1.2, ms=2, marker='o', label=lbl)
        ax_js.plot(t, j_s  * 1e12,  color=col, lw=1.2, ms=2, marker='o')
        ax_st.plot(t, m_s  * 1e6,   color=col, lw=1.2, ms=2, marker='o')
        ax_rat.plot(t, ratio * 1e3, color=col, lw=1.2, ms=2, marker='o')

    ax_jw.set_ylabel('j_w  (cm³/d)');   ax_jw.set_title('Transpirational water flux j_w')
    ax_jw.legend(fontsize=7, loc='upper right')
    ax_js.set_ylabel('j_s  (pmol/d)');  ax_js.set_title('Sucrose export flux j_s (to sink)')
    ax_st.set_ylabel('mean starch (µM)'); ax_st.set_title('Interior starch (cell mean)')
    ax_rat.set_ylabel('j_s / j_w  (mM)'); ax_rat.set_xlabel('t  (d)')
    ax_rat.set_title('Export ratio j_s / j_w')

    fig.suptitle('Diurnal needle: sucrose ↔ starch sweep  (k_f, k_r)')
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path
