"""
cell_panel.py  —  performance-first, Bokeh-native polygon panel
================================================================

Architecture
------------
One Bokeh figure is built ONCE per geometry load.  All per-mode and per-BC
updates are handled by overwriting ONLY the `fill_color` column in the polygon
ColumnDataSource — no polygon geometry is ever re-serialised.

Display modes  (RadioButtonGroup)
---------------------------------
  Cell type   — polygons coloured by cgroup; legend on the right
  Water BC    — grey for unset cells; colour-mapped for cells with Ψ BC
  Solute BC   — grey for unset cells; colour-mapped for cells with c BC

Right panel   (changes with mode)
---------------------------------
  Cell type   tissue-type filter  +  colour legend
  Water BC    tissue-type filter  +  Ψ BC value input  +  Apply / Remove
  Solute BC   tissue-type filter  +  c  BC value input  +  Apply / Remove

Selection
---------
  click           replace selection
  shift + click   add / remove  [native Bokeh]
  ctrl  + drag    pan the view  [PanTool takes priority over TapTool on drag]
"""

import numpy as np
import pandas as pd
import param
import panel as pn
import matplotlib.pyplot as _mpl
import matplotlib.colors as _mcol
from bokeh.models import (
    ColumnDataSource, TapTool, HoverTool, BoxSelectTool,
    PanTool, CustomJS,
)
from bokeh.plotting import figure as bk_figure

from .state import AppState, CGROUP_NAMES

_W = 220

_PALETTE = {
    1:  "#a6cee3",  2:  "#1f78b4",  3:  "#b2df8a",  4:  "#33a02c",
    5:  "#fb9a99",  9:  "#e31a1c",  10: "#fdbf6f",   11: "#ff7f00",
    12: "#cab2d6",  13: "#6a3d9a",  14: "#ffff99",   15: "#b15928",
    16: "#dddddd",  19: "#8b6bb1",  20: "#5e3c8a",   23: "#ff9900",
}
_DEFAULT_CLR = "#cccccc"
_GREY        = "#d8d8d8"


def _sf(v, n: int = 4) -> float:
    """Round *v* to *n* significant figures for display in widgets."""
    if v == 0 or not np.isfinite(v):
        return v
    return float(f"{v:.{n}g}")


# ── selection state ───────────────────────────────────────────────────────────

class _SelState(param.Parameterized):
    indices = param.List(default=[])


# ── data preparation ──────────────────────────────────────────────────────────

def _augment_gdf(state: AppState) -> pd.DataFrame:
    mecha = state.mecha
    nwj   = state.nwj
    gdf   = mecha.network._cells_gdf.copy()
    gdf["cx"]      = gdf.geometry.centroid.x
    gdf["cy"]      = gdf.geometry.centroid.y
    gdf["node_id"] = (nwj + gdf["id_cell"].astype(int)).astype(int)
    cg_lookup = {
        n: int(mecha.network.graph.nodes[n].get("cgroup", -1))
        for n in range(nwj, nwj + state.n_cells)
    }
    gdf["cgroup"]      = gdf["node_id"].map(cg_lookup)
    gdf["cgroup_name"] = gdf["cgroup"].map(
        lambda cg: CGROUP_NAMES.get(cg, f"cgroup_{cg}")
    )
    return gdf.reset_index(drop=True)


def _extract_poly_list(gdf: pd.DataFrame) -> list:
    result = []
    for _, row in gdf.iterrows():
        xs, ys = row.geometry.exterior.coords.xy
        result.append({
            "x":           list(xs),
            "y":           list(ys),
            "id_cell":     int(row["id_cell"]),
            "node_id":     int(row["node_id"]),
            "cgroup":      int(row["cgroup"]),
            "cgroup_name": str(row["cgroup_name"]),
        })
    return result


# ── colour computation (fast — no geometry work) ──────────────────────────────

def _vals_to_hex(values: list, cmap_name: str) -> list:
    """Map list of floats (NaN → grey) to hex colour strings."""
    finite = [v for v in values if not np.isnan(v)]
    if not finite:
        return [_GREY] * len(values)
    vmin, vmax = min(finite), max(finite)
    if vmin == vmax:
        vmax = vmin + 1.0
    cmap = _mpl.get_cmap(cmap_name)
    norm = _mcol.Normalize(vmin=vmin, vmax=vmax)
    return [_mcol.to_hex(cmap(norm(v))) if not np.isnan(v) else _GREY
            for v in values]


def _compute_colors(mode: str, poly_list: list,
                    bc_w: dict, bc_s: dict) -> list:
    """Return one hex-colour string per polygon. Called on every mode/BC change."""
    if mode == "Cell type":
        return [_PALETTE.get(d["cgroup"], _DEFAULT_CLR) for d in poly_list]

    if mode == "Water BC":
        vals = [bc_w.get(d["node_id"], np.nan) for d in poly_list]
        return _vals_to_hex(vals, "RdYlBu")

    # Solute BC
    vals = [bc_s[d["node_id"]] * 1e6 if d["node_id"] in bc_s else np.nan
            for d in poly_list]
    return _vals_to_hex(vals, "plasma")


def _compute_hover_columns(poly_list: list,
                           bc_w: dict, bc_s: dict) -> tuple:
    """
    Return (psi_strings, conc_strings) — one entry per cell.
    Both columns are always shown in the hover tooltip regardless of display mode.
    """
    psi_vals  = [f"{bc_w[d['node_id']]:.4g} hPa" if d["node_id"] in bc_w else "—"
                 for d in poly_list]
    conc_vals = [f"{bc_s[d['node_id']]*1e6:.4g} µM" if d["node_id"] in bc_s else "—"
                 for d in poly_list]
    return psi_vals, conc_vals


def _ring_data(bc_w: dict, bc_s: dict,
               node_pos: dict) -> tuple:
    """Return (water_data, solute_data, both_data) for ring glyphs."""
    w, s = set(bc_w), set(bc_s)
    both = w & s

    def _pos(nodes):
        pts = [node_pos[n] for n in nodes if n in node_pos]
        return {"cx": [p[0] for p in pts], "cy": [p[1] for p in pts]}

    return _pos(w - both), _pos(s - both), _pos(both)


# ── Bokeh figure ──────────────────────────────────────────────────────────────

def _build_figure(poly_list: list, gdf: pd.DataFrame):
    """Build one Bokeh figure with polygon patches + centroid circles + BC rings."""

    # Polygon ColumnDataSource
    bg_src = ColumnDataSource({
        "xs":          [d["x"]          for d in poly_list],
        "ys":          [d["y"]          for d in poly_list],
        "color":       [_PALETTE.get(d["cgroup"], _DEFAULT_CLR) for d in poly_list],
        "id_cell":     [d["id_cell"]    for d in poly_list],
        "cgroup_name": [d["cgroup_name"] for d in poly_list],
    })

    # Centroid ColumnDataSource ("psi" and "conc" drive the hover tooltip)
    pts_src = ColumnDataSource({
        "cx":          gdf["cx"].tolist(),
        "cy":          gdf["cy"].tolist(),
        "id_cell":     gdf["id_cell"].tolist(),
        "cgroup_name": gdf["cgroup_name"].tolist(),
        "node_id":     gdf["node_id"].tolist(),
        "cgroup":      gdf["cgroup"].tolist(),
        "color":       [_PALETTE.get(cg, _DEFAULT_CLR) for cg in gdf["cgroup"]],
        "psi":         ["—"] * len(gdf),
        "conc":        ["—"] * len(gdf),
    })

    # BC ring ColumnDataSources
    w_ring  = ColumnDataSource({"cx": [], "cy": []})
    s_ring  = ColumnDataSource({"cx": [], "cy": []})
    b_ring  = ColumnDataSource({"cx": [], "cy": []})

    p = bk_figure(
        sizing_mode="stretch_both", min_height=520,
        tools="wheel_zoom,reset,save",   # pan and box-select added explicitly below
        title="Cell type",
        output_backend="canvas",
    )

    # Polygon patches
    p.patches("xs", "ys", source=bg_src,
              fill_color="color", line_color="black", line_width=0.35,
              name="poly")

    # BC rings
    for src, colour in [(w_ring, "#1E88E5"), (s_ring, "#FB8C00"), (b_ring, "#43A047")]:
        p.scatter("cx", "cy", source=src,
                  marker="circle", size=16, fill_alpha=0,
                  line_color=colour, line_width=2.5)

    # Centroid tap targets — capture renderer to pass to tools
    pts_renderer = p.scatter(
        "cx", "cy", source=pts_src,
        marker="circle", size=9,
        fill_color="color", line_color="black", line_width=0.5,
        alpha=0.7,
        selection_fill_color="red", selection_line_color="darkred",
        nonselection_alpha=0.25,
    )

    # Hover: always show Ψ and c regardless of display mode
    p.add_tools(HoverTool(
        renderers=[pts_renderer],
        tooltips=[
            ("Cell",   "@id_cell"),
            ("Tissue", "@cgroup_name"),
            ("Ψ",      "@psi"),
            ("c",      "@conc"),
        ],
    ))

    # Tap: click → select; shift+click → add/remove (native Bokeh)
    tap = TapTool(renderers=[pts_renderer])
    p.add_tools(tap)
    p.toolbar.active_tap = tap

    # Box-select: left-drag draws a selection rectangle (default active drag)
    box_sel = BoxSelectTool(renderers=[pts_renderer])
    p.add_tools(box_sel)
    p.toolbar.active_drag = box_sel

    # PanTool available in toolbar for manual switching
    pan = PanTool()
    p.add_tools(pan)

    # ctrl + drag → pan view via JavaScript range manipulation.
    # The capture-phase mousedown listener fires before Bokeh's BoxSelectTool,
    # so setting e.preventDefault()+stopPropagation() suppresses box-select
    # when ctrl is held, leaving range updates as the only action.
    p.js_on_event("document_ready", CustomJS(
        args=dict(xr=p.x_range, yr=p.y_range),
        code="""
        if (window._mechaPanSetup) return;
        window._mechaPanSetup = true;

        let panStart = null, sxs, sxe, sys, sye;

        // Find the Bokeh canvas overlay element (the one that receives events)
        function getEventsEl() {
            return document.querySelector('.bk-canvas-events') ||
                   document.querySelector('.bk-events-canvas') ||
                   document.querySelector('canvas');
        }

        function attach() {
            const el = getEventsEl();
            if (!el) { setTimeout(attach, 200); return; }

            el.addEventListener('mousedown', function(e) {
                if (!e.ctrlKey || e.button !== 0) return;
                e.preventDefault(); e.stopPropagation();
                const r = el.getBoundingClientRect();
                panStart = {x: e.clientX, y: e.clientY, w: r.width, h: r.height};
                sxs = xr.start; sxe = xr.end;
                sys = yr.start; sye = yr.end;
            }, {capture: true, passive: false});

            document.addEventListener('mousemove', function(e) {
                if (!panStart || !(e.buttons & 1) || !e.ctrlKey) {
                    if (!e.ctrlKey) panStart = null;
                    return;
                }
                const dx = (e.clientX - panStart.x) / panStart.w * (sxe - sxs);
                const dy = (e.clientY - panStart.y) / panStart.h * (sye - sys);
                xr.start = sxs - dx;  xr.end = sxe - dx;
                yr.start = sys + dy;  yr.end = sye + dy;
            });

            document.addEventListener('mouseup', function() { panStart = null; });
        }
        attach();
        """,
    ))

    return p, bg_src, pts_src, w_ring, s_ring, b_ring


# ── right-panel helpers ───────────────────────────────────────────────────────

def _tissue_filter(gdf: pd.DataFrame, pts_src: ColumnDataSource,
                   sel_state: _SelState) -> pn.Column:
    present = sorted(gdf["cgroup"].unique())
    opts    = {"— pick tissue —": -1}
    opts.update({CGROUP_NAMES.get(cg, f"cgroup_{cg}"): cg for cg in present})
    dd  = pn.widgets.Select(name="Tissue", options=opts, value=-1,
                            width=_W, sizing_mode="fixed")
    btn = pn.widgets.Button(name="Select all of this type",
                            button_type="warning", width=_W, sizing_mode="fixed")

    def _sel(_):
        cg = dd.value
        if cg == -1:
            return
        idx = list(gdf.index[gdf["cgroup"] == cg])
        pts_src.selected.indices = idx   # triggers on_change → sel_state

    btn.on_click(_sel)

    clear_btn = pn.widgets.Button(
        name="Clear selection", button_type="light",
        width=_W, sizing_mode="fixed",
    )

    def _clear(_):
        pts_src.selected.indices = []

    clear_btn.on_click(_clear)

    return pn.Column(
        pn.pane.Markdown("**Select by tissue type**", width=_W),
        dd, btn, clear_btn,
        width=_W + 10, sizing_mode="fixed",
    )


def _legend_panel(gdf: pd.DataFrame) -> pn.pane.HTML:
    rows = []
    for cg in sorted(gdf["cgroup"].unique()):
        col  = _PALETTE.get(cg, _DEFAULT_CLR)
        name = CGROUP_NAMES.get(cg, f"cgroup_{cg}")
        rows.append(
            f'<div style="display:flex;align-items:center;margin:2px 0">'
            f'<div style="width:13px;height:13px;background:{col};'
            f'border:1px solid #555;margin-right:6px;flex-shrink:0"></div>'
            f'<span style="font-size:0.83em">{name}</span></div>'
        )
    rings = (
        '<div style="margin-top:8px;font-size:0.82em">'
        'BC rings:&nbsp; '
        '<span style="color:#1E88E5">&#9679;</span> water &nbsp;'
        '<span style="color:#FB8C00">&#9679;</span> solute &nbsp;'
        '<span style="color:#43A047">&#9679;</span> both</div>'
    )
    return pn.pane.HTML("\n".join(rows) + rings, width=_W)


def _sel_header(n: int) -> str:
    return f"**{n} cell{'s' if n > 1 else ''} selected**" if n else "_No cells selected._"


def _right_celltype(gdf, pts_src, sel_state) -> pn.Column:
    return pn.Column(
        _tissue_filter(gdf, pts_src, sel_state),
        pn.layout.Divider(),
        pn.pane.Markdown("**Legend**", width=_W),
        _legend_panel(gdf),
        width=_W + 20, sizing_mode="fixed", scroll=True,
        styles={"max-height": "calc(100vh - 80px)", "overflow-y": "auto"},
    )


def _right_waterbc(indices, poly_list, gdf, pts_src,
                   sel_state, state) -> pn.Column:
    node_ids = [poly_list[i]["node_id"] for i in indices] if indices else []
    n        = len(node_ids)
    existing = {state.bc_water.get(nid) for nid in node_ids} - {None}

    bc_type = pn.widgets.Select(
        name="BC type", options=["free", "Dirichlet"],
        value="Dirichlet" if existing else "free",
        width=_W, sizing_mode="fixed",
    )
    bc_val = pn.widgets.FloatInput(
        name="Ψ (hPa)", value=_sf(next(iter(existing), 0.0), 4),
        step=10.0, width=_W, sizing_mode="fixed",
    )
    apply_btn = pn.widgets.Button(
        name=f"Apply to {n} cell{'s' if n > 1 else ''}",
        button_type="primary", width=_W, sizing_mode="fixed",
        disabled=(n == 0),
    )
    status = pn.pane.Markdown("", width=_W)

    def _apply(_):
        for nid in node_ids:
            state.set_bc_water(nid, bc_val.value if bc_type.value == "Dirichlet" else None)
        pts_src.selected.indices = []
        status.object = f"✓ {n} cell{'s' if n > 1 else ''}"

    apply_btn.on_click(_apply)

    return pn.Column(
        _tissue_filter(gdf, pts_src, sel_state),
        pn.layout.Divider(),
        pn.pane.Markdown(_sel_header(n), width=_W),
        pn.pane.Markdown("**Water potential BC**", width=_W),
        bc_type, bc_val,
        apply_btn, status,
        width=_W + 20, sizing_mode="fixed", scroll=True,
        styles={"max-height": "calc(100vh - 80px)", "overflow-y": "auto"},
    )


def _right_solutebc(indices, poly_list, gdf, pts_src,
                    sel_state, state) -> pn.Column:
    node_ids = [poly_list[i]["node_id"] for i in indices] if indices else []
    n        = len(node_ids)
    existing = {state.bc_solute.get(nid) for nid in node_ids} - {None}

    bc_type = pn.widgets.Select(
        name="BC type", options=["free", "Dirichlet"],
        value="Dirichlet" if existing else "free",
        width=_W, sizing_mode="fixed",
    )
    bc_val = pn.widgets.FloatInput(
        name="c (mol cm⁻³)", value=_sf(next(iter(existing), 0.0), 4),
        step=1e-6, width=_W, sizing_mode="fixed",
    )
    apply_btn = pn.widgets.Button(
        name=f"Apply to {n} cell{'s' if n > 1 else ''}",
        button_type="primary", width=_W, sizing_mode="fixed",
        disabled=(n == 0),
    )
    status = pn.pane.Markdown("", width=_W)

    def _apply(_):
        for nid in node_ids:
            state.set_bc_solute(nid, bc_val.value if bc_type.value == "Dirichlet" else None)
        pts_src.selected.indices = []
        status.object = f"✓ {n} cell{'s' if n > 1 else ''}"

    apply_btn.on_click(_apply)

    return pn.Column(
        _tissue_filter(gdf, pts_src, sel_state),
        pn.layout.Divider(),
        pn.pane.Markdown(_sel_header(n), width=_W),
        pn.pane.Markdown("**Solute concentration BC**", width=_W),
        bc_type, bc_val,
        apply_btn, status,
        width=_W + 20, sizing_mode="fixed", scroll=True,
        styles={"max-height": "calc(100vh - 80px)", "overflow-y": "auto"},
    )


# ── public factory ────────────────────────────────────────────────────────────

def build_cell_panel(state: AppState) -> pn.Column:
    if state.mecha is None:
        return pn.pane.Markdown("*Build geometry first.*")

    # ── one-time pre-computation ──────────────────────────────────────────
    gdf       = _augment_gdf(state)
    poly_list = _extract_poly_list(gdf)
    node_pos  = {d["node_id"]: (d["x"][0], d["y"][0]) for d in poly_list}
    # Use centroid for rings (more reliable than first vertex)
    ring_pos  = {int(r["node_id"]): (r["cx"], r["cy"])
                 for _, r in gdf.iterrows()}

    # ── build Bokeh figure (ONCE) ─────────────────────────────────────────
    p, bg_src, pts_src, w_ring, s_ring, b_ring = _build_figure(poly_list, gdf)
    plot_pane = pn.pane.Bokeh(p, sizing_mode="stretch_both")

    # ── selection state ───────────────────────────────────────────────────
    sel_state = _SelState()

    def _on_pts_select(attr, old, new):
        sel_state.indices = list(new)

    pts_src.selected.on_change("indices", _on_pts_select)

    # ── fast color + ring updates (no geometry re-serialisation) ──────────
    def _refresh_colors(mode=None):
        if mode is None:
            mode = mode_btn.value
        colors = _compute_colors(mode, poly_list,
                                 state.bc_water, state.bc_solute)
        psi_vals, conc_vals = _compute_hover_columns(
            poly_list, state.bc_water, state.bc_solute)
        bg_src.data  = dict(bg_src.data,  color=colors)
        pts_src.data = dict(pts_src.data, psi=psi_vals, conc=conc_vals)
        p.title.text = mode

    def _refresh_rings():
        wd, sd, bd = _ring_data(state.bc_water, state.bc_solute, ring_pos)
        w_ring.data = wd
        s_ring.data = sd
        b_ring.data = bd

    def _full_refresh(mode=None):
        _refresh_colors(mode)
        _refresh_rings()

    # Patch state setters so visual refresh fires automatically after every BC change
    _orig_set_bc_water  = state.set_bc_water
    _orig_set_bc_solute = state.set_bc_solute

    def _set_bc_water_and_refresh(nid, val):
        _orig_set_bc_water(nid, val)
        _full_refresh()

    def _set_bc_solute_and_refresh(nid, val):
        _orig_set_bc_solute(nid, val)
        _full_refresh()

    state.set_bc_water  = _set_bc_water_and_refresh
    state.set_bc_solute = _set_bc_solute_and_refresh

    # ── display mode toggle ───────────────────────────────────────────────
    mode_btn = pn.widgets.RadioButtonGroup(
        options=["Cell type", "Water BC", "Solute BC"],
        value="Cell type",
        button_type="default",
        sizing_mode="fixed",
    )

    def _on_mode(event):
        _full_refresh(event.new)

    mode_btn.param.watch(_on_mode, "value")

    # ── reactive right panel ──────────────────────────────────────────────
    def _right(mode, indices):
        if mode == "Cell type":
            return _right_celltype(gdf, pts_src, sel_state)
        if mode == "Water BC":
            return _right_waterbc(indices, poly_list, gdf, pts_src, sel_state, state)
        return _right_solutebc(indices, poly_list, gdf, pts_src, sel_state, state)

    right_pane = pn.bind(_right,
                         mode=mode_btn.param.value,
                         indices=sel_state.param.indices)

    return pn.Column(
        pn.Row(
            pn.pane.Markdown("**Display:**", width=70, align="center"),
            mode_btn,
            margin=(4, 0),
        ),
        pn.Row(
            plot_pane,
            pn.panel(right_pane, sizing_mode="fixed", width=_W + 20),
            sizing_mode="stretch_both",
        ),
        sizing_mode="stretch_both",
    )
