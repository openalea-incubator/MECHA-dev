"""
app.py
======
Main MECHA GUI application.

Workflow
--------
1. Select tissue type (needle | root) and click Build.
2. Two tabs: Cells (interactive BC editor) and Results.
3. Click Solve → hydraulic solve then solute solve; results appear in Results tab.

Launch
------
    python -m mecha.gui          # opens browser on http://localhost:5006
    pn.serve(build_app())        # programmatic
"""

import sys
import os
import traceback

import numpy as np
import panel as pn
import holoviews as hv
import hvplot.pandas       # noqa: F401 — registers .hvplot on DataFrame/GeoDataFrame

hv.extension("bokeh")
pn.extension(sizing_mode="fixed")

# ── MECHA / GRANAP source paths ───────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_MECHA_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
_GRANAP_SRC = os.path.abspath(
    os.path.join(_HERE, "..", "..", "..", "..", "GRANAP-dev", "src")
)
for _p in (_MECHA_SRC, _GRANAP_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mecha.mecha_class import Mecha          # noqa: E402
from mecha.utils.data_loader import InData   # noqa: E402
from mecha.utils.network_builder import NetworkBuilder  # noqa: E402
from mecha.solute_transport import SoluteTransport      # noqa: E402

from .state import AppState                   # noqa: E402
from .cell_panel import build_cell_panel     # noqa: E402


# ── Geometry builders ─────────────────────────────────────────────────────────

def _build_needle(barrier: int) -> Mecha:
    from granap.needle_class import NeedleAnatomy
    needle = NeedleAnatomy()
    needle.update_params("transfusion_tissue", "transfusion_type", True)
    needle.update_params("central_cylinder",   "shape", "half_ellipse")
    needle.export_to_adjencymatrix()

    indata = InData()
    indata.geometry.set_maturity_stages([barrier], [200.0])

    network = NetworkBuilder(needle)
    network.populate_from_network()

    return Mecha(indata, network=network)


def _build_root(barrier: int) -> Mecha:
    from granap.root_class import RootAnatomy
    root = RootAnatomy()
    root.export_to_adjencymatrix()

    indata = InData()
    indata.geometry.set_maturity_stages([barrier], [200.0])

    network = NetworkBuilder(root)
    network.populate_from_network()

    return Mecha(indata, network=network)


_BUILDERS = {"needle": _build_needle, "root": _build_root}


# ── Solve ─────────────────────────────────────────────────────────────────────

def _run_solve(state: AppState) -> None:
    mecha       = state.mecha
    I_MAT, I_SCE, H_IDX = 0, 0, 0

    # Hydraulic solve — get matrix and rhs so we can inject user BCs
    sol_W_default, _, mat_W, _, rhs_s = mecha.solve_W(h=H_IDX, i_maturity=I_MAT)
    sol_arr = np.asarray(sol_W_default, dtype=float).ravel()

    # Apply user-defined water potential Dirichlet BCs
    # (MECHA's built-in BCs — xylem, phloem — are already baked into mat_W/rhs_s;
    #  we add the GUI-defined ones on top by injecting identity rows.)
    if state.bc_water:
        import scipy.sparse.linalg as _spla
        rhs_w   = np.asarray(rhs_s, dtype=float).ravel()
        mat_lil = mat_W.tolil()
        n_sys   = rhs_w.size
        for node_id, psi_val in state.bc_water.items():
            if 0 <= node_id < n_sys:
                mat_lil[node_id, :]           = 0.0
                mat_lil[node_id, node_id]     = 1.0
                rhs_w[node_id]                = float(psi_val)
        sol_arr = _spla.spsolve(mat_lil.tocsr(), rhs_w)

    state.sol_W = sol_arr

    coo = mat_W.tocoo()
    fluxes = []
    for i, j, v in zip(coo.row, coo.col, coo.data):
        if i < j and v > 0:
            pi, pj = float(sol_arr[i]), float(sol_arr[j])
            if not (np.isnan(pi) or np.isnan(pj)):
                fluxes.append({"source": int(i), "target": int(j),
                                "flux": v * (pi - pj)})
    mecha.edge_flux_list[I_MAT][I_SCE] = fluxes

    # Solute solve
    DP = dict(
        apo_wall=state.D_APO,
        membrane=state.D_MEM,
        plasmodesmata=state.D_PD,
        sigma={cg: state.SIGMA for cg in range(1, 24)},
    )
    n_total = mecha.network.n_wall_junction + mecha.network.n_cells
    c_prev  = state.ic_solute if state.ic_solute is not None else np.zeros(n_total)

    c_sol = mecha.standard_solute_flux(
        h=H_IDX, i_maturity=I_MAT, i_scenario=I_SCE,
        diffusion_params=DP,
        boundary_conditions=state.bc_solute,
        c_prev=c_prev,
        theta=state.theta,
        operators="T",
        rhs=np.zeros(n_total),
    )
    state.c_sol      = np.asarray(c_sol).ravel()
    state.solve_done = True


# ── Results panel ─────────────────────────────────────────────────────────────

def _build_results_panel(state: AppState) -> pn.Column:
    if not state.solve_done or state.c_sol is None:
        return pn.pane.Markdown("*Run a solve first.*")

    nwj     = state.nwj
    c_cells = state.c_sol[nwj:]
    gdf     = state.mecha.network._cells_gdf.copy()

    gdf["c_uM"] = [
        float(c_cells[int(cid)]) * 1e6
        if int(cid) < len(c_cells) else np.nan
        for cid in gdf["id_cell"]
    ]
    c_plot = gdf.hvplot.polygons(
        geo=False, c="c_uM", cmap="plasma",
        line_color="black", line_width=0.3,
        responsive=True, min_height=500,
        title="Concentration (µM sucrose)",
        clabel="c (µM)",
    )

    if state.sol_W is not None:
        gdf["psi"] = [
            float(state.sol_W[nwj + int(cid)])
            if (nwj + int(cid)) < len(state.sol_W) else np.nan
            for cid in gdf["id_cell"]
        ]
        psi_plot = gdf.hvplot.polygons(
            geo=False, c="psi", cmap="RdYlBu",
            line_color="black", line_width=0.3,
            responsive=True, min_height=500,
            title="Water potential (hPa)",
            clabel="Ψ (hPa)",
        )
        plots = pn.Row(
            pn.pane.HoloViews(c_plot,   sizing_mode="stretch_both"),
            pn.pane.HoloViews(psi_plot, sizing_mode="stretch_both"),
            sizing_mode="stretch_both",
        )
    else:
        plots = pn.pane.HoloViews(c_plot, sizing_mode="stretch_both")

    c_min  = float(np.nanmin(c_cells)) * 1e6
    c_max  = float(np.nanmax(c_cells)) * 1e6
    c_mean = float(np.nanmean(c_cells)) * 1e6
    summary = pn.pane.Markdown(
        f"**Concentration (cells):**  "
        f"min = {c_min:.4g} µM  ·  mean = {c_mean:.4g} µM  ·  max = {c_max:.4g} µM"
    )

    return pn.Column(summary, plots, sizing_mode="stretch_both")


# ── Public factory ────────────────────────────────────────────────────────────

def build_app() -> pn.template.BootstrapTemplate:
    state = AppState()

    # ── Sidebar: geometry ─────────────────────────────────────────────────
    tissue_sel = pn.widgets.Select(
        name="Tissue type", options=["needle", "root"], value="needle",
        width=190, sizing_mode="fixed",
    )
    barrier_inp = pn.widgets.IntInput(
        name="Barrier (maturity index)", value=1, start=0,
        width=190, sizing_mode="fixed", visible=False,
    )

    def _on_tissue_change(event):
        barrier_inp.visible = (event.new == "root")

    tissue_sel.param.watch(_on_tissue_change, "value")

    build_btn = pn.widgets.Button(
        name="Build geometry", button_type="success",
        width=190, sizing_mode="fixed",
    )
    status_box = pn.pane.Alert(
        state.status, alert_type="info", width=190, sizing_mode="fixed",
    )

    def _update_status(_=None):
        status_box.object = state.status

    state.param.watch(_update_status, "status")

    solve_btn = pn.widgets.Button(
        name="Solve", button_type="primary",
        width=190, sizing_mode="fixed", disabled=True,
    )
    clear_btn = pn.widgets.Button(
        name="Clear", button_type="warning",
        width=190, sizing_mode="fixed", disabled=True,
    )

    # ── Tab container ─────────────────────────────────────────────────────
    tabs = pn.Tabs(
        ("Cells",   pn.pane.Markdown("*Build geometry to start.*")),
        ("Results", pn.pane.Markdown("*Run a solve to see results.*")),
        dynamic=False,
    )

    def _rebuild_cells():
        tabs[0] = ("Cells", build_cell_panel(state))

    def _rebuild_results():
        tabs[1] = ("Results", _build_results_panel(state))

    # ── Build callback ────────────────────────────────────────────────────
    def _on_build(_):
        build_btn.disabled = True
        solve_btn.disabled = True
        state.status = f"Building {tissue_sel.value} geometry…"
        try:
            state.tissue_type    = tissue_sel.value
            state.barrier        = barrier_inp.value if tissue_sel.value == "root" else 1
            state.mecha          = _BUILDERS[tissue_sel.value](state.barrier)
            state.bc_water       = {}
            state.bc_solute      = {}
            state.ic_solute      = None
            state.sol_W          = None
            state.c_sol          = None
            state.geometry_built = True
            state.solve_done     = False
            _rebuild_cells()
            nwj = state.mecha.network.n_wall_junction
            nw  = state.mecha.network.n_walls
            nc  = state.mecha.network.n_cells
            state.status = (
                f"{tissue_sel.value} built  \n"
                f"{nc} cells  \n"
                f"{nw} walls  \n"
                f"{nwj - nw} junctions"
            )
            solve_btn.disabled = False
            clear_btn.disabled = False
        except Exception:
            state.status = f"Build failed:\n{traceback.format_exc(limit=4)}"
        finally:
            build_btn.disabled = False

    build_btn.on_click(_on_build)

    # ── Solve callback ────────────────────────────────────────────────────
    def _on_solve(_):
        solve_btn.disabled = True
        state.status = "Solving…"
        try:
            _run_solve(state)
            _rebuild_results()
            c_cells = state.c_sol[state.nwj:]
            state.status = (
                f"Solve done  \n"
                f"c [{c_cells.min()*1e6:.4g},  \n"
                f"{c_cells.max()*1e6:.4g}] µM"
            )
            tabs.active = 1   # switch to Results
        except Exception:
            state.status = f"Solve failed:\n{traceback.format_exc(limit=4)}"
        finally:
            solve_btn.disabled = False

    solve_btn.on_click(_on_solve)

    # ── Clear callback ────────────────────────────────────────────────────
    def _on_clear(_):
        if state.mecha is None:
            return
        state.bc_water   = {}
        state.bc_solute  = {}
        state.ic_solute  = None
        state.sol_W      = None
        state.c_sol      = None
        state.solve_done = False
        _rebuild_cells()
        tabs[1] = ("Results", pn.pane.Markdown("*Run a solve to see results.*"))
        tabs.active = 0
        state.status = "Cleared."

    clear_btn.on_click(_on_clear)

    # ── Sidebar assembly ──────────────────────────────────────────────────
    sidebar = pn.Column(
        pn.pane.Markdown("## MECHA GUI"),
        pn.layout.Divider(),
        pn.pane.Markdown("**Geometry**"),
        tissue_sel,
        barrier_inp,
        build_btn,
        status_box,
        pn.layout.Divider(),
        solve_btn,
        pn.layout.Spacer(height=350),
        clear_btn,
        width=210, sizing_mode="fixed",
    )

    template = pn.template.BootstrapTemplate(
        title="MECHA — interactive cross-section",
        sidebar=[sidebar],
        sidebar_width=220,
    )
    template.main.append(
        pn.Column(tabs, sizing_mode="stretch_width")
    )
    return template


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pn.serve(build_app(), port=5006, show=True, title="MECHA GUI")
