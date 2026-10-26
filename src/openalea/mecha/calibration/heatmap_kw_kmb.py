"""Evaluate the transpiration forward model on a 2-D (kw, kmb) grid.

Sweeps the wall conductivity ``kw`` and the background membrane conductivity
``kmb`` across their :data:`ParamSpace.DEFAULT_BOUNDS` (log10-spaced) while all
other transport coefficients (``kaqp``, ``kpl``) keep their needle-default
values. The forward model is solved once per grid node under a single fixed
boundary condition and the resulting radial conductance ``k_r`` is rendered
as a 2-D heatmap.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# GRANAP builds the needle anatomy; MECHA consumes it.
from openalea.granap.needle_class import NeedleAnatomy

from openalea.mecha import NetworkBuilder
from openalea.mecha.calibration import ForwardModel, ParamSpace, rh_to_water_potential


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
#: Grid resolution (N x N).
N = 32

#: Parameters swept on the two axes.
X_PARAM = "kw"    # x-axis: bulk wall conductivity   [cm hPa^-1 d^-1]
Y_PARAM = "kmb"   # y-axis: background membrane       [cm hPa^-1 d^-1]

#: Boundary conditions.
PSI_XYL_MPA = -0.5           # xylem water potential [MPa]
PSI_XYL_HPA = PSI_XYL_MPA * 1.0e4  # MECHA works in hPa: -0.5 MPa = -5000 hPa
RH_AIRSPACE = 0.99          # substomatal air-space relative humidity [-]
RH_POTENTIAL = rh_to_water_potential(RH_AIRSPACE, T=298.15) * 1.0e-4  # Convert to MPa

#: Model settings.
TEMPERATURE_K = 298.15       # room temperature for p_sat / Kelvin conversion
I_MATURITY = 0               # maturity-stage index to solve
OUTPUT = "k_r"               # "k_r" radial conductance or "Q" total flux


def build_forward_model() -> ForwardModel:
    """Build the (cached) needle forward model once."""
    anatomy = NeedleAnatomy()
    anatomy.export_to_adjencymatrix()
    network = NetworkBuilder(anatomy)
    network.populate_from_network()
    return ForwardModel(network, T=TEMPERATURE_K, i_maturity=I_MATURITY)


def evaluate_grid(fm: ForwardModel, n: int = N):
    """Evaluate the forward model on an ``n x n`` (kw, kmb) grid.

    Returns
    -------
    x_vals, y_vals : ndarray
        Log10-spaced parameter values along each axis.
    Z : ndarray, shape (n, n)
        Model output; ``Z[j, i]`` is the response at ``(x_vals[i], y_vals[j])``.
    """
    space = ParamSpace()  # pulls DEFAULT_BOUNDS for all coefficients
    (x_lo, x_hi) = space.DEFAULT_BOUNDS[X_PARAM]
    (y_lo, y_hi) = space.DEFAULT_BOUNDS[Y_PARAM]

    # Log10-spaced grids: the coefficients span several orders of magnitude.
    x_vals = np.logspace(np.log10(x_lo), np.log10(x_hi), n)
    y_vals = np.logspace(np.log10(y_lo), np.log10(y_hi), n)

    Z = np.empty((n, n), dtype=float)
    total = n * n
    for j, y in enumerate(y_vals):
        for i, x in enumerate(x_vals):
            theta = {X_PARAM: float(x), Y_PARAM: float(y)}
            # kaqp / kpl are omitted -> keep their needle-default values.
            Z[j, i] = fm.transpiration_flux(
                theta,
                psi_xyl=PSI_XYL_HPA,
                rh_airspace=RH_AIRSPACE,
                output=OUTPUT,
            )
            done = j * n + i + 1
            print(f"\r  grid {done}/{total}", end="", flush=True)
    print()
    return x_vals, y_vals, Z


def plot_heatmap(x_vals, y_vals, Z, save_path: str | None = None) -> None:
    """Render the ``(kw, kmb)`` grid response as a 2-D heatmap."""
    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    # pcolormesh with log-spaced edges keeps the axes physically meaningful.
    x_edges = _log_edges(x_vals)
    y_edges = _log_edges(y_vals)
    mesh = ax.pcolormesh(x_edges, y_edges, Z, shading="auto", cmap="viridis")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"{X_PARAM}  [cm hPa$^{{-1}}$ d$^{{-1}}$]")
    ax.set_ylabel(f"{Y_PARAM}  [cm hPa$^{{-1}}$ d$^{{-1}}$]")

    label = ("radial conductance $k_r$  [cm hPa$^{-1}$ d$^{-1}$]"
             if OUTPUT == "k_r" else "transpiration flux $Q$  [cm$^3$ d$^{-1}$]")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(label)

    ax.set_title(
        f"Forward model over ({X_PARAM}, {Y_PARAM})\n"
        f"$\\psi_{{xyl}}$ = {PSI_XYL_MPA} MPa,  $\\psi_{{air}}$ = {RH_POTENTIAL:.0f} MPa"
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved heatmap to {save_path}")
    plt.show()


def _log_edges(centers: np.ndarray) -> np.ndarray:
    """Return log-spaced cell edges for ``centers`` (length N -> N+1)."""
    log_c = np.log10(centers)
    step = (log_c[-1] - log_c[0]) / (len(log_c) - 1)
    log_edges = np.concatenate((
        [log_c[0] - step / 2.0],
        (log_c[:-1] + log_c[1:]) / 2.0,
        [log_c[-1] + step / 2.0],
    ))
    return 10.0 ** log_edges


def main() -> None:
    print(
        f"Evaluating forward model on a {N}x{N} ({X_PARAM}, {Y_PARAM}) grid\n"
        f"  psi_xyl = {PSI_XYL_MPA} MPa ({PSI_XYL_HPA:.0f} hPa),  "
        f"  psi_air = {RH_POTENTIAL:.0f} hPa"
    )
    fm = build_forward_model()
    x_vals, y_vals, Z = evaluate_grid(fm, n=N)
    plot_heatmap(x_vals, y_vals, Z, save_path="heatmap_kw_kmb.png")


if __name__ == "__main__":
    main()
