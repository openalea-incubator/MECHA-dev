"""
flux_cdiff.py
=============
Parameter sweep: plasmodesmatal sucrose diffusivity (d_pd) × total transpiration
water flux (Q_tot, driven by the air ↔ xylem water-potential difference).

Grid: 10 × 10 = 100 experiments, each a DynamicParams run of exactly 200 steps.
Output metric: cumulative solute mass transported to sink [pmol] after 200 steps.

Axes
----
  x : Q_tot = total water flux out through the evaporating (wall_air) walls
              [cm³/d], read back from MECHA at the pulse-free baseline solve.
              Set indirectly by sweeping |ψ_air| over [500, 30 000] hPa.
  y : d_pd  = sucrose diffusivity inside the plasmodesmata [cm²/d], swept over
              three orders of magnitude below the pure-water value.

d_pd calibration
----------------
``d_pd`` is passed straight through to ``hormones.diff1_pd1`` and enters the PD
solute conductance in ``_fill_plasmodesmata`` as

    DF = pd_section · temp_factor / thickness · 1e-4 · diff1_pd1     (cm³/d),

i.e. the plasmodesmatal *geometry* (pd_section, temp_factor, thickness) is a
separate multiplicative prefactor.  ``d_pd`` therefore IS the physical sucrose
diffusion coefficient inside the PD cytoplasm — no extra geometric conversion is
needed.  The top of the sweep is anchored to the free-solution (pure-water)
sucrose diffusivity and lowered over three decades:

    D_SUCROSE_WATER ≈ 5.2e-6 cm²/s  (sucrose in water at 25 °C)
                    = 5.2e-6 × 86 400 s/d ≈ 0.449 cm²/d
    d_pd ∈ [1e-3 · D_SUCROSE_WATER, D_SUCROSE_WATER].
"""

import os
import sys
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR    = os.path.join(_SCRIPT_DIR, 'outputs')
os.makedirs(_OUT_DIR, exist_ok=True)

sys.path.insert(0, _SCRIPT_DIR)   # so 'import needle' finds the sibling file

from needle import DynamicParams, run_experiments

# ── Sweep definition ──────────────────────────────────────────────────────────
N       = 10
SEED    = 42
PSI_XYL = -200.0   # hPa  (fixed)

# Free-solution sucrose diffusion coefficient (pure water, 25 °C).
D_SUCROSE_WATER_CM2_S = 5.2e-6            # cm²/s  (literature)
D_SUCROSE_WATER       = D_SUCROSE_WATER_CM2_S * 86400.0   # cm²/d ≈ 0.449

# y-axis: d_pd from 1e-3·D_sugar up to the pure-water value (3 decades).
D_PD_VALUES  = np.logspace(np.log10(1e-3 * D_SUCROSE_WATER),
                           np.log10(D_SUCROSE_WATER), N)   # cm²/d

# x-axis is a computed output (total water flux); it is *driven* by |ψ_air|.
PSI_AIR_ABS  = np.logspace(np.log10(500),  np.log10(30000), N)  # |ψ_air| hPa
# ψ_air must be more negative than ψ_xyl to drive outward water flow.
PSI_AIR_VALUES = -PSI_AIR_ABS                                    # hPa (negative)

# ── Build 100 parameter sets (row-major: outer = d_pd, inner = psi_air) ───────
params_grid = [
    DynamicParams(
        seed=SEED,
        psi_xyl=PSI_XYL,
        air_mode='psi',
        psi_atm=psi_atm,
        d_pd=d_pd,
        max_steps=200,
        sink_fraction=float('inf'),   # disable early stop; always run 200 steps
        label=f'dpd{d_pd:.2e}_psiair{abs(psi_atm):.0f}',
    )
    for d_pd in D_PD_VALUES
    for psi_atm in PSI_AIR_VALUES
]


def _plot_matrix(mass_matrix, flux_axis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mass_matrix, origin='lower', aspect='auto', cmap='viridis',
                   extent=[0, N, 0, N])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('cumulative mass to sink after 200 steps (pmol)')

    tick_pos = np.arange(N) + 0.5
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f'{v:.2e}' for v in flux_axis],
                       rotation=45, ha='right', fontsize=7)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([f'{v:.2e}' for v in D_PD_VALUES], fontsize=7)
    ax.set_xlabel('total water flux Q_tot (cm³/d)')
    ax.set_ylabel('d_pd (cm²/d)  [top = sucrose in pure water]')
    ax.set_title(f'Solute transport to sink — 200-step dynamic sweep\n'
                 f'ψ_xyl = {PSI_XYL:.0f} hPa  |  seed = {SEED}')
    plt.tight_layout()

    out_path = os.path.join(_OUT_DIR, 'flux_cdiff_matrix.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == '__main__':
    print(f"Running {len(params_grid)} experiments ({N}×{N} grid) …")
    results = run_experiments(params_grid, max_workers=None)
    print("Done.")

    mass_matrix = np.array([
        (float(res.frac_to_sink[-1]) * 1e12) if res.frac_to_sink is not None else 0.0
        for res in results
    ]).reshape(N, N)

    # Total water flux per column (independent of d_pd at the sucrose-free
    # baseline); average over rows to smooth any tiny numerical variation.
    flux_matrix = np.array([
        (float(res.water_flux_total) if res.water_flux_total is not None else np.nan)
        for res in results
    ]).reshape(N, N)
    flux_axis = np.nanmean(flux_matrix, axis=0)   # one value per psi_air column

    _plot_matrix(mass_matrix, flux_axis)

    npz_path = os.path.join(_OUT_DIR, 'flux_cdiff_matrix.npz')
    np.savez_compressed(npz_path, mass_pmol=mass_matrix,
                        d_pd=D_PD_VALUES, psi_air=PSI_AIR_VALUES,
                        water_flux=flux_matrix, flux_axis=flux_axis,
                        psi_xyl=PSI_XYL, d_sucrose_water=D_SUCROSE_WATER)
    print(f"Saved → {npz_path}")
