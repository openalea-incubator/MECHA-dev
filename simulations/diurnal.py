"""
diurnal.py
==========
Diurnal (day/night) needle scenario.  All logic lives in ``needle.py``;
this module is the runnable entry point.

Configuration (per specification):
    d_pd    = 0.1 * 5.2e-6 * 86400  cm²/d
    PSI_XYL = -200 hPa
    PSI_AIR = -1e3 * f(t)  hPa   (diurnally modulated, peaks at noon)
    c_meso  = 50e-6 * g(t)  mol/cm³  (photosynthetic source, daytime only)

Measured outputs: j_w, j_s, j_s/j_w  — see needle.diurnal_flux_plot.
"""

import os
from needle import (
    _OUT_DIR,
    DiurnalParams,
    GeometryBase,
    NeedleDiurnalSimulation,
    SimResult,
    get_geometry_base,
    diurnal_flux_plot,
)


def run_diurnal(params: DiurnalParams = None,
                geometry: GeometryBase = None,
                verbose: bool = True) -> SimResult:
    params   = params   or DiurnalParams(seed=42)
    geometry = geometry or get_geometry_base(params.seed)
    return NeedleDiurnalSimulation(params, geometry, verbose=verbose).run()


if __name__ == '__main__':
    SEED   = 42
    params = DiurnalParams(seed=SEED, label='diurnal', max_steps=48)
    res    = run_diurnal(params)
    print(res.summary())
    stem = os.path.join(_OUT_DIR, f"needle_{res.params['label']}")
    res.save(stem + '.npz')
    diurnal_flux_plot(res, stem + '_fluxes.png')
    print(f"Saved: {stem}_fluxes.png")
