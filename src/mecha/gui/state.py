import param
import numpy as np


CGROUP_NAMES = {
    1: "epidermis",
    2: "exodermis / cortex",
    3: "endodermis",
    4: "cortex / mesophyll",
    5: "pericycle",
    9: "hypodermis",
    10: "proto-phloem",
    11: "phloem sieve",
    12: "Strasburger / companion",
    13: "xylem",
    14: "cambium",
    15: "parenchyma",
    16: "airspace",
    19: "meta-xylem",
    20: "proto-xylem",
    23: "phloem",
}


class AppState(param.Parameterized):
    """Shared mutable state for all MECHA GUI panels."""

    # ── Geometry ────────────────────────────────────────────────────────────
    mecha        = param.Parameter(default=None, doc="Mecha instance (None until built)")
    tissue_type  = param.ObjectSelector(default="needle", objects=["needle", "root"])
    barrier      = param.Integer(default=1, bounds=(0, None),
                                 doc="Maturity-stage index passed to set_maturity_stages")

    # ── Boundary conditions (Dirichlet) ─────────────────────────────────────
    # Keys are full-network node indices (0 … n_wall_junction + n_cells - 1).
    # Nodes absent from the dict are treated as free (no Dirichlet pin).
    bc_water  = param.Dict(default={}, doc="Water potential BCs {node_id: hPa}")
    bc_solute = param.Dict(default={}, doc="Solute concentration BCs {node_id: mol/cm³}")

    # ── Initial concentrations (t = 0 only) ─────────────────────────────────
    # None → zeros everywhere when the transient solver is called.
    ic_solute = param.Parameter(default=None, doc="c₀ array (full-network size) [mol/cm³]")

    # ── Transport parameters ─────────────────────────────────────────────────
    D_APO = param.Number(default=0.10,  bounds=(0, None), doc="Apoplastic wall diffusivity [cm²/d]")
    D_MEM = param.Number(default=1e-6,  bounds=(0, None), doc="Passive transmembrane diffusivity [cm²/d]")
    D_PD  = param.Number(default=1e-4,  bounds=(0, None), doc="Plasmodesmatal diffusivity [cm²/d]")
    SIGMA = param.Number(default=0.99,  bounds=(0, 1),    doc="Staverman reflection coefficient")
    theta = param.Number(default=0.5,   bounds=(0, 1),    doc="Crank-Nicolson θ (0=explicit, 1=implicit)")

    # ── Solve results ────────────────────────────────────────────────────────
    sol_W = param.Parameter(default=None, doc="Hydraulic solution array [hPa]")
    c_sol = param.Parameter(default=None, doc="Solute concentration solution [mol/cm³]")

    # ── UI flags ─────────────────────────────────────────────────────────────
    geometry_built = param.Boolean(default=False)
    solve_done     = param.Boolean(default=False)
    status         = param.String(default="Ready.")

    # ── Convenience helpers ──────────────────────────────────────────────────

    @property
    def nwj(self):
        return self.mecha.network.n_wall_junction if self.mecha else 0

    @property
    def n_walls(self):
        return self.mecha.network.n_walls if self.mecha else 0

    @property
    def n_cells(self):
        return self.mecha.network.n_cells if self.mecha else 0

    @property
    def n_total(self):
        return self.nwj + self.n_cells

    def cgroup_of(self, node_id: int) -> int:
        """Return cgroup integer for a full-network node_id."""
        if self.mecha is None:
            return -1
        return int(self.mecha.network.graph.nodes[node_id].get("cgroup", -1))

    def set_bc_water(self, node_id: int, value: float | None):
        bc = dict(self.bc_water)
        if value is None:
            bc.pop(node_id, None)
        else:
            bc[node_id] = float(value)
        self.bc_water = bc

    def set_bc_solute(self, node_id: int, value: float | None):
        bc = dict(self.bc_solute)
        if value is None:
            bc.pop(node_id, None)
        else:
            bc[node_id] = float(value)
        self.bc_solute = bc

    def set_ic(self, node_id: int, value: float):
        if self.ic_solute is None:
            self.ic_solute = np.zeros(self.n_total)
        arr = self.ic_solute.copy()
        arr[node_id] = float(value)
        self.ic_solute = arr
