# Transpiration Implementation Plan in Needles

The `mecha` library is designed to model root section hydraulics. To simulate leaves and needles structures and functions, we need to adapt the boundary conditions, tissue barriers, and solver options. 

Below is the phased implementation plan for simulating transpiration in needles.

---

## Phase 1: Support New Maturity Stage for Needle Barriers (Stage 10)
- **Objective**: Define a new maturity stage `10` representing Casparian Strip on endodermis, plus lignified epidermis and hypodermis.
- **Tasks**:
  1. Add support for maturity stage index/value `10` in geometry configuration loading.
  2. Implement specific hydraulic conductivity parameters for stage `10` (e.g. suberised endodermis, lignified outer layers).
  3. Verify that the maturity stage parses correctly in `test_transpiration.py` via `default_input.geometry.set_maturity_stages([10])`.

## Phase 2: Set Boundary Water Potential in Air Space
- **Objective**: Target intercellular/air space for boundary conditions instead of the epidermis.
- **Tasks**:
  1. Allow setting boundary conditions in the internal air spaces (intercellular nodes) rather than the standard epidermis border.
  2. Utilize a custom flag to indicate that boundary contacts lie within the air space.
  3. Ensure that boundary water potentials are correctly initialized for air space similarly to xylem water potentials but lower and relatively to the distance from the stomata.

## Phase 3: Exclude Epidermis / Border Walls from Boundary Conditions
- **Objective**: Ensure that no boundary condition is assigned to the epidermis/is_border=True edges/walls.
- **Tasks**:
  0. Create a new method `_apply_transpiration_boundary` in `src/mecha/hydraulic_solver.py` to apply boundary conditions to the air space. new `rhs_a` (air)
  1. Modify `_apply_soil_boundary` in `src/mecha/hydraulic_solver.py` to check if transpiration mode or needle mode is enabled.
  2. If modeling transpiration in needles, bypass assigning the standard soil boundary conditions to the external border walls and junctions (where `is_border=True` or `wall_id in self.network.border_walls`).

## Phase 4: Debug and Resolve Solver Errors
- **Objective**: Fix the `ValueError: cannot convert float NaN to integer` during water flux calculations.
- **Tasks**:
  1. Inspect `standard_transmembrane_fractions` in `src/mecha/mecha_class.py` where:
     ```python
     rank = int(self.network.cell_ranks[j-self.network.n_wall_junction])
     row = int(self.network.rank_to_row[rank])
     ```
  2. Handle cases where cell ranks or row mappings are NaN or missing for needle anatomies (which do not share the exact same concentric layer structure as roots).
  3. Provide fallback or default integer conversions when converting ranks/rows to integer.

---

## Work Summary

The transpiration modeling implementation was successfully completed with the following updates:
1. **Maturity Stage 10 support**: Added to `get_wall_conductivities` in `data_loader.py` to represent the needle barriers with a Casparian strip on the endodermis, and lignified epidermis and hypodermis.
2. **Air Space & Transpiration Boundary**:
   - Added `_apply_transpiration_boundary` to map boundary conditions to intercellular air space cells instead of external border walls/junctions.
   - Evaluated distances from air cells to stomata/pores and applied an exponential potential decay scaling from stomatal boundaries.
   - Properly integrated `psi_air` / `psi_left_soil` lookup dynamically in `hydraulic_solver.py` and `mecha_class.py` when in transpiration mode.
3. **Solver Stability**:
   - Resolved the concentric layers `NaN` to integer conversion crash (`ValueError`) by adding guards in `standard_transmembrane_fractions` and `initialize_scenarios` within `mecha_class.py`.
4. **Interface Flows**:
   - Modified `_calculate_interface_flows` to properly sum transpiration fluxes from air spaces instead of traditional border nodes in transpiration mode.
