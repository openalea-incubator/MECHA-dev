# The parameters in plain words

MECHA's main parameters, with their default values. MECHA works in centimetres, hectopascals and days:
`openalea.mecha.simulation.parameters()` lists every parameter with its value, its unit, its description and its declared bounds, and
{doc}`components` gives the complete reference.

| parameter | what it controls | default |
|---|---|---|
| `kw` | how easily water flows along the cell walls (apoplastic path) | 2.4 × 10⁻⁴ cm² hPa⁻¹ d⁻¹ |
| `kmb` | how easily water crosses a membrane without aquaporins | 3 × 10⁻⁵ cm hPa⁻¹ d⁻¹ |
| `kaqp` | what the aquaporins (water channels) add to a membrane's conductivity | 4.3 × 10⁻⁴ cm hPa⁻¹ d⁻¹ |
| `kpl` | the conductance of one plasmodesma (the channels joining neighbouring cells) | 5.3 × 10⁻¹² cm³ hPa⁻¹ d⁻¹ |
| `kw_casparian` | the walls' conductivity across a Casparian strip (nearly 0: a barrier) | 10⁻¹⁶ cm² hPa⁻¹ d⁻¹ |
| `kw_suberin_in`, `kw_suberin_out` | the same across suberized walls | 10⁻¹⁶ cm² hPa⁻¹ d⁻¹ |
| `sigma_casparian_endodermis`, `sigma_casparian_exodermis` (osmotic presets) | how much a Casparian strip holds the solutes back while water crosses it (its reflection coefficient, 0 to 1): the osmotic difference across the strip then also drives water through it, and the water crossing it carries that much less solute | 0 (none) |
| `kw_casparian_exodermis`, `kw_suberin_exodermis` | the exodermis' own Casparian strip and suberin, when they differ from the endodermis' | the endodermis' (`kw_casparian`; `kw_suberin_out`, `kw_suberin_in`) |
| `suberin_model`, `secondary_wall_thickness`, `tertiary_wall_thickness` | the suberin of the endodermis and the exodermis as a lamella (`suberin_model` 1): a layer of its own thickness on each suberized face, of conductivity `kw_suberin_out`, `kw_suberin_in` or `kw_suberin_radial` (toward the cortex, the stele, the neighbouring endodermal cell), and the endodermis' tertiary thickening behind it; `primary_wall_thickness` the wall between them | 0 (suberin as a conductivity of the wall half next to the membrane); a third of the wall's thickness, 0 µm; the wall's thickness |
| `kw_phi_thickening_fraction`, `kw_sclerenchyma_fraction` | the walls of φ-thickened cortex cells and of lignified sclerenchyma (barrier 11), as a fraction of `kw` (`kw_phi_thickening`, `kw_sclerenchyma`: a value of their own instead) | 0.01, 0.1 |
| `sclerenchyma_dead_fraction`, `sclerenchyma_plasmodesmata_factor` | at barrier 11, the fraction of the sclerenchyma's fibres that are dead (their plasmodesmata closed; drawn, `sclerenchyma_seed`, unless the input `living_cell` gives them) and the plasmodesmata left to the living ones | 0.8, 0.05 |
| `membrane_lipid_permeability`, `membrane_aquaporin_permeability` | the membranes' permeability to sucrose through their lipids and through their aquaporins (× the cell's `aquaporin_factor`), in series with the suberin lamella where there is one: given, they replace the diffusivity `d_membrane` | none (`d_membrane`, 10⁻⁶ cm² d⁻¹) |
| `thickness` | the cell walls' thickness when one value is used for all (`thickness_source` 1; by default each GRANAP wall has its own) | 1.5 µm |
| `kaqp_endo_factor`, `kaqp_exo_factor`, `kaqp_cortex_factor`, `kaqp_stele_factor` | a tissue's aquaporin activity relative to `kaqp` | 1 |
| `ratio_cortex` | the cortex's aquaporin activity at its innermost walls over its outermost | 1 (uniform) |

The barriers' stage is not one of MECHA's parameters but an input of each root segment, `barrier` (MECHA's
maturity code), given by a development model (`Development`, its stand-in, from each segment's distance to its tip): 1 an
endodermal Casparian strip, 2 the endodermis suberized with passage cells, 3 fully suberized, 4–9 the exodermis'
strip and suberin in various combinations, 10 the exodermis suberized with its strip, 11 both endodermis and
exodermis suberized and the sclerenchyma's (fibres') walls lignified, 0 none (a root tip whose xylem does not
conduct yet).

The radial conductivity kr is given in cm hPa⁻¹ d⁻¹; 1 cm hPa⁻¹ d⁻¹ is 1/864 m s⁻¹ MPa⁻¹, the unit of most
publications.

To change a parameter: `simulate({"kaqp": 8e-4})`; the anatomy is GRANAP's, changed by
`simulate(anatomy={"cortex": {"n_layers": 3}})` or `simulate(anatomy={"preset": "for_dicot_root"})`.
