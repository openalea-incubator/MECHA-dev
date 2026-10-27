# The parameters in plain words

MECHA's main parameters, with their default values. MECHA works in centimetres, hectopascals and days:
`openalea.mecha.simulation.parameters()` lists every parameter with its value, its unit and its description, and
{doc}`components` gives the complete reference.

| parameter | what it controls | default |
|---|---|---|
| `kw` | how easily water flows along the cell walls (apoplastic path) | 2.4 × 10⁻⁴ cm² hPa⁻¹ d⁻¹ |
| `kmb` | how easily water crosses a membrane without aquaporins | 3 × 10⁻⁵ cm hPa⁻¹ d⁻¹ |
| `kaqp` | what the aquaporins (water channels) add to a membrane's conductivity | 4.3 × 10⁻⁴ cm hPa⁻¹ d⁻¹ |
| `kpl` | the conductance of one plasmodesma (the channels joining neighbouring cells) | 5.3 × 10⁻¹² cm³ hPa⁻¹ d⁻¹ |
| `kw_casparian` | the walls' conductivity across a Casparian strip (nearly 0: a barrier) | 10⁻¹⁶ cm² hPa⁻¹ d⁻¹ |
| `kw_suberin_in`, `kw_suberin_out` | the same across suberized walls | 10⁻¹⁶ cm² hPa⁻¹ d⁻¹ |
| `thickness` | the cell walls' thickness when one value is used for all (`thickness_source` 1; by default each GRANAP wall has its own) | 1.5 µm |
| `kaqp_endo_factor`, `kaqp_exo_factor`, `kaqp_cortex_factor`, `kaqp_stele_factor` | a tissue's aquaporin activity relative to `kaqp` | 1 |
| `ratio_cortex` | the cortex's aquaporin activity at its innermost walls over its outermost | 1 (uniform) |

The barriers' stage is not a parameter but an input of each root segment, `barrier` (MECHA's maturity code): 1 an
endodermal Casparian strip, 2 the endodermis suberized with passage cells, 3 fully suberized, 4–9 the exodermis'
strip and suberin in various combinations, 10 the exodermis suberized with its strip, 11 both endodermis and
exodermis suberized and the sclerenchyma's (fibres') walls lignified, 0 none (a root tip whose xylem does not
conduct yet).

The radial conductivity kr is given in cm hPa⁻¹ d⁻¹; 1 cm hPa⁻¹ d⁻¹ is 1/864 m s⁻¹ MPa⁻¹, the unit of most
publications.

To change a parameter: `simulate({"kaqp": 8e-4})`; the anatomy is GRANAP's, changed by
`simulate(anatomy={"cortex": {"n_layers": 3}})` or `simulate(anatomy={"preset": "for_dicot_root"})`.
