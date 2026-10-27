=========
Equations
=========

The physics of the port, with the functions that compute it. MECHA (Couvreur et al. 2018, *Plant Physiology*,
doi:10.1104/pp.18.01006) computes the water flow across a root cross-section on a network of cells, walls and wall
junctions. The port reproduces the legacy's operations and summation orders first, so that its numbers are the
legacy's; the equations below are what those operations compute. The parameters that select the original MECHA's rules
or the legacy's are given where they change an equation (their defaults are the original's, see :doc:`legacy`).

Notation and units (MECHA's): lengths in µm, areas in µm², conductances :math:`K` in cm³ hPa⁻¹ d⁻¹, potentials in hPa,
flows in cm³ d⁻¹. A section is one segment's cross-section (a SubOrgan); its height :math:`h` is the segment's length.
Its nodes are the protoplasts (cells) :math:`c`, the walls :math:`w` and the wall junctions :math:`j` (GRANAP's
Compartments); its edges are the membranes (wall–cell), the wall edges (wall–junction) and the plasmodesmata
(cell–cell) (GRANAP's Connections); successive sections are joined by axial junctions between their conducting cells.
:math:`t` is a wall's thickness, :math:`\ell_e` an edge's length, :math:`d_e` the distance between its nodes, a cell's
cgroup its MECHA cell group (1 exodermis, 2 epidermis, 3 endodermis, 4 cortex, 5 stele, 11 sieve tube, 12 companion
cell, 13 xylem, 16 pericycle, 22 cambium, 23 protophloem). The barrier code :math:`b` (0–11) is the segment's
maturity: 0 no barrier, 1 an endodermal Casparian strip, 2 suberin with passage cells, 3 a fully suberized endodermis,
4–9 exodermal variants, 10 a suberized exodermis with its Casparian strip (no endodermal strip), 11 the endodermis and
exodermis suberized and the sclerenchyma's walls lignified.


Geometry (``geometry.section_geometry``)
========================================

From GRANAP's columns (mm, mm²), converted to µm and µm², once per distinct anatomy:

- **Lateral distance** of a wall, the sum of its membranes' distances, :math:`L_w = \sum_{m \ni w} d_m`.
- **Wall edges**: a wall–junction edge is kept when its wall has at least two junctions; its length is half the wall's,
  :math:`\ell_e = \ell_w / 2`.
- **Border walls**: walls with one membrane, to an epidermis cell (the soil interface); **border junctions**: junctions
  between two border walls, of length :math:`\ell_j = (\ell_{w_1} + \ell_{w_2})/4`.
- **Gravity centre** :math:`G`: the mean position of the endodermis cells (of all the cells in a section without an
  endodermis: stems and leaves); each wall's distance to it :math:`D_w = |x_w - G|`.
- **Perimeter** (cm): :math:`P = 2\pi\,\bar D_\text{epi}\,10^{-4}`, :math:`\bar D_\text{epi}` the mean of
  :math:`D_w` over the epidermis cells' membranes (a wall counted once per epidermis membrane).
- **Cell ranks** (``_ranks``): the cgroup, then the cortex layers 40–44 counted from the endodermis and 49–45 from the
  outer layer (exodermis, or epidermis without one), and the stele layers 50–60 counted from the pericycle (or the
  endodermis without one), 61 for the rest of the stele; each layer is the cells next to the previous one, and its
  mean distance to :math:`G` is kept.
- **Plasmodesmata concentrations** (``_plasmodesmata_ratios``) at the three cortical interfaces :math:`k` (outer
  layer–cortex, cortex–cortex, cortex–endodermis), for aerenchyma:

  .. math::

     c_k = \frac{\sum_{\text{plasmodesmata at } k \text{ from non-intercellular cells}} \ell_e}
                {\sum_{\text{the same, both ends non-intercellular}} \ell_e} \;\ge 1,

  which moves the plasmodesmata of the walls lost to intercellular spaces onto the remaining ones (applied at
  :math:`b > 0`).


Conductances (``conductance.edge_conductances``)
================================================

**Walls** (wall–junction edges):

.. math::

   K_w = k_w \, 10^{-4}\, \frac{(L_w + h)\,t - t^2}{\ell_e},

with :math:`k_w` (cm² hPa⁻¹ d⁻¹) chosen by rules (``wall_configuration``, ``_select``): the interface of the wall's
two cells (exodermis, epidermis–exodermis, exodermis–cortex, cortex–cortex, endodermis, endodermis–cortex,
endodermis–stele), each of which the barrier code sets to ``kw`` or to a barrier value (``kw_casparian``,
``kw_suberin_in``, ``kw_suberin_out``, ``kw_lignin``; the table is in :doc:`legacy`); then ``kw_lignin`` between two
φ-thickened cells of the same rank, ``kw_lignin`` for every wall of a sclerenchyma cell at :math:`b = 11`,
``kw_passage`` next to a passage cell, ``kw_septa`` for a wall with two
intercellular cells at :math:`b > 0` and, with ``xylem_pieces``, :math:`K_w = 10^{-16}` for a wall between two xylem
cells (a vessel's septa). The thickness :math:`t` is each wall's own (``thickness_source`` 2, GRANAP's) or one value
for every wall (``thickness_source`` 1, ``thickness``).

**Membranes** (wall–cell edges), the wall's half thickness in series with the membrane:

.. math::

   K_m = \left( \frac{t/2 \cdot 10^{-4}}{k_w} + \frac{1}{k_{mb} + k_{aqp}} \right)^{-1} 10^{-8}\,(h + d_e)\,\ell_e,

:math:`k_w` the wall's interface value (``kw_passage`` at an endodermis wall next to a passage cell),
:math:`k_{mb}` the membrane's conductivity without aquaporins (cm hPa⁻¹ d⁻¹), :math:`k_{aqp}` the aquaporins' by the
cell: ``kaqp`` × its tissue factor (exodermis, epidermis, endodermis, stele), × 10⁴ in a mature xylem vessel
(:math:`b > 0`), ``k_interc`` in an intercellular space at :math:`b > 0`, the cortex gradient below in the cortex.
:math:`K_m = 0` where :math:`k_w = 0`, and :math:`K_m = 10^{-16}` where :math:`k_{aqp} = 0` and the wall is not a
barrier interface (two endodermis cells, two exodermis cells, or a stele and an endodermis cell): an air-filled
intercellular space has no water-conducting membrane.

**The cortex gradient** (``conductance.cortex_parameters``): with ``ratio_cortex`` :math:`r \ne 1`,

.. math::

   k_{aqp}(D_w) = a\,(D_w + s)\,10^{-4}, \qquad s = \frac{r D_\text{max} - D_\text{min}}{1 - r}, \qquad
   a = k_{aqp,\text{cortex}} \frac{\sum_k S_k}{\sum_k S_k\,10^{-4}(D_k + s)},

the sums over the cortex cells' walls :math:`k` (:math:`S_k = (h + \delta_k)\,\ell_k\,10^{-8}`, :math:`\delta_k` the
wall's distance to its cell's centre), :math:`D_\text{min}`, :math:`D_\text{max}` the cortex walls' extreme distances
to :math:`G`. The surface-weighted mean is :math:`k_{aqp,\text{cortex}}`, and
:math:`k_{aqp}(D_\text{min}) / k_{aqp}(D_\text{max}) = r`: ``ratio_cortex`` is the activity at the innermost cortex
walls over the outermost (the original's ``Mecha._calculate_cortex_parameters`` and the port alike; the original's
input describes it only as a "ratio related to cortex hydraulic properties"). With :math:`r = 1`,
:math:`k_{aqp} = k_{aqp,\text{cortex}}`.

**Plasmodesmata** (cell–cell edges): :math:`K_p = k_{pl}\,N_p`, ``kpl`` the conductance of one plasmodesma, and their
quantity

.. math::

   N_p = F_{ij}\,10^{-4}\,\ell_e \cdot \alpha_{ij} \cdot c_k \cdot f_\text{endo} \cdot s_{ij},

:math:`F_{ij}` the interface's frequency × height (cm⁻¹, ``fplxheight_*``), :math:`\alpha_{ij}` its aperture factor,
:math:`s_{ij}` the aperture by the distance to the soil contact (``pd_spatial_profile``; 1 by default, and between two
xylem cells): with :math:`d` the plasmodesma's mean end distance to the section's soil-contact walls and junctions
along its graph (Euclidean edge lengths: ``contact_distance``), :math:`u = (d - d_\min)/(d_\max - d_\min)` over the
section's plasmodesmata, :math:`s = c_w - (c_w - c_d)\,u` (profile 1) or :math:`c_d + (c_w - c_d)\,e^{-(d -
d_\min)/L}` (profile 2), :math:`c_w`, :math:`c_d`, :math:`L` the parameters ``pd_spatial_c_wet``, ``pd_spatial_c_dry``,
``pd_spatial_length_scale`` (NaN: a third of the range), :math:`c_d` where not connected;
:math:`c_k` the concentration above at the cortical interfaces. With ``plasmodesmata_rules`` 1 (the original MECHA's,
``_plasmodesmata_original``), first match wins: none at an intercellular space (:math:`b > 0`), ``10000`` ×
``fplxheight_default`` between the pieces of one vessel (``xylem_pieces``), none to mature xylem; the factors
(``kpl_cortex_factor``, ``kpl_PPP_factor`` on the phloem-pole pericycle, ``kpl_PCC_factor``) enter as
:math:`\alpha = f` between two cells of the factor's type and as the harmonic mean :math:`2 f_1 f_2 / (f_1 + f_2)`
(with 1 for the other side) across interfaces; then :math:`f_\text{endo}` is ``kpl_endo_in_factor`` when both cells are
endodermis or stele (or the section has no xylem), ``kpl_endo_out_factor`` otherwise. With ``plasmodesmata_rules`` 2
(the legacy's, ``_plasmodesmata_legacy``), the legacy's frequency table, :math:`\alpha = f` or :math:`2f/(f + 1)` from
``develop``'s factor map (``conductance.KPL_FACTOR_INTERFACES``: the cortex's, the endodermis' inner and outer, PPP,
PCC, PST and ``kpl_cambium_stele_factor``, each at its interfaces; doc/legacy.rst, d2), 10000 × the frequency between
any two xylem cells, and no :math:`f_\text{endo}`.

**Soil and axial boundary terms** (``conductance.boundary_terms``), per node: a border wall and a border junction at
:math:`x \ge` ``x_contact``, on root sections only,

.. math::

   K_{s,w} = k_w\,10^{-4}\,\frac{\ell_w/2 \cdot h}{t/2}, \qquad K_{s,j} = k_w\,10^{-4}\,\frac{\ell_j\,h}{t/2}

(a junction's :math:`t` the mean of its walls'); a xylem vessel (:math:`b > 0`) or a protophloem cell (:math:`b = 0`)
exchanges with the axial boundary through :math:`K_{ax} = K_P(A_c, h, \eta)` (``standard_axial_source`` 1, the vessel
over the section's height) or the prescribed ``k_xyl``, ``k_sieve`` (``standard_axial_source`` 2).


Poiseuille's law (``postprocess``)
==================================

A conduit of cross-section :math:`A` over a length :math:`h` (``poiseuille``):

.. math::

   K_P(A, h, \eta) = \frac{A^2}{8\pi\,h\,\eta}\,10^{-12},

:math:`\eta` in hPa d: water at 10 °C (1.3076 × 10⁻³ Pa s) in the xylem, phloem sap (1.7 × 10⁻³ Pa s, Ross-Elliott et
al. 2017) in the phloem; :math:`\pi` is written 3.141592, as in the legacy.

**Axial conductances** of a section (``axial_conductances``), cm⁴ hPa⁻¹ d⁻¹:

.. math::

   K_x = \frac{h}{10^4} \sum_{\text{vessels}} K_P(A_v, h, \eta_\text{xylem}), \qquad
   K_{x,\text{phloem}} = \frac{h}{10^4} \sum_{\text{sieve tubes}} K_P(A_s, h, \eta_\text{phloem}),

with :math:`K_x = 0` at :math:`b = 0` (no mature xylem), where :math:`K_{x,\text{phloem}}` sums the protophloem cells.

**Axial junctions** between two sections (``MechaHydraulics._junction_conductance``): each junction's two cells, each
over half its own segment's length :math:`L`, in series,

.. math::

   \frac{1}{K_\text{junction}} = \frac{1}{K_P(A_a, L_a/2, \eta)} + \frac{1}{K_P(A_b, L_b/2, \eta)},

for the xylem junctions (``_xylem_axial_conductance``, 0 where either section has :math:`b = 0`) and the phloem
junctions (``_phloem_axial_conductance``, not in the water problem).


The water balance (``component.MechaHydraulics``)
==================================================

The unknown is each node's potential :math:`\psi` (the pressure potential; the water potential when there is no
osmotic term). Along an edge from node :math:`i` to node :math:`j`, the flow (``_flow``, ``_standard_flow``) is

.. math::

   Q_{ij} = K_{ij}\left[(\psi_i - \psi_j) + \sigma_{ij}\,(\pi_i - \pi_j)\right],

:math:`\pi` the osmotic potentials, :math:`\sigma` the membranes' reflection coefficients (0 on walls and
plasmodesmata); :math:`\sigma(\pi_i - \pi_j)` is the edge's osmotic drive (``_osmotic_drive``). Each node's balance
(``_balance``) is

.. math::

   \sum_j Q_{ij} + \sum_{\text{wall–air}} Q_{w \to a} = \sum_{\text{boundaries}} q_i,

with the boundary inflows :math:`q_i`: a Robin term :math:`K_b(\psi_b - \psi_i)`, a Dirichlet potential, or a Neumann
flow. The system is linear; it is solved by a Newton step on the sparse Jacobian
:math:`B\,\mathrm{diag}(K)\,B^\top` plus the wall–air terms (:math:`B` the incidence matrix).

**The standard problem** (``_standard_problem``, MECHA's definition of :math:`k_r`): each root section alone, the axial
junctions without conductance; Robin terms with the soil (:math:`K_s`, ``standard_psi_soil``) and with the axial
boundary (:math:`K_{ax}`, ``standard_psi_xylem`` at :math:`b > 0`, ``standard_psi_sieve`` at :math:`b = 0`, none where
that potential is NaN). With ``standard_axial_condition`` 2, the axial boundary is a Neumann flow instead, split among
the vessels as their areas (``_standard_axial_inflow``): :math:`q_v = -Q_\text{std}\,A_v / \sum A`. The osmotic drives
enter only with ``standard_osmotic`` 1.

**The plant problem** (``_plant_problem``): every section's edges and the xylem junctions; the soil as a Robin term
with ``soil_psi`` (an input); at the collar, on the vessels of each root system's first section (``_collar_kind``),
the potential ``collar_psi`` (``collar_condition`` 1) or the flow ``collar_flow`` split as the vessels' Poiseuille
conductances, :math:`q_v = -Q_\text{collar}\,A_v^2 / \sum A^2` (``collar_condition`` 2); each leaf organ's
transpiration :math:`E` as a Neumann outflow at its leaf-surface node, :math:`q = -E`.

**Water potential** (``_water_potential``): :math:`\psi_w = \psi + \pi` on the liquid nodes, and Kelvin's liquid
equivalent :math:`\psi_w = \ln(x)/\alpha` on the air nodes (``_liquid_equivalent``).


The leaves' vapour path (``conductance.vapour_conductances``)
=============================================================

On the air nodes (air spaces, stomatal pores, the leaf surface) the unknown is the vapour pressure relative to
saturation, :math:`x`. Vapour diffusion gives a conductance in liquid water per unit of :math:`x`,

.. math::

   G = \frac{D_v\,A}{d}\,\frac{e_\text{sat}\,V_w}{R\,T}, \qquad
   e_\text{sat} = 610.78 \exp\left(\frac{17.27\,T}{T + 237.3}\right) \text{ Pa},

:math:`D_v` ``vapour_diffusivity``, :math:`T` ``leaf_temperature``, :math:`V_w = 1.8 \times 10^{-5}` m³ mol⁻¹,
:math:`R = 8.314` J mol⁻¹ K⁻¹ (``vapour_factor``; :math:`e_\text{sat}` by Tetens' formula, :math:`T`
in °C). A liquid at :math:`\psi` is in equilibrium with :math:`x = \exp(\alpha\psi)`,
:math:`\alpha = V_w/(RT)` (``kelvin_alpha``).

- **Wall → air space** (``wall_air`` edges): :math:`A` the wall's length × :math:`h`, :math:`d` the distance to the air
  space's centre; Kelvin's law linearised at :math:`\psi_\text{ref}` (``psi_ref_wall_air``),
  :math:`e_\text{ref} = \exp(\alpha\psi_\text{ref})`:

  .. math::

     Q_{w \to a} = G\,e_\text{ref}\,\alpha\,\psi_w - G\,x_a + G\,e_\text{ref}\,(1 - \alpha\psi_\text{ref}).

- **Air ↔ air** (``air_link`` edges: air spaces, a pore and its chamber): :math:`A = h \times` the narrower node's width
  (a pore's aperture width, an air space's :math:`\sqrt{\text{area}}`), :math:`d` the distance between their centres;
  :math:`Q = G\,(x_a - x_b)`.
- **Pore → leaf surface** (stomatal edges): the pore stands for :math:`n = L /` ``stomatal_spacing`` stomata over the
  segment, each a diffusion tube with its end correction (the anatomical maximum, after Brown and Escombe; Franks and
  Beerling 2009), times the aperture fraction ``stomatal_aperture``:

  .. math::

     K_\text{stoma} = n\,f_\text{aperture}\,\frac{D_v\,a}{l + \frac{\pi}{2}\sqrt{a/\pi}}\,\frac{e_\text{sat}\,V_w}{R\,T},
     \qquad a = w_\text{pore}\,l_\text{pore},

  :math:`w_\text{pore}`, :math:`l` the pore's width and depth (GRANAP's), :math:`l_\text{pore}` ``pore_length``. Each
  pore's flow to the surface is the output ``evaporation``.


Section results (``postprocess``, ``MechaHydraulics._kr``)
==========================================================

The uptake (``soil_uptake``) and the radial conductivity (``radial_conductivity``, cm hPa⁻¹ d⁻¹) of the standard
problem:

.. math::

   U = \sum_{\text{border walls, junctions}} K_s\,(\psi_\text{soil} - \psi), \qquad
   k_r = \frac{U}{(\psi_\text{soil} - \psi_\text{axial})\,P\,h\,10^{-4}},

:math:`\psi_\text{axial}` the imposed potential, or the vessels' mean potential with the flow condition. From the plant
problem, the effective one:
:math:`k_{r,\text{effective}} = U / \big((\bar\psi_\text{soil} - \bar\psi_\text{xylem})\,P\,h\,10^{-4}\big)`, the means
over the section's soil-contact nodes and vessels (NaN at :math:`b = 0`). :math:`k_r`, the uptake and :math:`P` are root
quantities (NaN on stems and leaves).

**Pathways** (``plots.pathway_breakdown``): each edge's radial component of :math:`|Q|` assigned to the nearest mean
radius of the cell layers, as shares of the walls, membranes and plasmodesmata.


Osmotic presets (``osmotic``)
=============================

``MechaOsmoticPresets`` writes :math:`\pi` (cells and walls) and :math:`\sigma` (membranes) on root sections
(``presets_sections`` 2: on every section). Per membrane (wall :math:`i`, cell :math:`j`), in the walls' order
(``section_presets``), the cell takes its tissue's value (``tissue_values``: ``os_hetero`` 0 ``os_cortex`` everywhere,
1 and 2 tabulated profiles with cortex layers c1–c8, 3 the cortex at ``os_cortex`` and the stele at −5000 hPa, 4 one
value per tissue; companion cells the mean of the sieve tubes' value and the cortex's, with ``os_hetero`` 0, or the
stele's otherwise), the wall the soil solution's
(cortical side) or the xylem sap's (stelar side, :math:`b > 0`). With the wall's radial position :math:`r` (``relative_positions``:
−1 at the xylem's 80th-percentile distance, 0 at the endodermis, 1 at the epidermis) and its relative :math:`x`:

.. math::

   \pi_\text{soil} = \pi_L (1 - x) + \pi_R\,x \;\; (\text{symmetry 1}), \qquad
   \pi_\text{soil} = \pi_L + (\pi_R - \pi_L)\,|r|^{n_s} \;\; (\text{symmetry 2}),

.. math::

   \pi_\text{xylem} = \pi_\text{endo} + (\pi_\text{xyl} - \pi_\text{endo})\,(1 - |r|)^{n_x} \;\; (\text{symmetry 2}),
   \qquad \pi_\text{xylem} = (\pi_\text{xyl} + \pi_\text{endo})/2 \;\; (\text{symmetry 1}),

and, with a non-zero diffusivity, the flat value (:math:`\pi_L`, :math:`\pi_\text{xyl}`): the legacy's exponential
profile in a convection velocity that is always 0. Intercellular spaces take the soil solution's :math:`\pi` with
:math:`\sigma = 0`, mature vessels the xylem sap's with :math:`\sigma = 0`. :math:`\sigma` is ``s_factor`` × 1, or ×
0.5 on the stelar (``s_hetero`` 1) or cortical (``s_hetero`` 2) membranes. The solutes' :math:`-RTc` is added to the
cells' :math:`\pi` (input ``sucrose_osmotic_potential``).


Solutes (``solutes.MechaSolutes``)
==================================

Sucrose :math:`c` (mobile) and starch :math:`s` (immobile), mol cm⁻³, on the liquid nodes (not the air nodes).

**Diffusion conductance** of an edge (cm³ d⁻¹), from factors computed with the conductances (``edge_conductances``
with ``factors``):

.. math::

   D_e = f_\text{apo}\,D_\text{apo} + f_\text{sym}\,D_\text{pd} + f_\text{mem}\,D_\text{mem},

:math:`f_\text{apo}` a wall's :math:`10^{-4}((L_w + h)t - t^2)/\ell_e \times k_w/\text{kw}` (the barriers block
solutes as they block water; a vessel's septa :math:`10^{-16}`), or a mature vessel's membrane
:math:`10^{-4}\,\ell_w h / t`; :math:`f_\text{sym} = a_\text{pd}\,N_p / t \cdot 10^{-4}` (``pd_section``, the
plasmodesmata's quantity and aperture as above, :math:`t` the thickness of the wall they cross: its own with
``thickness_source`` 2, ``thickness`` with 1); :math:`f_\text{mem} = \ell_e h \, 10^{-4} / d_e` on every membrane; :math:`D_\text{apo}`, :math:`D_\text{pd}`, :math:`D_\text{mem}` the diffusivities
(cm² d⁻¹).

**Advection** by the water's flow :math:`F = Q` (the plant problem's), :math:`F = (1 - \sigma)\,Q` on the membranes,
with the Scharfetter–Gummel scheme (``edge_weights``): with :math:`P = F / D_e` and :math:`B(x) = x/(e^x - 1)`
(``bernoulli``), the solute's flow from tail to head is

.. math::

   J = D_e\,\big(B(-P)\,c_\text{tail} - B(P)\,c_\text{head}\big),

upwind (:math:`J = \max(F,0)\,c_\text{tail} + \min(F,0)\,c_\text{head}`) where :math:`D_e = 0`.

**Balance** (``_transport``), implicit Euler over :math:`\Delta t` (days), :math:`V` the node's volume
(``_node_volumes``: a wall :math:`\ell_w h t`, a junction :math:`\ell_j h t / 2`, a cell :math:`A_c h`, × 10⁻¹²):

.. math::

   V\frac{c - c^{\text{prev}}}{\Delta t} + \sum J + k_\text{syn} V c - k_\text{deg} V s = S, \qquad
   V\frac{s - s^{\text{prev}}}{\Delta t} - k_\text{syn} V c + k_\text{deg} V s = 0,

the reaction in the cells only, :math:`S` the source ``sucrose_source`` (mol d⁻¹), the sinks (the phloem-loading cells:
Strasburger cells, else the phloem; and the xylem) held at ``c_sink``. After the step, the cells' osmotic potential
(van 't Hoff, ``_sucrose_osmotic_potential``) is :math:`\pi_c = -R\,T\,c` with :math:`R = 8.314 \times 10^4` hPa cm³
mol⁻¹ K⁻¹, which the next water solve reads.
