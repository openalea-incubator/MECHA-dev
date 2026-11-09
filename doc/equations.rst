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

:math:`\eta` in hPa d, the parameters ``xylem_viscosity`` and ``phloem_viscosity`` (given in hPa s): by default
water at 10 °C (1.3076 × 10⁻³ Pa s) in the xylem, phloem sap (1.7 × 10⁻³ Pa s, Ross-Elliott et al. 2017) in the
phloem; :math:`\pi` is written 3.141592, as in the legacy.

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

   \sum_j Q_{ij} + \sum_{\text{wall–air}} Q_{w \to a} + e_i = \sum_{\text{boundaries}} q_i,

with the boundary inflows :math:`q_i`: a Robin term :math:`K_b(\psi_b - \psi_i)`, a Dirichlet potential, or a Neumann
flow; and :math:`e_i` the water an elongating node withdraws (``conductance.elongation_sink``, sections without
barriers, :math:`b = 0`: the elongation zone; ``Mecha.elongation_BC``):

.. math::

   e_w = \ell_w\,\frac{t_w}{2}\,10^{-8}\,v(x_w)\,\theta_\text{apo}, \qquad
   e_c = \max\!\Big(0,\ A_c - P_c\,\frac{\bar t_c}{2}\Big)\,10^{-8}\,v(x_c)\,\theta_\text{sym},
   \qquad v(x) = v_0 + (x - 0.5)\,\Delta v,

for a wall of length :math:`\ell_w` and thickness :math:`t_w`, a protoplast of area :math:`A_c`, perimeter :math:`P_c`
(its walls' lengths) and mean wall thickness :math:`\bar t_c`; :math:`x` the node's relative x in its section
(``relative_x``), :math:`v_0` ``elongation_midpoint_rate`` and :math:`\Delta v` ``elongation_side_rate_difference``
(cm d⁻¹), :math:`\theta_\text{apo}`, :math:`\theta_\text{sym}` the water fractions ``water_fraction_apoplast``,
``water_fraction_symplast``. The rate is 0 by default (MECHA's ``BCs.xml`` gives 2.8 cm d⁻¹ for maize). The system is linear; it is solved by a Newton step on the sparse Jacobian
:math:`B\,\mathrm{diag}(K)\,B^\top` plus the wall–air terms (:math:`B` the incidence matrix).

**The standard problem** (``_standard_problem``, MECHA's definition of :math:`k_r`): each root section alone, the axial
junctions without conductance; Robin terms with the soil (:math:`K_s`, ``standard_psi_soil``; with
``standard_psi_soil_right``, :math:`\psi_s = \psi_\text{left}(1 - x) + \psi_\text{right}\,x` at the node's relative
x, the legacy's scenarios' left and right potentials) and with the axial boundary (:math:`K_{ax}`,
``standard_psi_xylem`` at :math:`b > 0`, ``standard_psi_sieve`` at :math:`b = 0`, none where that potential is NaN);
at :math:`b = 0` (the tip zone, its xylem immature) the stele's boundary is the protophloem, as in the original MECHA:
its protosieve cells (cgroup 23, tissue ``protosieve`` or ``protophloem``), or, in an anatomy typing none (GRANAP's,
whose phloem poles are ``phloem`` or ``sieve element`` cells), its sieve tubes, the first conduits to mature in the
root tip (``SectionGeometry.protophloem_cells``), at ``standard_psi_sieve`` = 1.1 × 10⁴ hPa by default (the original's
``BCs.xml``: barley's phloem pressure, Pritchard 1996; NaN, the legacy's default without that file, leaves the
section without an axial boundary, its :math:`k_r` undefined), or with ``standard_sieve_flow`` in its place;
at :math:`b > 0`, with ``standard_psi_sieve_tube``, a Robin term between the sieve tubes and the phloem
(``conductance.sieve_tube_terms``: :math:`K_P(A_c, h, \eta_\text{phloem})` or ``k_sieve``, as
``standard_axial_source``). The elongation sink enters with ``standard_elongation`` 1 (the legacy's scenarios ≥ 1; its
scenario 0, MECHA's :math:`k_r`, has none). The uptake is :math:`\sum K_s (\psi_s - \psi)` with each border node's
own :math:`\psi_s`, and :math:`k_r` takes their mean. With ``standard_axial_condition`` 2, the axial boundary is a Neumann flow instead, split among
the vessels as their areas (``_standard_axial_inflow``): :math:`q_v = -Q_\text{std}\,A_v / \sum A`. The osmotic drives
enter only with ``standard_osmotic`` 1.

**The plant problem** (``_plant_problem``): every section's edges and the xylem junctions; the soil as a Robin term
with ``soil_psi`` (an input); at the collar, on the vessels of each root system's first section (``_collar_kind``),
the potential ``collar_psi`` (``collar_condition`` 1) or the flow ``collar_flow`` split as the vessels' Poiseuille
conductances, :math:`q_v = -Q_\text{collar}\,A_v^2 / \sum A^2` (``collar_condition`` 2); each leaf organ's
transpiration :math:`E` as a Neumann outflow at its leaf-surface node, :math:`q = -E` (a leaf organ: any organ
its anatomy gave a leaf surface, whatever its label: GRANAP's leaves, Adel's blades and sheaths alike). The soil
``soil_boundary`` ``"per_node"`` acts on the contact nodes, which a reduced root section eliminates (it keeps its
conduits and its soil node): with reduced roots, the soil is on the soil nodes (``"node_dirichlet"``,
``"node_robin"``), and ``"per_node"`` raises.

**Section height.** A section's height :math:`h` is its segment's length: its walls' and membranes' conductances are
∝ :math:`h`, its conduits' Poiseuille conductances ∝ :math:`1/h`. A section with an anatomy and no length raises
(``_check_heights``): its nodes would be cut off from the network. GRANAP gives no anatomy to a segment without length
(Adel's internodes and sheaths before they grow): a pass-through, the junctions spanning it, until it grows.

**Root apices.** An axis' tip (label ``RootApex``, ArchiSimple's: the part of the root beyond its last segment) is a
root section as a segment is (``ROOT_LABELS``: its soil contact, :math:`k_r`, uptake and perimeter): GRANAP gives it
the root's anatomy (its kind ``root_apex_anatomy``, by default the segments') once it has a length, ``Development``
its barriers by its distance from the tip (none with a tip zone of barrier 0), and a reduction reduces it when its
labels list ``RootApex``. Its meristem is not represented; its maturity is its barrier code's: at barrier 0 (a tip
zone reaching it) its xylem is immature, without xylem junction (D3), and the apex takes up almost nothing (its only
axial path the protophloem's); at barrier 1 it conducts as a young segment of its length.

**Transmembrane fractions** (``_standard_transmembrane_fraction``, MECHA's standard transmembrane fractions): per
membrane, its standard flow from wall to cell over its section's standard uptake, :math:`f_m = Q_{w \to c} / Q_\text{std}`;
none (0) on the membranes of the intercellular spaces and of the mature vessels (:math:`b > 0`). Each membrane's layer
row (``osmotic.membrane_rows``: its cell's rank's row; + 1 on an exodermis membrane facing the epidermis; on an
endodermis membrane facing the cortex + 3, + 2 for a passage cell at :math:`b = 2`; + 1 on a passage cell's facing the
stele at :math:`b = 2`). ``postprocess.transmembrane_layers`` and ``transmembrane_cells`` sum them by row and by cell,
the positive and the negative apart.

**Water potential** (``_water_potential``): :math:`\psi_w = \psi + \pi` on the liquid nodes, and Kelvin's liquid
equivalent :math:`\psi_w = \ln(x)/\alpha` on the air nodes (``_liquid_equivalent``). A section's mean water potential
(a leaf element's, read by a gas exchange's stomata) is the liquid's: weighted by ``liquid_volume``, each node's
volume :math:`V`, 0 on the air nodes and the soil nodes.

A reduced section's kept node stands for the eliminated nodes condensed into it (``_condensed_volumes``): its
``node_volume`` is its own plus a share of theirs, each eliminated node :math:`i` giving its volume to the kept nodes of
a liquid potential :math:`k` (the conduits, the evaporating walls, a root's soil node, whose potential is its contact
walls'; not the air nodes) by the Kron reduction's recovery weights :math:`x_I = W x_K`,
:math:`V_k \mathrel{+}= \sum_i V_i W_{ik} / \sum_{k'} W_{ik'}`; its ``liquid_volume`` is that, 0 on the air nodes.
Each section's volume and liquid volume are the full anatomy's, and a liquid mean on the kept nodes is the full
section's when its eliminated nodes have no sources of their own (the steady state's local gradients aside). The
volumes per unit height are computed once per kind of section (``_condensed_unit_volume``, metafspm's
``@kind_derived``), on the reduction's library while the kind's factorisation is live: the shares
:math:`\sum_i V_i W_{ik} / \sum_{k'} W_{ik'}` from two single solves (``KronReduction.shares``), no dense
:math:`W`.

A reduced section's kind (``plant_reduction``) is its anatomy, its barrier and its length class: the classes
:math:`(1 + w)^k` of relative width ``length_classes`` (:math:`w = 0.1` by default, about 60 from 1 mm to 30 cm, each
evaluated once and shared by the plants), a section of length :math:`h` between the classes :math:`h_k \le h <
h_{k+1}` given their kinds' blend, :math:`D = (1 - t) D_k + t D_{k+1}`, :math:`t = (h - h_k) / (h_{k+1} - h_k)`, and
so its source terms, kr, standard uptake, condensed volumes and recovered interior (metafspm's
``Relative(w, interpolate=True)``). A section's walls' and membranes' conductances are :math:`\propto h`, its
conduits' :math:`\propto 1/h` and its vapour path's not: the reduced term is nearly affine in :math:`h` and the blend
within :math:`w^2/8` of its curvature (a root's uptake and a blade's water potential within :math:`10^{-6}` of the
exact kinds' on a growing seedling, ``test_reduction_accuracy.py``). ``length_classes="exact"`` gives a kind per
length (the exactness tests' reference).

In the frames (metafspm's ``SceneRecorder``, ``display="instances"``, the default), a reduced section's ``psi`` on
every protoplast is not written: each section is an instance of its full template (GRANAP's ``display_instances``)
holding its kept nodes' ``psi``, and each kind's recovery :math:`x = M [x_K; 1]` (the identity on the kept nodes,
:math:`A_{II}^{-1}(-A_{IK} x_K - c_I)` on the interior, a lumped member its node's value) is written once, as its
sparse system :math:`(S, R)`, :math:`M = S^{-1} R` (dense when small); a section between two length classes is the
weighted sum of its two kinds' operators, as its terms are. The reader (encoding3d's ``evaluate``, ParaView's
Select at scale) rebuilds ``psi`` for the sections chosen, equal to the recovered interior of ``display="full"``.

**The shoot's lumped kept nodes** (``plant_reduction(leaf_ports=, vapour_path=)``, scenario keys ``leaf_ports``,
``vapour_path``). A reduced section's dense block has a row per kept node: a blade keeps its conduits and its vapour
path (446 nodes on use case 1's day 8: 118 conduits, 282 evaporating walls, 26 air spaces, 20 pores), a stem its
conduits (198). Two options lump kept nodes into one (metafspm's ``Reduction(merge=)``): a merged group is one node,
its members held at one potential :math:`\psi_G`. With :math:`P` the matrix summing a group's members (a column per
node, a 1 in each member's row), the kind's full matrix and sources are contracted, :math:`A \to P^\top A P`,
:math:`c \to P^\top c`, then Kron-reduced as before; the group's live terms are its representative's (its first
member's) and those of the members' explicit edges, joined to it. It is exact when the members' potentials are equal
in the full section and an approximation otherwise: the potential differences across a group (the lateral gradients
between the vessels of a bundle, or the walls lining an air space) are dropped.

- ``leaf_ports="bundles"`` (default ``"conduits"``): a shoot section's conduits per vascular bundle and family
  (GRANAP's ``pole_id`` and ``family``: its xylem vessels one node, its sieve tubes another; a root section, which has
  a soil node, keeps each conduit). The junctions between two sections then join bundles; the merged conduit's area is
  its members' root sum of squares, :math:`a_G = (\sum_i a_i^2)^{1/2}`, so that its Poiseuille conductance
  (:math:`\propto a^2`) is theirs summed: between two sections of one anatomy, a bundle's junction conducts as its
  vessels' junctions together; where two anatomies meet, the bundles are paired by position as the vessels are.
- ``vapour_path="air_spaces"`` (default ``"cells"``): a leaf's evaporating walls per air space they face (each wall
  faces one, ``air_space_of_wall``): one node, its wall-air exchanges summed on it
  (:math:`Q_{G \to a} = \sum_{w \in G} G_w e_\text{ref} \alpha\,\psi_G - G_w x_a + G_w e_\text{ref}(1 -
  \alpha\psi_\text{ref})`, the walls' linearised Kelvin terms at their common potential). The air spaces and the
  pores stay nodes.

On the growing seedling of ``test_reduction_accuracy.py`` (3 days), the two together keep 107 nodes in its stem and
sheath (447 each conduit and wall a node) and 107 in its blade (434), the roots' 26 unchanged; the roots' uptake
within 0.03 % of the transpiration and the blade's water potential within 0.2 % of its drop below the soil
(``leaf_ports`` alone 0.19 %, ``vapour_path`` alone 0.17 %).
On metafspm's use case 1 (2 plants, day 8), ``leaf_ports="bundles"`` cuts the dense blocks' nonzeros from 1.96M to
1.09M and the plant solve by a third, the transpiration, uptake and gas exchange within 0.04 %, the leaves' water
potential 0.45 % above (0.8 % of its drop below the roots'); both options together to 0.30M (the solve 60 % less),
the fluxes within 0.2 %, the leaves' water potential 3 % above: the lumped walls' common potential stands for walls
whose potentials differ. At 16 plants (day 8; measured on 2026-11-06 with the 4-process pool of its time),
``leaf_ports="bundles"`` takes the nodes from 113k to 99k and the dense nonzeros from 15.7M to 8.8M: MECHA's time per
step 34 % less (2.18 to 1.44 s), the transpiration and uptake unchanged, the stomatal conductance within 0.04 %, the
leaves' water potential 0.36 % above on average, 0.54 % at most. metafspm's process pool was removed since (its QG-3,
2026-11-08): a reduction's kinds are now evaluated in the run's threads. The bundles' comparison was not measured
again without it. With the default options at 16 plants, day 8, ``threads="auto"`` (2026-11-08, metafspm
``86a4a60``), the whole scene's step took 3.26 s on the last day without the pool, against 2.96 s with it, at a
memory peak of 3.68 GB against 4.60 GB.

Both are options, not recommendations: the defaults are ``"conduits"`` and ``"cells"``, every conduit and every
evaporating wall its own node, at any scene size. What ``leaf_ports="bundles"`` gives up is each vessel's own
axial path and its own junction to its successor in the next section: a bundle's vessels are held at one
potential and joined bundle to bundle, so the potential differences between the vessels of a bundle and the way
water passes from one vessel into the next section's are not represented. Its measured cost is a bias of the
leaves' water potential, which the stomata read (Tuzet): 0.45 % above at 2 plants, 0.36 % on average and 0.54 %
at most at 16 plants (day 8), 3 % above with ``vapour_path="air_spaces"`` too.


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

:math:`\psi_\text{axial}` the imposed potential (the xylem's at :math:`b > 0`, the protophloem's at :math:`b = 0`:
there the protophloem's pressure exceeds the soil's, the water leaves the section, :math:`U < 0` and :math:`k_r > 0`,
the original MECHA's ``kr_tot`` of its elongation zone), or the conduits' mean potential with a flow condition (the
vessels', the protophloem's cells'). From the plant problem, the effective one:
:math:`k_{r,\text{effective}} = U / \big((\bar\psi_\text{soil} - \bar\psi_\text{xylem})\,P\,h\,10^{-4}\big)`, the means
over the section's soil-contact nodes and vessels (NaN at :math:`b = 0`: the plant problem gives the tip zone no
stele boundary, its xylem immature and its protophloem's potential not imposed, so that it takes up almost nothing,
:math:`U \approx 0` with :math:`\bar\psi_\text{soil} - \bar\psi \approx 0`). :math:`k_r`, the uptake and :math:`P` are root
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

**Advection** by the water's flow :math:`F = Q` (the plant problem's; the standard problem's with ``solute_flows`` 2,
the legacy's ``standard_solute_flux``), :math:`F = (1 - \sigma)\,Q` on the membranes,
with the Scharfetter–Gummel scheme (``edge_weights``): with :math:`P = F / D_e` and :math:`B(x) = x/(e^x - 1)`
(``bernoulli``), the solute's flow from tail to head is

.. math::

   J = D_e\,\big(B(-P)\,c_\text{tail} - B(P)\,c_\text{head}\big),

upwind (:math:`J = \max(F,0)\,c_\text{tail} + \min(F,0)\,c_\text{head}`) where :math:`D_e = 0`. With
``solute_scheme`` 2, the upwind scheme on every edge (the reactions' default): :math:`J = D_e (c_\text{tail} -
c_\text{head}) + \max(F,0)\,c_\text{tail} + \min(F,0)\,c_\text{head}`.

**Pathways** (``solute_pathway``; the reactions' modes): 1 the whole liquid network; 2 the apoplast only (the walls and
junctions, along the wall edges); 3 the symplast only (the protoplasts, through the plasmodesmata). The other nodes
keep their concentration; the liquid boundary below is still the water's (the whole liquid network's), so that the
water crossing a membrane out of the pathway leaves its solutes behind.

**Balance** (``_transport``), implicit Euler over :math:`\Delta t` (days) as written below, :math:`V` the node's volume
(``_node_volumes``: a wall :math:`\ell_w h t`, a junction :math:`\ell_j h t / 2`, a cell :math:`A_c h`, × 10⁻¹²);
with ``solute_theta`` :math:`\theta < 1` (0.5: Crank–Nicolson), every term but the time derivatives and the sources is
:math:`\theta` × its value at the step's end + :math:`(1 - \theta)` × its value at its start, at the step's flows:

.. math::

   C V\frac{c - c^{\text{prev}}}{\Delta t} + \sum J + k_\text{syn} V c - k_\text{deg} V s + k_\text{decay} V c + O\,c
   = S + I\,c_b,
   \qquad V\frac{s - s^{\text{prev}}}{\Delta t} - k_\text{syn} V c + k_\text{deg} V s = 0,

the reaction in the cells only, :math:`C` the storage fraction (``storage_fraction_wall`` on the walls and junctions,
``storage_fraction_cell`` on the protoplasts; 1 by default), :math:`k_\text{decay}` sucrose's first-order loss
(``k_decay``, every liquid node; 0 by default), :math:`S` the source ``sucrose_source`` (mol d⁻¹), the sinks (the phloem-loading cells:
Strasburger cells, else the phloem; and the xylem) held at ``c_sink``. :math:`O` and :math:`I` (cm³ d⁻¹, ≥ 0;
``liquid_boundary_exchange``) are the water the node gives the boundary and takes from it, with
``solute_boundary_advection`` 1 (develop's ``conserve_boundary``; 0 none): minus and plus its net outflow
:math:`\sum_e F_e` along the liquid edges and the vapour edges. The boundary is what the section's liquid network
exchanges with its outside: the soil, the axial boundaries, its axial junctions (the sections are solved apart);
water leaving as vapour is an edge of that sum, so it leaves its solute behind. :math:`c_b` is ``c_boundary``. After the step, the cells' osmotic potential
(van 't Hoff, ``_sucrose_osmotic_potential``) is :math:`\pi_c = -R\,T\,c` with :math:`R = 8.314 \times 10^4` hPa cm³
mol⁻¹ K⁻¹, which the next water solve reads.

**The coupled steady state** (``solute_solve`` 2, ``_steady_coupled``; the reactions' ``coupled_water_solute_solve``):
each root section's standard problem with the osmotic drives of :math:`\pi_c = \pi_{c,0} - R T c` on its cells
(:math:`\pi_0` the presets' potential without the sucrose's, ``osmotic_baseline``), and the steady sucrose on that
water's flows,

.. math::

   B\,Q(\psi, c) + e = q_b, \qquad \sum J(Q, c) + k_\text{decay} V c + O(Q)\,c = S + I(Q)\,c_b,

one Newton system in :math:`(\psi, c)` (``coupled_psi``, ``sucrose``; no starch), from the standard problem's
:math:`\psi`. Its Jacobian is analytic: :math:`\partial Q/\partial\psi = \mathrm{diag}(K) B^\top`,
:math:`\partial Q/\partial c = -\mathrm{diag}(K\sigma) B^\top \mathrm{diag}(RT)`; the weights' derivatives in the
advective flow (``weight_derivatives``: :math:`-B'(-P)`, :math:`B'(P)` with ``bernoulli_derivative``; upwind, the
indicators) and the boundary exchange's. Each balance is divided by its node's total conductance (``_row_scales``),
so that the tolerance bounds the potentials (hPa) and the concentrations (µmol cm⁻³).
