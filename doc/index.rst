=====
MECHA
=====

MECHA computes how water crosses a plant organ, cell by cell: from the soil (or the air) to the vessels, along the
cell walls, across the membranes and through the plasmodesmata, with the barriers of the endodermis and the
exodermis. This version is its port to metafspm components, on GRANAP's anatomies: ``MechaHydraulics`` (the hydraulics
of the sections and of the whole plant), ``MechaSolutes`` (solutes, diffusion and reactions), ``MechaOsmoticPresets``
(the osmotic presets). The README gives the installation.

User guide
==========

For newcomers to the model, to plant hydraulics or to Python modelling: solve a root section, change its parameters,
explore them, use experimental data, follow the water along a root. Each tutorial is a notebook in
``example/tutorials/user/``, shown here with its results.

.. toctree::
    :maxdepth: 1

    notebooks/user/01_first_run
    notebooks/user/02_change_a_parameter
    notebooks/user/03_combine_parameters
    notebooks/user/04_sensitivity_analysis
    notebooks/user/05_experimental_data
    notebooks/user/06_along_a_root
    parameters

Developer guide
===============

For those who read, check, change or couple the model: how it is written as metafspm components on GRANAP's
anatomies, how it was ported and validated against the original MECHA, its equations and its performance. Best read
after the first user tutorial.

.. toctree::
    :maxdepth: 1

    notebooks/developer/00_overview
    notebooks/developer/01_granap_mecha
    notebooks/developer/02_soil_gradient
    notebooks/developer/03_needle_solutes
    notebooks/developer/04_full_plant
    notebooks/developer/05_extend_the_component
    components
    equations
    legacy
    performance
