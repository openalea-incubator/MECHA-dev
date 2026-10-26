=====
MECHA
=====

MECHA computes the water flows across a plant organ's cross-section, cell by cell: through the walls, the membranes
and the plasmodesmata, between the soil (or the air) and the vessels, with the barriers of the endodermis and the
exodermis, and the osmotic and solute terms. This version is its port to metafspm components on GRANAP's anatomies:
``MechaHydraulics`` (the hydraulics of the sections and of the whole plant), ``MechaSolutes`` (solutes, diffusion and
reactions), ``MechaOsmoticPresets`` (the osmotic presets).

The tutorials are executed notebooks in ``example/tutorials/`` (start with GRANAP's cross-section tutorial for the
anatomies): GRANAP and MECHA coupled in a plant model; a soil gradient around a root section; a pine needle's water
and sugars over a day; a whole plant in its soil. The README gives the installation.

.. toctree::
    :maxdepth: 1

    components
    equations
    legacy
    performance
