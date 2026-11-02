==========
Components
==========

MECHA's components, rendered from their declarations (their variables by kind, their schedule). The equations are
in :doc:`equations`.

.. metafspm-component:: openalea.mecha.component.MechaHydraulics

.. metafspm-component:: openalea.mecha.solutes.MechaSolutes

.. metafspm-component:: openalea.mecha.osmotic.MechaOsmoticPresets

.. metafspm-component:: openalea.mecha.root_system.RootSystemConductance

The stand-ins for the models MECHA's inputs come from (a root development model, a soil model, a transpiration
model), until a plant model couples the real ones:

.. metafspm-component:: openalea.mecha.stand_ins.Development

.. metafspm-component:: openalea.mecha.stand_ins.UniformSoil

.. metafspm-component:: openalea.mecha.stand_ins.Transpiration
