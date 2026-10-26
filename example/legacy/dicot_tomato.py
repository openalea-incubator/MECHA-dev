"""Build a tomato-like dicot root from the ``for_root`` preset and plot it.

Configuration model: an organ is configured through its ``OrganInputData`` and
then **built once** — construction snapshots the params and parses the vascular
geometry. So all tuning is applied to ``data`` *before* the organ is built, using
``data.set_value(...)`` for existing entries and ``data.params.append(...)`` for
new tissue layers.

Tomato (Solanum lycopersicum) root anatomy
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

SEED = 0


def anatomy_metrics(root: RootAnatomy) -> dict:
    """Geometry summary of a generated root: root/stele diameters and total
    xylem (metaxylem + protoxylem) cross-sectional area.

    Diameters are area-equivalent (``2 * sqrt(area / pi)``), robust to
    non-circular outlines.

    Stele note: ``generate_layer_polygons`` stores each ring at its cell
    *centre-line*, and the stele is a stack of nested rings. So neither the first
    "stele" ring (inset by ``stele_cell/2``) nor the enclosing layer's centre-line
    (out by ``pericycle_cell/2``) is the true boundary. The stele<->pericycle
    interface is the enclosing layer's *inner* edge = its centre-line radius minus
    half its cell diameter, which recovers the nominal ``stele.thickness``.
    """
    root_area = root.generate_base_shape().area

    polys = root.generate_layer_polygons()
    i0 = next((i for i, p in enumerate(polys) if p["name"] == "stele"), None)
    if i0 is not None and i0 > 0:
        enclosing = polys[i0 - 1]
        stele_radius = np.sqrt(enclosing["polygon"].area / np.pi) - enclosing["cell_diameter"] / 2
    else:
        stele_radius = float("nan")

    xylem_area = sum(c.area for c in root.all_cells.cells if "xylem" in c.type)
    return {
        "root_diameter":  np.sqrt(root_area / np.pi) * 2,
        "stele_diameter": stele_radius * 2,
        "xylem_area":   xylem_area,
    }



def main(show=True):
    tomato = OrganInputData.for_dicot_root()  # dicot preset (planttype=2)

    # -- Stele size + radial parenchyma gradient ----------------------------
    tomato.set_value("stele", "thickness",            0.116)
    tomato.set_value("stele", "cell_diameter",        0.01)
    tomato.set_value("stele", "cell_diameter_center", 0.030)

    # -- Star xylem (n arms, radial extent, vessel size gradient) -----------
    tomato.set_value("xylem", "n_vascular_peak",     2)
    tomato.set_value("xylem", "radius_valley_side",        0.015)
    tomato.set_value("xylem", "radius_peak_side",        0.058)
    tomato.set_value("xylem", "arc_peak_side",             0.008)
    tomato.set_value("xylem", "arc_valley_side",          0.015)
    tomato.set_value("xylem", "vessel_diameter",     0.0166)
    tomato.set_value("xylem", "vessel_diameter_min", 0.01)
    tomato.set_value("xylem", "pith_radius",         0.0)
    tomato.set_value("xylem", "gradient_inflection", 0.8)
    tomato.set_value("xylem", "gradient_steepness",  5)
    tomato.set_value("xylem", "allow_ellipse", True)

    # -- Primary phloem (between the star arms, near the cambium) -----------
    tomato.set_value("phloem", "sieve_diameter",    0.005)
    tomato.set_value("phloem", "relative_distance", 0.8)
    tomato.set_value("phloem", "cluster_height",    0.013)
    tomato.set_value("phloem", "cluster_width",     0.035)

    # -- Primary cambium ring (valleys first, maturing outward) -------------
    tomato.set_value("cambium", "radius_valley_side",   0.017)
    tomato.set_value("cambium", "radius_peak_side",   0.059)
    tomato.set_value("cambium", "visible_distance", 0.02)
    tomato.set_value("cambium", "arc_peak_side",          0.013)
    tomato.set_value("cambium", "arc_valley_side",       0.016)
    tomato.set_value("cambium", "cell_diameter",    0.005)
    tomato.set_value("cambium", "cell_width",       0.01)

    # -- Endodermis / pericycle ---------------------------------------------
    tomato.set_value("endodermis", "cell_diameter", 0.015)
    tomato.set_value("endodermis", "cell_width",    0.019)
    tomato.set_value("pericycle", "cell_diameter",  0.0141)
    tomato.set_value("pericycle", "cell_width",     0.0086)

    # -- Cortex layers (inner / main / outer) -------------------------------
    tomato.params.append({
        "name": "inner_cortex",
        "cell_diameter": 0.015,
        "cell_width": 0.014,
        "n_layers": 1,
        "shift": 0.5,
        "order": 3.5,
    })
    tomato.set_value("cortex", "cell_diameter", 0.034)
    tomato.set_value("cortex", "cell_width",    0.03)
    tomato.set_value("cortex", "n_layers",      1)
    tomato.params.append({
        "name": "outer_cortex",
        "cell_diameter": 0.0225,
        "cell_width": 0.0225,
        "n_layers": 2,
        "shift": 0.5,
        "order": 4.5,
    })

    # -- Exodermis / epidermis ----------------------------------------------
    tomato.set_value("epidermis", "cell_diameter", 0.027)
    tomato.set_value("epidermis", "cell_width",    0.0255)

    # -- Intercellular spaces across the cortex tissues ---------------------
    tomato.set_value("inter_cellular_spaces", "smoothness", 0.05)
    tomato.set_value("inter_cellular_spaces", "tissue",
                     ["inner_cortex", "cortex", "outer_cortex"])

    # Build once, after all configuration is in place.
    tomato_anatomy = RootAnatomy(tomato, seed=SEED)
    tomato_anatomy.remove_layer("exodermis")  # remove the default exodermis layer
    tomato_anatomy.generate_cells()

    # Merge the inner/outer cortex tags into a single "cortex" tag.
    tomato_anatomy.retag_cells("inner_cortex", "cortex")
    tomato_anatomy.retag_cells("outer_cortex", "cortex")

    # Other anatomy of tomato
    tomato_secondary_1 = OrganInputData.for_dicot_root()  # dicot preset (planttype=2)

    tomato_secondary_1.set_value("secondary_growth", "value", True)
    # -- Stele size + radial parenchyma gradient ----------------------------
    tomato_secondary_1.set_value("stele", "thickness",            0.11)
    tomato_secondary_1.set_value("stele", "cell_diameter",        0.006)
    tomato_secondary_1.set_value("stele", "cell_diameter_center", 0.006)

    # -- Star xylem (n arms, radial extent, vessel size gradient) -----------
    tomato_secondary_1.set_value("xylem", "n_vascular_peak",     2)
    tomato_secondary_1.set_value("xylem", "radius_valley_side",        0.011)
    tomato_secondary_1.set_value("xylem", "radius_peak_side",        0.04)
    tomato_secondary_1.set_value("xylem", "arc_peak_side",             0.008)
    tomato_secondary_1.set_value("xylem", "arc_valley_side",          0.02)
    tomato_secondary_1.set_value("xylem", "vessel_diameter",     0.02)
    tomato_secondary_1.set_value("xylem", "vessel_diameter_min", 0.01)
    tomato_secondary_1.set_value("xylem", "pith_radius",         0.0)
    tomato_secondary_1.set_value("xylem", "gradient_inflection", 0.5)
    tomato_secondary_1.set_value("xylem", "gradient_steepness",  1)
    tomato_secondary_1.set_value("xylem", "allow_ellipse", True)
    # -- Secondary xylem  ---------------------------------------------------
    tomato_secondary_1.set_value("secondary_xylem", "prop_stele", 1)
    tomato_secondary_1.set_value("secondary_xylem", "cell_diameter", 0.006)
    tomato_secondary_1.set_value("secondary_xylem", "cell_width", 0.006)
    tomato_secondary_1.set_value("secondary_xylem", "vessel_diameter", 0.040)
    tomato_secondary_1.set_value("secondary_xylem", "vessel_diameter_sd", 0.005)
    tomato_secondary_1.set_value("secondary_xylem", "vessel_diameter_min", 0.027)
    tomato_secondary_1.set_value("secondary_xylem", "gradient_function", "gaussian")
    tomato_secondary_1.set_value("secondary_xylem", "allow_ellipse", True)
    tomato_secondary_1.set_value("secondary_xylem", "prop_vessel_ring", 0.4)
    tomato_secondary_1.set_value("secondary_xylem", "must_be_adjacent", False)

    # -- Primary phloem (between the star arms, near the cambium) -----------
    tomato_secondary_1.set_value("phloem", "sieve_diameter",    0.005)
    tomato_secondary_1.set_value("phloem", "relative_distance", 1)
    tomato_secondary_1.set_value("phloem", "cluster_height",    0.006)
    tomato_secondary_1.set_value("phloem", "cluster_width",     0.03)
    # -- No secondary phloem for this variant -----------------------------------------
    tomato_secondary_1.remove_param("secondary_phloem")

    # -- Primary cambium ring (valleys first, maturing outward) -------------
    tomato_secondary_1.set_value("cambium", "radius_valley_side", 0.011)
    tomato_secondary_1.set_value("cambium", "radius_peak_side",   0.04)
    tomato_secondary_1.set_value("cambium", "visible_distance", 1)
    tomato_secondary_1.set_value("cambium", "arc_peak_side",    0.008)
    tomato_secondary_1.set_value("cambium", "arc_valley_side",  0.02)
    tomato_secondary_1.set_value("cambium", "cell_diameter",    0.003)
    tomato_secondary_1.set_value("cambium", "cell_width",       0.01)
    # -- Secondary cambium ring  --------------------------------------------
    tomato_secondary_1.set_value("secondary_cambium", "cell_diameter",    0.005)
    tomato_secondary_1.set_value("secondary_cambium", "cell_width",       0.007)
    tomato_secondary_1.set_value("secondary_cambium", "radius_peak_side",   0.055)
    tomato_secondary_1.set_value("secondary_cambium", "radius_valley_side", 0.045)
    tomato_secondary_1.set_value("secondary_cambium", "arc_valley_side", 0.02)
    tomato_secondary_1.set_value("secondary_cambium", "arc_peak_side",   0.02)
    # -- Endodermis / pericycle ---------------------------------------------
    tomato_secondary_1.set_value("endodermis", "cell_diameter", 0.008)
    tomato_secondary_1.set_value("endodermis", "cell_width",    0.014)
    tomato_secondary_1.set_value("pericycle", "cell_diameter",  0.008)
    tomato_secondary_1.set_value("pericycle", "cell_width",     0.008)

    # -- Cortex layers (inner / main / outer) -------------------------------
    tomato_secondary_1.params.append({
        "name": "inner_cortex",
        "cell_diameter": 0.015,
        "cell_width": 0.021,
        "n_layers": 2,
        "shift": 0.5,
        "order": 3.5,
    })
    tomato_secondary_1.set_value("cortex", "cell_diameter", 0.0261)
    tomato_secondary_1.set_value("cortex", "cell_width",    0.028)
    tomato_secondary_1.set_value("cortex", "n_layers",      1)
    tomato_secondary_1.params.append({
        "name": "outer_cortex",
        "cell_diameter": 0.017,
        "cell_width": 0.018,
        "n_layers": 2,
        "shift": 0.5,
        "order": 4.5,
    })

    # -- Exodermis / epidermis ----------------------------------------------
    tomato_secondary_1.set_value("epidermis", "cell_diameter", 0.019)
    tomato_secondary_1.set_value("epidermis", "cell_width",    0.018)

    # -- Intercellular spaces across the cortex tissues ---------------------
    tomato_secondary_1.set_value("inter_cellular_spaces", "smoothness", 0.05)
    tomato_secondary_1.set_value("inter_cellular_spaces", "tissue",
                     ["inner_cortex", "cortex", "outer_cortex"])

    # Build once, after all configuration is in place.
    tomato_secondary_1_anatomy = RootAnatomy(tomato_secondary_1, seed=SEED)
    tomato_secondary_1_anatomy.remove_layer("exodermis")  # remove the default exodermis layer

    tomato_secondary_1_anatomy.generate_cells()

    # Merge the inner/outer cortex tags into a single "cortex" tag.
    tomato_secondary_1_anatomy.retag_cells("inner_cortex", "cortex")
    tomato_secondary_1_anatomy.retag_cells("outer_cortex", "cortex")

    # Other anatomy of tomato (variant 2)
    tomato_secondary_2 = OrganInputData.for_dicot_root()  # dicot preset (planttype=2)

    tomato_secondary_2.set_value("secondary_growth", "value", True)
    # -- Stele size + radial parenchyma gradient ----------------------------
    tomato_secondary_2.set_value("stele", "thickness",            0.61)
    tomato_secondary_2.set_value("stele", "cell_diameter",        0.01)
    tomato_secondary_2.set_value("stele", "cell_diameter_center", 0.01)

    # -- Star xylem (n arms, radial extent, vessel size gradient) -----------
    tomato_secondary_2.set_value("xylem", "n_vascular_peak",     2)
    tomato_secondary_2.set_value("xylem", "radius_valley_side",  0.010)
    tomato_secondary_2.set_value("xylem", "radius_peak_side",    0.04)
    tomato_secondary_2.set_value("xylem", "arc_peak_side",       0.008)
    tomato_secondary_2.set_value("xylem", "arc_valley_side",     0.02)
    tomato_secondary_2.set_value("xylem", "vessel_diameter",     0.018)
    tomato_secondary_2.set_value("xylem", "vessel_diameter_min", 0.009)
    tomato_secondary_2.set_value("xylem", "pith_radius",         0.0)
    tomato_secondary_2.set_value("xylem", "gradient_inflection", 0.5)
    tomato_secondary_2.set_value("xylem", "gradient_steepness",  1)
    tomato_secondary_2.set_value("xylem", "allow_ellipse", True)
    # -- Secondary xylem  ---------------------------------------------------
    tomato_secondary_2.set_value("secondary_xylem", "prop_stele", 1)
    tomato_secondary_2.set_value("secondary_xylem", "cell_diameter", 0.008)
    tomato_secondary_2.set_value("secondary_xylem", "cell_width", 0.006)
    tomato_secondary_2.set_value("secondary_xylem", "vessel_diameter", 0.06)
    tomato_secondary_2.set_value("secondary_xylem", "vessel_diameter_min", 0.03)
    tomato_secondary_2.set_value("secondary_xylem", "gradient_function", "uniform")
    tomato_secondary_2.set_value("secondary_xylem", "allow_ellipse", True)
    tomato_secondary_2.set_value("secondary_xylem", "prop_vessel_ring", 0.35)
    tomato_secondary_2.set_value("secondary_xylem", "must_be_adjacent", False)

    # -- No primary phloem for this variant ----------------------------------
    tomato_secondary_2.remove_param("phloem")

    # Secondary Phloem (between the star arms, near the cambium) -----------
    tomato_secondary_2.set_value("secondary_phloem", "shape", "band")
    tomato_secondary_2.set_value("secondary_phloem", "height",    0.06)
    tomato_secondary_2.set_value("secondary_phloem", "alive_distance", 0.06)
    tomato_secondary_2.set_value("secondary_phloem", "sieve_diameter", 0.010)
    tomato_secondary_2.set_value("secondary_phloem", "sieve_diameter_sd", 0.001)
    tomato_secondary_2.set_value("secondary_phloem", "sieve_diameter_min", 0.010)
    tomato_secondary_2.set_value("secondary_phloem", "prop_sieve", 0.1)
    tomato_secondary_2.set_value("secondary_phloem", "companion_diameter", 0.005)
    tomato_secondary_2.set_value("secondary_phloem", "companion_width", 0.003)
    tomato_secondary_2.set_value("secondary_phloem", "parenchyma_diameter", 0.008)
    tomato_secondary_2.set_value("secondary_phloem", "parenchyma_width", 0.008)

    # -- Primary cambium ring (valleys first, maturing outward) -------------
    tomato_secondary_2.set_value("cambium", "radius_valley_side",   0.01)
    tomato_secondary_2.set_value("cambium", "radius_peak_side",   0.04)
    tomato_secondary_2.set_value("cambium", "arc_peak_side",          0.008)
    tomato_secondary_2.set_value("cambium", "arc_valley_side",       0.02)
    tomato_secondary_2.set_value("cambium", "cell_diameter",    0.005)
    tomato_secondary_2.set_value("cambium", "cell_width",       0.01)
    # -- Secondary cambium ring  --------------------------------------------
    n_layers = 3 
    tomato_cell_diameter = 0.005
    tomato_secondary_2.set_value("secondary_cambium", "cell_diameter",    tomato_cell_diameter)
    tomato_secondary_2.set_value("secondary_cambium", "cell_width",       0.01)
    tomato_secondary_2.set_value("secondary_cambium", "radius_peak_side",   0.22 + n_layers * tomato_cell_diameter)
    tomato_secondary_2.set_value("secondary_cambium", "radius_valley_side", 0.22 + n_layers * tomato_cell_diameter)
    tomato_secondary_2.set_value("secondary_cambium", "n_layers",   3)

    # -- Endodermis / pericycle ---------------------------------------------
    tomato_secondary_2.set_value("endodermis", "cell_diameter", 0.015)
    tomato_secondary_2.set_value("endodermis", "cell_width",    0.038)
    tomato_secondary_2.set_value("pericycle", "cell_diameter",  0.01)
    tomato_secondary_2.set_value("pericycle", "cell_width",     0.012)

    # -- Cortex layers (inner / main / outer) -------------------------------
    tomato_secondary_2.params.append({
        "name": "inner_cortex",
        "cell_diameter": 0.02,
        "cell_width": 0.045,
        "n_layers": 2,
        "shift": 0.5,
        "order": 3.5,
    })
    tomato_secondary_2.set_value("cortex", "cell_diameter", 0.052)
    tomato_secondary_2.set_value("cortex", "cell_width",    0.057)
    tomato_secondary_2.set_value("cortex", "n_layers",      2)
    tomato_secondary_2.params.append({
        "name": "outer_cortex",
        "cell_diameter": 0.024,
        "cell_width": 0.035,
        "n_layers": 2,
        "shift": 0.5,
        "order": 4.5,
    })

    # -- Exodermis / epidermis ----------------------------------------------
    tomato_secondary_2.set_value("epidermis", "cell_diameter", 0.021)
    tomato_secondary_2.set_value("epidermis", "cell_width",    0.046)

    # -- Intercellular spaces across the cortex tissues ---------------------
    tomato_secondary_2.set_value("inter_cellular_spaces", "smoothness", 0.05)
    tomato_secondary_2.set_value("inter_cellular_spaces", "tissue",
                     ["inner_cortex", "cortex", "outer_cortex"])

    # Build once, after all configuration is in place.
    tomato_secondary_2_anatomy = RootAnatomy(tomato_secondary_2, seed=SEED)
    tomato_secondary_2_anatomy.remove_layer("exodermis")  # remove the default exodermis layer

    tomato_secondary_2_anatomy.generate_cells()

    # Merge the inner/outer cortex tags into a single "cortex" tag.
    tomato_secondary_2_anatomy.retag_cells("inner_cortex", "cortex")
    tomato_secondary_2_anatomy.retag_cells("outer_cortex", "cortex")

    tomato_roots = []
    tomato_roots.append(tomato_anatomy)
    tomato_roots.append(tomato_secondary_1_anatomy)
    tomato_roots.append(tomato_secondary_2_anatomy)

    n_cols = 3
    n_rows = 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axs_flat = axs.flatten()

    labels = ["Tomato primary", "Tomato secondary (variant 1)", "Tomato secondary (variant 2)"]
    for i, root in enumerate(tomato_roots):
        m = anatomy_metrics(root)
        title = (
            f"{labels[i]}\n"
            f"root d = {m['root_diameter']:.3f}   "
            f"stele d = {m['stele_diameter']:.3f}   "
            f"xylem area = {m['xylem_area']:.4f}"
        )
        root.plot_cells(show=False, ax=axs_flat[i], title=title)
        legend = axs_flat[i].get_legend()
        if legend:
            legend.remove()

    for j in range(len(tomato_roots), len(axs_flat)):
        axs_flat[j].set_visible(False)

    plt.suptitle("Tomato anatomy - Cross section", fontsize=14)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()