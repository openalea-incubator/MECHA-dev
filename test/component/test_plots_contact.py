"""
develop's views of a section partly in contact with the soil (openalea.mecha.plots): a root whose border walls and
junctions touch the soil only on the right half (x_contact at its centre), the plasmodesmata's aperture by distance
(pd_spatial_profile 1). The wet→dry axis points from the wet side through the centre; the transect's coordinates are
negative on the wet side; the maps' values span the profile.
"""
import matplotlib
import numpy as np
import pytest

from openalea.mecha import plots
from openalea.mecha.component import MechaHydraulics

from plants import anatomy_data_structure, root

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def frame():
    g, ds, anatomy = anatomy_data_structure(root(1))
    ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    segment = int(ds.entity_ids("SubOrgan")[0])
    mecha = MechaHydraulics(data_structure=ds)
    mecha.x_contact = float(mecha.section_geometries()[segment][2].gravity_center[0])
    mecha.pd_spatial_profile = 1.
    mecha()
    return plots.port_frame(mecha, segment)


def test_the_wet_dry_axis(frame):
    origin, unit = plots.wetdry_axis(frame)
    assert unit[0] < -0.99                                   # the right half is wet: towards −x
    coordinate, _, _ = plots._coordinate(frame, "signed")
    assert (coordinate(frame.xy[frame.contact]) < 0).mean() > 0.9


def test_the_transect_profiles(frame):
    R, ranks, flows = plots.pathway_flows(frame, axis="signed")
    assert R.min() < 0 < R.max() and all((q >= 0).all() for q in flows.values())
    assert sum(q.sum() for q in flows.values()) > 0


def test_the_maps(frame):
    import matplotlib.pyplot as plt
    pd = frame.path == "plasmodesmata"
    assert np.nanmin(frame.aperture[pd]) == pytest.approx(0.6) and np.nanmax(frame.aperture[pd]) == pytest.approx(1.)
    assert np.isfinite(frame.distance[frame.kind == 0]).all()
    fig, axes = plt.subplots(1, 2)
    distance, aperture = plots.contact_distance_map(axes, frame)
    assert aperture.norm.vmin >= 0.6 - 1e-12 and distance.norm.vmax > 0
    plt.close(fig)


def test_the_contact_all_around_is_centred():
    """With the soil all around, the contact's centroid is within a few µm of the centre: the axis (develop's 1e-9 µm
    threshold kept) is then not meaningful, which the plots' users must know."""
    g, ds, anatomy = anatomy_data_structure(root(1))
    ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    mecha = MechaHydraulics(data_structure=ds)
    mecha()
    whole = plots.port_frame(mecha, int(ds.entity_ids("SubOrgan")[0]))
    transect = plots.wetdry_axis(whole)
    assert transect is None or np.hypot(*(whole.xy[whole.contact].mean(axis=0) - np.asarray(whole.centre))) < 5.
