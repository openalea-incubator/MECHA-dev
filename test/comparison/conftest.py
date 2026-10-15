"""The legacy MECHA and the port on the same GRANAP sections: one-segment roots (monocot, dicot), 200 µm long."""
import numpy as np
import pytest

from legacy_run import legacy_section
from plants import anatomy_component, anatomy_data_structure, root, template_organ

BARRIERS = (1, 3)


PRESETS = {"monocot_root": None, "dicot_root": "for_dicot_root"}


@pytest.fixture(scope="session", params=list(PRESETS))
def case(request):
    g, ds, anatomy = anatomy_data_structure(root(1), anatomy_component(PRESETS[request.param]))
    segment = int(ds.entity_ids("SubOrgan")[0])
    height = float(np.asarray(ds.get("length"))[0]) * 1.0E6            # the port's height (µm), given to the legacy
    run = legacy_section(template_organ(anatomy, g, segment), barriers=BARRIERS, height=height)
    return dict(g=g, ds=ds, anatomy=anatomy, segment=segment, legacy=run, height=height)
