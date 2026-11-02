"""
MECHA's stand-ins for the models its inputs come from (stand_ins): the barriers by distance from the tip
(Development, its zones from a root's barrier codes), the leaves' transpiration (Transpiration, the plant's shared by
its leaf organs).
"""
import numpy as np
import pytest

from openalea.mecha.stand_ins import development_parameters

SEGMENT = 5e-3                                             # m


def test_a_root_s_codes_make_its_zones():
    """The codes from the collar to the tip at their distances from the tip: the tip's, the young root's, the mature
    root's zones; a single code is mature_barrier alone."""
    distances = [3 * SEGMENT, 2 * SEGMENT, SEGMENT, 0.]
    assert development_parameters(distances, (3, 3, 1, 0)) == {"mature_barrier": 3., "young_zone": SEGMENT,
                                                               "young_barrier": 1., "tip_zone": 0., "tip_barrier": 0.}
    assert development_parameters(distances, (3, 1, 1, 1)) == {"mature_barrier": 3., "young_zone": 2 * SEGMENT,
                                                               "young_barrier": 1.}
    assert development_parameters(distances, (1, 1, 1, 1)) == {"mature_barrier": 1.}


def test_codes_that_do_not_follow_the_distance_raise():
    with pytest.raises(ValueError, match="different barriers"):     # a lateral's segment differs from the main's
        development_parameters([SEGMENT, 0., 0.], (3, 1, 0))
    with pytest.raises(ValueError, match="three"):
        development_parameters([3 * SEGMENT, 2 * SEGMENT, SEGMENT, 0.], (4, 3, 1, 0))
