import numpy as np
from numpy import empty, zeros, cos, sin, arctan, sign, inf, isnan, array

# Never tried, was vibe coding this part

def prepare_geometrical_properties(general, network, position, indice):
    """
    Prepare geometrical properties arrays for MECHA simulation.

    Parameters
    ----------
    general : GeneralData
        General configuration data.
    network : NetworkBuilder
        The network structure.
    position : dict
        Node positions (XY coordinates).
    indice : dict
        Node indices mapping.

    Returns
    -------
    dict
        A dictionary containing all prepared geometrical properties arrays.
    """
    # Short aliases
    
    layer_dist         = network.layer_dist
    distance_from_c    = network.distance_from_center
    rank_to_row        = network.rank_to_row
    cell_ranks         = network.cell_ranks
    border_link        = network.border_link
    x_min, x_max       = network.x_min, network.x_max
    xylem80            = network.xylem_80_percentile_distance
    r_rel              = network.r_rel
    x_rel              = network.x_rel

    use_thick = (
        general.paraview == 1 or general.par_track == 1 or
        general.apo_contagion > 0 or general.sym_contagion > 0
    )

    def nan_array(shape):
        arr = empty(shape)
        arr[:] = np.nan
        return arr

    # Common arrays (used in both thick and thin cases)
    wall_to_cell = nan_array((network.n_walls, 2))
    n_wall_to_cell = zeros((network.n_walls, 1))
    junction_wall_cell = nan_array((network.n_wall_junction - network.n_walls, 12))
    n_junction_wall_cell = zeros((network.n_wall_junction - network.n_walls, 1))
    wall_to_junction = nan_array((network.n_walls, 2))
    n_wall_to_junction = zeros((network.n_walls, 1))

    # Diffusion wall lengths (cm)
    L_diff = (
        abs(float((layer_dist[3] - layer_dist[2]) * 1.0e-4)),
        abs(float((layer_dist[3] - xylem80) * 1.0e-4)),
    )

    # -------------------------------------------------------------------------
    # Thick-wall specific arrays
    # -------------------------------------------------------------------------
    thick_wall      = []
    thick_wall_x      = []
    thick_wall_polygon_x = None
    wall_to_wall     = None
    wall_to_wall_x    = None
    junction_to_wall   = None
    cell_to_thick_wall  = None

    n_thick_wall      = None
    n_wall_to_wall    = None
    n_wall_to_wall_x   = None
    n_junction_to_wall  = None
    n_thick_wall_polygon_x = None
    n_cell_to_thick_wall = None

    if use_thick:
        print("Preparing geometrical properties for general.paraview")

        # Counts of new junction wall IDs already saved
        n_thick_wall = zeros((2 * network.n_walls, 1))

        thick_wall_x = []  # thick-wall points including extra info

        wall_to_wall   = nan_array((network.n_walls, 2))
        n_wall_to_wall  = zeros((network.n_walls, 1))

        wall_to_wall_x  = nan_array((network.n_wall_junction, 8))  # includes junction→new junction ID mapping
        n_wall_to_wall_x = zeros((network.n_wall_junction, 1))

        thick_wall_polygon_x  = nan_array((2 * network.n_walls, 4))
        n_thick_wall_polygon_x = zeros((2 * network.n_walls, 1))

        wall_to_cell[:] = np.nan  # already set, but kept for clarity

        junction_to_wall   = nan_array((network.n_wall_junction - network.n_walls, 12))
        n_junction_to_wall  = zeros((network.n_wall_junction - network.n_walls, 1))

        cell_to_thick_wall  = nan_array((network.n_cells, 32))
        n_cell_to_thick_wall = zeros((network.n_cells, 1))

        # re-define L_diff here as in original (same numerical intent)
        L_diff = (
            abs(float((layer_dist[3][0] - layer_dist[2][0]) * 1.0e-4)),
            abs(float((layer_dist[3][0] - xylem80) * 1.0e-4)),
        )

    # -------------------------------------------------------------------------
    # FIRST PASS on adjacency: build wall_to_cell, wall_to_junction, r_rel, x_rel,
    # and optionally thick-wall geometry
    # -------------------------------------------------------------------------
    twpid  = 0  # Thick wall point ID (for ThickWalls)
    twpidX = 0  # Thick wall point ID (for thick_wall_x)

    for node, edges in network.graph.adjacency():
        wall_id = indice[node]
        if wall_id >= network.n_walls:  # only "walls", not junctions/cells
            continue

        # Loop on neighbors
        for neighboor, eattr in edges.items():
            cid = int(indice[neighboor])
            ntype = network.graph.nodes[cid]["type"]

            if ntype == "cell":
                # Common: build wall_to_cell
                idx = int(n_wall_to_cell[wall_id][0])
                wall_to_cell[wall_id][idx] = cid
                n_wall_to_cell[wall_id] += 1

                if use_thick:
                    # Geometry of thick walls (cell side)
                    if position[cid][0] != position[wall_id][0]:
                        slopeCG = (position[cid][1] - position[wall_id][1]) / (
                            position[cid][0] - position[wall_id][0]
                        )
                    else:
                        slopeCG = inf

                    dx = cos(arctan(slopeCG)) * general.thickness_disp / 2.0
                    dy = sin(arctan(slopeCG)) * general.thickness_disp / 2.0
                    sgn = sign(position[cid][0] - position[wall_id][0])

                    xw = position[wall_id][0] + dx * sgn
                    yw = position[wall_id][1] + dy * sgn

                    # thick_wallentry
                    ThickWalls.append(
                        array(
                            (
                                twpid,
                                wall_id,
                                cid,
                                xw,
                                yw,
                                inf,  # neighbor new junction walls IDs not known yet
                                inf,
                                border_link[wall_id],
                            )
                        )
                    )

                    cell_idx = int(cid - network.n_wall_junction)
                    cw_idx = int(n_cell_to_thick_wall[cell_idx])
                    cell_to_thick_wall[cell_idx][cw_idx] = twpid
                    n_cell_to_thick_wall[cell_idx] += 1

                    # thick_wall_x entry
                    thick_wall_x.append((twpidX, xw, yw, wall_id, cid))

                    # Polygon association (wall has 2 polygons)
                    for poly_row in (2 * wall_id, 2 * wall_id + 1):
                        pidx = int(n_thick_wall_polygon_x[poly_row])
                        thick_wall_polygon_x[poly_row][pidx] = twpidX
                        n_thick_wall_polygon_x[poly_row] += 1

                    # Mapping from original wall to new thick-wall node
                    wall_to_wall[wall_id][int(n_wall_to_wall[wall_id])] = twpid
                    n_wall_to_wall[wall_id] += 1

                    wall_to_wall_x[wall_id][int(n_wall_to_wall_x[wall_id])] = twpidX
                    n_wall_to_wall_x[wall_id] += 1

                    twpid += 1
                    twpidX += 1

                    # If border wall, create opposite thick-wall node
                    if border_link[wall_id] == 1:
                        xw2 = position[wall_id][0] - dx * sgn
                        yw2 = position[wall_id][1] - dy * sgn
                        thick_wall_x.append((twpidX, xw2, yw2, wall_id, inf))

                        for poly_row in (2 * wall_id, 2 * wall_id + 1):
                            pidx = int(n_thick_wall_polygon_x[poly_row])
                            thick_wall_polygon_x[poly_row][pidx] = twpidX
                            n_thick_wall_polygon_x[poly_row] += 1

                        wall_to_wall_x[wall_id][int(n_wall_to_wall_x[wall_id])] = twpidX
                        n_wall_to_wall_x[wall_id] += 1
                        twpidX += 1

            elif ntype == "apo":
                # j is a junction node
                idx = int(n_wall_to_junction[wall_id][0])
                wall_to_junction[wall_id][idx] = indice[neighboor]
                n_wall_to_junction[wall_id] += 1

    # -------------------------------------------------------------------------
    # SECOND PASS on adjacency: build junction_wall_cell (+ thick junctions)
    # -------------------------------------------------------------------------

    def find_index(arr, value):
        for idx, v in enumerate(arr):
            if v == value:
                return idx
        return None

    for node, edges in network.graph.adjacency():
        i = indice[node]
        if i >= network.n_walls:
            continue  # only walls

        for neighboor, eattr in edges.items():
            j = indice[neighboor]
            if network.graph.nodes[j]["type"] != "apo":
                continue  # only junctions

            j_idx = j - network.n_walls  # index in junction_wall_cell / junction_to_wall

            for cid in wall_to_cell[i]:
                if isnan(cid):
                    continue

                # If cell not yet associated to this junction via any wall
                if cid not in junction_wall_cell[j_idx]:
                    pos = int(n_junction_wall_cell[j_idx][0])
                    junction_wall_cell[j_idx][pos] = cid
                    n_junction_wall_cell[j_idx] += 1

                    if use_thick:
                        junction_to_wall[j_idx][int(n_junction_to_wall[j_idx])] = i
                        n_junction_to_wall[j_idx] += 1
                else:
                    # Already associated through another wall => "thick junction node"
                    if not use_thick:
                        continue  # nothing else to do in the thin case

                    # Existing association: find wid1 (the other wall)
                    pos = find_index(junction_wall_cell[j_idx], cid)
                    if pos is None:
                        continue
                    wid1 = int(junction_to_wall[j_idx][pos])

                    # thick wall nodes for wid1 and i for this cell
                    pos1 = find_index(wall_to_cell[wid1], cid)
                    twpid1 = int(wall_to_wall[wid1][pos1])

                    pos2 = find_index(wall_to_cell[i], cid)
                    twpid2 = int(wall_to_wall[i][pos2])

                    # slope of line junction j → cell cid
                    if position[cid][0] != position[j][0]:
                        slopeCG = (position[cid][1] - position[j][1]) / (
                            position[cid][0] - position[j][0]
                        )
                    else:
                        slopeCG = inf

                    dx = cos(arctan(slopeCG)) * general.thickness_junction_disp / 2.0
                    dy = sin(arctan(slopeCG)) * general.thickness_junction_disp / 2.0
                    sgn = sign(position[cid][0] - position[j][0])

                    xj = position[j][0] + dx * sgn
                    yj = position[j][1] + dy * sgn

                    # thick_wallentry for junction node
                    ThickWalls.append(
                        array(
                            (
                                twpid,
                                j,
                                int(cid),
                                xj,
                                yj,
                                twpid1,
                                twpid2,
                                border_link[j],
                            )
                        )
                    )
                    # Add link from existing thick-wall points to this one
                    ThickWalls[twpid1][int(5 + n_thick_wall[twpid1])] = twpid
                    ThickWalls[twpid2][int(5 + n_thick_wall[twpid2])] = twpid
                    n_thick_wall[twpid1] += 1
                    n_thick_wall[twpid2] += 1

                    # Cell ↔ thick-node map
                    cell_idx = int(cid - network.n_wall_junction)
                    pos_c  = int(n_cell_to_thick_wall[cell_idx])
                    cell_to_thick_wall[cell_idx][pos_c] = twpid
                    n_cell_to_thick_wall[cell_idx] += 1

                    # thick_wall_x: (twpidX, x, y, junction j, cell cid, i, wid1)
                    thick_wall_x.append((twpidX, xj, yj, j, cid, i, wid1))

                    # ---- Polygons for wall i ----
                    for id1, val in enumerate(wall_to_junction[i]):
                        if val == j:
                            break  # 2*i+id1 row is polygon row for wall i and junction j
                    poly_i = 2 * i + id1
                    first_pt_idx = int(thick_wall_polygon_x[poly_i][1])
                    first_cell   = thick_wall_x[int(first_pt_idx)][4]

                    if first_cell == cid:
                        thick_wall_polygon_x[poly_i][2] = twpidX
                    else:
                        thick_wall_polygon_x[poly_i][3] = twpidX
                    n_thick_wall_polygon_x[poly_i] += 1

                    # ---- Polygons for wall wid1 ----
                    for id1, val in enumerate(wall_to_junction[wid1]):
                        if val == j:
                            break
                    poly_w = 2 * wid1 + id1
                    first_pt_idx = int(thick_wall_polygon_x[poly_w][1])
                    first_cell   = thick_wall_x[int(first_pt_idx)][4]

                    if first_cell == cid:
                        thick_wall_polygon_x[poly_w][2] = twpidX
                    else:
                        thick_wall_polygon_x[poly_w][3] = twpidX
                    n_thick_wall_polygon_x[poly_w] += 1

                    # New wall ID for each original wall/junction
                    wall_to_wall_x[j][int(n_wall_to_wall_x[j])] = twpidX
                    n_wall_to_wall_x[j] += 1

                    twpid += 1
                    twpidX += 1

                    # Border thick junction
                    if border_link[j] == 1:
                        if slopeCG != 0.0:
                            slopeCG = -1.0 / slopeCG
                        else:
                            slopeCG = inf

                        dxb = cos(arctan(slopeCG)) * general.thickness_disp / 2.0
                        dyb = sin(arctan(slopeCG)) * general.thickness_disp / 2.0

                        xb = position[j][0] - dxb * sgn
                        yb = position[j][1] - dyb * sgn

                        thick_wall_x.append((twpidX, xb, yb, j, inf, i))

                        # This border polygon is for wall i or wid1?
                        wall_id = i if border_link[i] == 1 else wid1

                        for id1, val in enumerate(wall_to_junction[wall_id]):
                            if val == j:
                                break
                        poly_b = 2 * wall_id + id1
                        first_pt_idx = int(thick_wall_polygon_x[poly_b][1])
                        first_cell   = thick_wall_x[int(first_pt_idx)][4]

                        if first_cell == cid:
                            thick_wall_polygon_x[poly_b][3] = twpidX
                        else:
                            thick_wall_polygon_x[poly_b][2] = twpidX
                        n_thick_wall_polygon_x[poly_b] += 1

                        wall_to_wall_x[j][int(n_wall_to_wall_x[j])] = twpidX
                        n_wall_to_wall_x[j] += 1
                        twpidX += 1


    return {
        'wall_to_cell': wall_to_cell,
        'n_wall_to_cell': n_wall_to_cell,
        'junction_wall_cell': junction_wall_cell,
        'n_junction_wall_cell': n_junction_wall_cell,
        'wall_to_junction': wall_to_junction,
        'n_wall_to_junction': n_wall_to_junction,
        'r_rel': r_rel,
        'x_rel': x_rel,
        'L_diff': L_diff,
        'thick_wall': thick_wall,
        'thick_wall_x': thick_wall_x,
        'thick_wall_polygon_x': thick_wall_polygon_x,
        'wall_to_wall': wall_to_wall,
        'wall_to_wall_x': wall_to_wall_x,
        'junction_to_wall': junction_to_wall,
        'cell_to_thick_wall': cell_to_thick_wall,
        'n_thick_wall': n_thick_wall,
        'n_wall_to_wall': n_wall_to_wall,
        'n_wall_to_wall_x': n_wall_to_wall_x,
        'n_junction_to_wall': n_junction_to_wall,
        'n_thick_wall_polygon_x': n_thick_wall_polygon_x,
        'n_cell_to_thick_wall': n_cell_to_thick_wall,
    }