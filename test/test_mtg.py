from openalea.mtg import MTG
from utils import mtg_to_arraydict


label_plant = 1
label_axis = 2
label_metamer = 3
label_organ = 4
label_segment = 5
label_layer = 6
label_cell = 7
label_node = 8
label_edge = 9

labels = [  label_plant,
            label_axis,
            label_metamer,
            label_organ,
            label_segment,
            label_layer,
            label_cell,
            label_node,
            label_edge  
            ]

scales = {j: i+1 for i, j in enumerate(labels)}
labels = [  label_plant,
            label_axis,
            label_metamer,
            label_organ,
            label_segment,
            # label_layer,
            # label_cell,
            # label_node,
            # label_edge
            ]


def build_mtg(labels: list) -> MTG:
    """
    Build an MTG with actual scales (OpenAlea MTG concept), not a 'scale' property.
    """
    g = MTG()
    root = g.root

    anchor = root
    for label in labels:
        anchor = g.add_component(anchor, label=label)

    return g

def mtg_summary(g: MTG):
    root = g.root
    print("MTG root:", root)
    max_scale = g.max_scale()
    print("max scale:", max_scale)
    for k in range(max_scale):
        print(f"scale {k+1}: {len(g.components_at_scale(root, scale=k+1))} elements")
    print("available properties:", g.properties().keys())
    

if __name__ == "__main__":
    g = build_mtg(labels=labels)
    mtg_summary(g=g)
    props = g.properties()

    # skipping the plant creation part to jump straight to a single segment
    segment_id = g.component_roots_at_scale(g.root, scale=scales[label_segment])[0]
    init_layer = dict(type = ["cotex", "endodermis", "stele"],
                      r_min = [0., 1e-5, 2e-5],
                      r_max = [1e-5, 2e-5, 3e-5])

    for k in range(len(init_layer[list(init_layer.keys())[0]])):
        init_dict = {i: l[k] for i, l in init_layer.items()}
        g.add_component(segment_id, label=label_layer, **init_dict)


    for layer_id in g.component_roots_at_scale(segment_id, scale=scales[label_layer]):
        r_min = props["r_min"][layer_id]
        init_cell = dict(type = [props["type"][layer_id]],
                         minor= [1e-5],
                         major=[1e-5],
                         x_cell = [r_min+5e-6],
                         y_cell = [0., 0., 0.],
                         polygon = [((r_min, 0.), (r_min, 5e-6), (r_min, -5e-6), (r_min+5e-6, 5e-6), (r_min+5e-6, -5e-6), (r_min+1e-5, 0.), (r_min+1e-5, 5e-6), (r_min+1e-5, -5e-6))],
                         vertex_type = [["wall", "junction", "junction", "wall", "wall", "wall", "junction", "junction"]])
        
        for k in range(len(init_cell[list(init_cell.keys())[0]])):
            init_dict = {i: l[k] for i, l in init_cell.items()}
            g.add_component(layer_id, label=label_cell, **init_dict)

    cells = g.component_roots_at_scale(segment_id, scale=scales[label_cell])
    symbolic_anchoring = cells[0]
    sorted_nodes = []
    for cell_id in cells:
        init_dict = dict(type = "cell",
                         x = props["x_cell"][cell_id],
                         y = props["y_cell"][cell_id],
                         c_type_a = props["type"][cell_id],
                         c_type_b = "",
                         c_type_c = "",
                         length = 0.
                         )
        g.add_component(symbolic_anchoring, label=label_node, **init_dict)

        for k, (x, y) in enumerate(props["polygon"][cell_id]):
            init_dict = dict(type = props["vertex_type"][cell_id][k],
                         x = props["x_cell"][cell_id],
                         y = props["y_cell"][cell_id],
                         c_type_a = props["type"][cell_id],
                         c_type_b = "",
                         c_type_c = "",
                         length = 0.
                         )
            
            g.add_component(symbolic_anchoring, label=label_node, **init_dict)

        # Algo: when common node, there's a condition on the node usually but here we can have only two neighbors on this test case so not complexified too much

        # Then how to connect? shoud probably be defined based on polygon that should be ordered


    mtg_to_arraydict(g)

    print(props)



