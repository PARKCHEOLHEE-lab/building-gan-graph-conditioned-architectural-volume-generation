import os
import sys
import json
import math
import tqdm

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

from building_gan.src.config import Configuration


def main():
    """compute the number of node types"""

    voxel_graphs = [
        os.path.join(Configuration.VOXEL_GRAPH_DATA_PATH, d)
        for d in os.listdir(Configuration.VOXEL_GRAPH_DATA_PATH)
    ]
    voxel_graphs = sorted(
        voxel_graphs, 
        key=lambda x: int(os.path.basename(x).replace(".json", "").split("_")[-1])
    )

    global_graphs = [
        os.path.join(Configuration.GLOBAL_GRAPH_DATA_PATH, d)
        for d in os.listdir(Configuration.GLOBAL_GRAPH_DATA_PATH)
    ]
    global_graphs = sorted(
        global_graphs, 
        key=lambda x: int(os.path.basename(x).replace(".json", "").split("_")[-1])
    )
    
    voxel_types = {}
    site_areas = []
    floor_levels = []
    dimensions = []
    locations = []
    coordinates = []
    for voxel_graph_path, global_graph_path in tqdm.tqdm(zip(voxel_graphs, global_graphs), total=len(voxel_graphs)):
        data_number_global = os.path.basename(global_graph_path).replace(".json", "").split("_")[-1]
        data_number_voxel = os.path.basename(voxel_graph_path).replace(".json", "").split("_")[-1]
        
        assert data_number_global == data_number_voxel

        with open(global_graph_path, "r") as f:
            global_graph_data = json.load(f)

        with open(voxel_graph_path, "r") as f:
            voxel_graph_data = json.load(f)

        site_areas.append(global_graph_data["site_area"])
        
        gfa = 0
        voxel_nodes = voxel_graph_data["voxel_node"]
        for voxel_node in voxel_nodes:
            
            voxel_type = voxel_node["type"]
            if voxel_type not in voxel_types:
                voxel_types[voxel_type] = 0
            
            voxel_types[voxel_type] += 1
            
            if voxel_type == Configuration.VOID_OLD:
                continue
            
            z_dim, y_dim, x_dim = voxel_node["dimension"]
            gfa += y_dim * x_dim
            
            floor_levels.append(voxel_node["location"][0])
            dimensions.extend([z_dim, y_dim, x_dim])
            locations.extend(voxel_node["location"])
            coordinates.extend(voxel_node["coordinate"])
            
        # check if the different data exists between the data FAR and the computed FAR exists  
        assert math.isclose(
            global_graph_data["far"], 
            gfa / global_graph_data["site_area"],
        )
    
    print("voxel_types:", sorted(list(voxel_types.items()), key=lambda x: x[0]))  
    print("min(site_areas):", min(site_areas))  
    print("max(site_areas):", max(site_areas))
    print("min(dimensions):", min(dimensions))
    print("max(dimensions):", max(dimensions))
    print("min(locations):", min(locations))
    print("max(locations):", max(locations))
    print("min(coordinates):", min(coordinates))
    print("max(coordinates):", max(coordinates))
    print("min(floor_levels):", min(floor_levels))
    print("max(floor_levels):", max(floor_levels))

    # 100%|█████████████████████████████████████████████████████████████████| 10000/10000 [00:56<00:00, 176.36it/s]
    # voxel_types: [(-1, 1342993), (0, 522887), (1, 253412), (2, 109624), (3, 197512), (4, 1520140), (5, 44545)]
    # min(site_areas): 324
    # max(site_areas): 1600
    # min(dimensions): 3.0
    # max(dimensions): 11.0
    # min(locations): 0
    # max(locations): 11
    # min(coordinates): 0.0
    # max(coordinates): 42.0
    # min(floor_levels): 0
    # max(floor_levels): 10

    # VOID_OLD = -1
    # LOBBY_CORRIDOR = 0
    # RESTROOM = 1
    # STAIRS = 2
    # ELEVATOR = 3
    # OFFICE = 4
    # MECHANICAL_ROOM = 5
    

if __name__ == "__main__":
    main()