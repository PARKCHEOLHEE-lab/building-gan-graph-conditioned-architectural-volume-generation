import os
import sys
import json
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
    
    voxel_types = {}
    for voxel_graph_path in tqdm.tqdm(voxel_graphs, total=len(voxel_graphs)):
        with open(voxel_graph_path, "r") as f:
            voxel_graph_data = json.load(f)
            
        voxel_nodes = voxel_graph_data["voxel_node"]
        for voxel_node in voxel_nodes:
            
            voxel_type = voxel_node["type"]
            if voxel_type not in voxel_types:
                voxel_types[voxel_type] = 0
            
            voxel_types[voxel_type] += 1
    
    print(sorted(list(voxel_types.items()), key=lambda x: x[0]))        

    # imbalanced types for 10000 data (000000~009999)
    # [(-1, 1342993), (0, 522887), (1, 253412), (2, 109624), (3, 197512), (4, 1520140), (5, 44545)]

    # VOID_OLD = -1
    # NOT_ALLOWED_OLD = -2

    # LOBBY_CORRIDOR = 0
    # RESTROOM = 1
    # STAIRS = 2
    # ELEVATOR = 3
    # OFFICE = 4
    # MECHANICAL_ROOM = 5
    

if __name__ == "__main__":
    main()