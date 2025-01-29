import os
import sys
import json
import torch

if os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from building_gan.src.config import Configuration


class DataCreatorHelper:
    @staticmethod
    def process_data(
        global_graph_data: dict,
        local_graph_data: dict,
        voxel_graph_data: dict,
        configuration: Configuration,
    ) -> dict:
        # processes local graph data
        local_graph_unique_indices = {}
        local_graph_floor_levels = []
        local_graph_labels = []

        local_graph_nodes = local_graph_data["node"]
        for ni, local_node in enumerate(local_graph_nodes):
            unique_index = local_node["floor"], local_node["type"], local_node["type_id"]
            local_graph_unique_indices[unique_index] = ni
            local_graph_floor_levels.append(local_node["floor"])
            local_graph_labels.append(local_node["type"])

        local_graph_floor_levels = torch.tensor(local_graph_floor_levels)
        local_graph_labels = torch.nn.functional.one_hot(
            torch.tensor(local_graph_labels), num_classes=configuration.NUM_CLASSES
        )

        # computes local graph edge indices
        local_graph_adjacency_matrix = torch.zeros(size=(len(local_graph_nodes), len(local_graph_nodes)))
        for local_node in local_graph_nodes:
            ui = local_graph_unique_indices[local_node["floor"], local_node["type"], local_node["type_id"]]

            for local_node_neighbor in local_node["neighbors"]:
                uj = local_graph_unique_indices[tuple(local_node_neighbor)]
                local_graph_adjacency_matrix[ui][uj] = 1

        # converts adjacency_matrix to the list of list
        local_graph_edge_indices = local_graph_adjacency_matrix.nonzero().t()

        # computes local graph negative edge indicies
        local_graph_negative_edge_indices_same_floor = []
        local_graph_negative_edge_indices_diff_floor = []
        for ni in range(len(local_graph_nodes)):
            for nj in range(ni + 1, len(local_graph_nodes)):
                if local_graph_adjacency_matrix[ni][nj] != 1:
                    if local_graph_floor_levels[ni] == local_graph_floor_levels[nj]:
                        local_graph_negative_edge_indices_same_floor.append([ni, nj])
                        local_graph_negative_edge_indices_same_floor.append([nj, ni])
                    else:
                        local_graph_negative_edge_indices_diff_floor.append([ni, nj])
                        local_graph_negative_edge_indices_diff_floor.append([nj, ni])

        local_graph_negative_edge_indices_same_floor = torch.tensor(local_graph_negative_edge_indices_same_floor).t()
        local_graph_negative_edge_indices_diff_floor = torch.tensor(local_graph_negative_edge_indices_diff_floor).t()

        # process global graph data
        far = torch.tensor([global_graph_data["far"]])
        type_ratio = {}
        global_graph_nodes = global_graph_data["global_node"]
        for global_node in global_graph_nodes:
            type_ratio[global_node["type"]] = global_node["proportion"]

        # process voxel graph data
        voxel_graph_unique_indices = {}
        voxel_graph_stacks = {}
        voxel_graph_features = []
        voxel_graph_labels = []
        voxel_graph_floor_levels = []

        voxel_graph_nodes = voxel_graph_data["voxel_node"]
        for vni, voxel_node in enumerate(voxel_graph_nodes):
            voxel_graph_unique_indices[tuple(voxel_node["location"])] = vni

            z, y, x = voxel_node["location"]
            voxel_graph_floor_levels.append(z)
            voxel_graph_features.append(
                [
                    *[c / configuration.NORMALIZATION_FACTOR for c in voxel_node["coordinate"]],
                    *[d / configuration.NORMALIZATION_FACTOR for d in voxel_node["dimension"]],
                    voxel_node["weight"],
                ]
            )

            if voxel_node["type"] < 0:
                voxel_graph_labels.append([0] * configuration.NUM_CLASSES)
            else:
                voxel_graph_labels.append(
                    torch.nn.functional.one_hot(
                        torch.tensor(voxel_node["type"]), num_classes=configuration.NUM_CLASSES
                    ).tolist()
                )

            horizontal_projection = (x, y)
            if horizontal_projection in voxel_graph_stacks:
                voxel_graph_stacks[horizontal_projection].append(vni)
            else:
                voxel_graph_stacks[horizontal_projection] = [vni]

        voxel_graph_projection = [-1] * len(voxel_graph_nodes)
        for vi, value in enumerate(voxel_graph_stacks.values()):
            for vj in value:
                voxel_graph_projection[vj] = vi

        voxel_graph_adjacency_matrix = torch.zeros(size=(len(voxel_graph_nodes), len(voxel_graph_nodes)))
        for vni, voxel_node in enumerate(voxel_graph_nodes):
            ui = voxel_graph_unique_indices[tuple(voxel_node["location"])]

            for voxel_node_neighbor in voxel_node["neighbors"]:
                uj = voxel_graph_unique_indices[tuple(voxel_node_neighbor)]

                voxel_graph_adjacency_matrix[ui][uj] = 1

        voxel_graph_edge_indices = voxel_graph_adjacency_matrix.nonzero().t()

        voxel_graph_labels = torch.tensor(voxel_graph_labels)
        voxel_graph_floor_levels = torch.tensor(voxel_graph_floor_levels)
        voxel_graph_features = torch.tensor(voxel_graph_features)
        voxel_graph_projection = torch.tensor(voxel_graph_projection)

        return {
            "far": far,
            "type_ratio": type_ratio,
            "local_graph_unique_indices": local_graph_unique_indices,
            "local_graph_floor_levels": local_graph_floor_levels,
            "local_graph_types": local_graph_labels,
            "local_graph_edge_indices": local_graph_edge_indices,
            "local_graph_negative_edge_indices_same_floor": local_graph_negative_edge_indices_same_floor,
            "local_graph_negative_edge_indices_diff_floor": local_graph_negative_edge_indices_diff_floor,
            "voxel_graph_labels": voxel_graph_labels,
            "voxel_graph_features": voxel_graph_features,
            "voxel_graph_projection": voxel_graph_projection,
            "voxel_graph_edge_indices": voxel_graph_edge_indices,
            "voxel_graph_floor_levels": voxel_graph_floor_levels,
        }


class DataCreator(DataCreatorHelper):
    def __init__(self, configuration: Configuration):
        self.configuration = configuration

    def create(self):
        global_graphs = [
            os.path.join(self.configuration.GLOBAL_GRAPH_DATA_PATH, d)
            for d in os.listdir(self.configuration.GLOBAL_GRAPH_DATA_PATH)
        ]

        local_graphs = [
            os.path.join(self.configuration.LOCAL_GRAPH_DATA_PATH, d)
            for d in os.listdir(self.configuration.LOCAL_GRAPH_DATA_PATH)
        ]

        voxel_graphs = [
            os.path.join(self.configuration.VOXEL_GRAPH_DATA_PATH, d)
            for d in os.listdir(self.configuration.VOXEL_GRAPH_DATA_PATH)
        ]

        for global_graph_path, local_graph_path, voxel_graph_path in zip(global_graphs, local_graphs, voxel_graphs):
            with open(global_graph_path, "r") as f:
                global_graph_data = json.load(f)

            with open(local_graph_path, "r") as f:
                local_graph_data = json.load(f)

            with open(voxel_graph_path, "r") as f:
                voxel_graph_data = json.load(f)

            _ = self.process_data(
                global_graph_data,
                local_graph_data,
                voxel_graph_data,
                self.configuration,
            )


if __name__ == "__main__":
    data_creator = DataCreator(configuration=Configuration())
    data_creator.create()
