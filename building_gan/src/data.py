import os
import sys
import json
import torch

from tqdm import tqdm
from torch_geometric.data import Data

if os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from building_gan.src.config import Configuration


class GraphData(Data):
    def __init__(self, processed_data: dict):
        super().__init__(self)

        self.processed_data = processed_data

        self.far = processed_data["far"]
        self.type_ratio = processed_data["type_ratio"]
        self.local_graph_types = processed_data["local_graph_types"]
        self.local_graph_floor_levels = processed_data["local_graph_floor_levels"]
        self.local_graph_edge_indices = processed_data["local_graph_edge_indices"]
        self.voxel_graph_labels = processed_data["voxel_graph_labels"]
        self.voxel_graph_features = processed_data["voxel_graph_features"]
        self.voxel_graph_edge_indices = processed_data["voxel_graph_edge_indices"]
        self.voxel_and_local_cross_edge_indices = processed_data["voxel_and_local_cross_edge_indices"]


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
        local_graph_types = []

        local_graph_nodes = local_graph_data["node"]
        for ni, local_node in enumerate(local_graph_nodes):
            unique_index = local_node["floor"], local_node["type"], local_node["type_id"]
            local_graph_unique_indices[unique_index] = ni
            local_graph_floor_levels.append(local_node["floor"])
            local_graph_types.append(local_node["type"])

        local_graph_types = torch.nn.functional.one_hot(
            torch.tensor(local_graph_types), num_classes=configuration.NUM_CLASSES
        )

        assert len(local_graph_floor_levels) == len(local_graph_nodes)
        assert len(local_graph_types) == len(local_graph_nodes)

        # computes local graph edge indices
        local_graph_adjacency_matrix = torch.zeros(size=(len(local_graph_nodes), len(local_graph_nodes)))
        for local_node in local_graph_nodes:
            ui = local_graph_unique_indices[local_node["floor"], local_node["type"], local_node["type_id"]]

            for local_node_neighbor in local_node["neighbors"]:
                uj = local_graph_unique_indices[tuple(local_node_neighbor)]
                local_graph_adjacency_matrix[ui][uj] = 1

        # converts adjacency_matrix to the list of list
        local_graph_edge_indices = local_graph_adjacency_matrix.nonzero().t()

        # process global graph data
        far = torch.tensor([global_graph_data["far"]])

        global_graph_nodes = global_graph_data["global_node"]
        type_ratio = [0] * len(global_graph_nodes)
        for global_node in global_graph_nodes:
            type_ratio[global_node["type"]] = global_node["proportion"]

        type_ratio = torch.tensor(type_ratio)

        # process voxel graph data
        voxel_graph_unique_indices = {}
        voxel_graph_features = []
        voxel_graph_labels = []
        voxel_graph_floor_levels = []

        voxel_graph_nodes = voxel_graph_data["voxel_node"]
        for vni, voxel_node in enumerate(voxel_graph_nodes):
            voxel_graph_unique_indices[tuple(voxel_node["location"])] = vni

            floor_level, _, _ = voxel_node["location"]
            voxel_graph_floor_levels.append(floor_level)

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

        # compute voxel graph edge indices
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

        assert len(voxel_graph_labels) == len(voxel_graph_nodes)
        assert len(voxel_graph_floor_levels) == len(voxel_graph_nodes)
        assert len(voxel_graph_features) == len(voxel_graph_nodes)

        local_graph_node_indices_per_level = [[] for _ in range(max(local_graph_floor_levels) + 1)]
        for i, floor_level in enumerate(local_graph_floor_levels):
            local_graph_node_indices_per_level[floor_level].append(i)

        each_floor_voxel_counts = voxel_graph_floor_levels.unique(return_counts=True)[1][0].item()

        # compute voxel and local graph cross edge indices
        voxel_id = 0
        local_graph_cross_edge_indices = []
        voxel_graph_cross_edge_indices = []
        for local_graph_node_indices_each_level in local_graph_node_indices_per_level:
            local_graph_cross_edge_indices.extend(local_graph_node_indices_each_level * each_floor_voxel_counts)
            for _ in range(each_floor_voxel_counts):
                voxel_graph_cross_edge_indices += [voxel_id] * len(local_graph_node_indices_each_level)
                voxel_id += 1

        local_graph_cross_edge_indices = torch.tensor(local_graph_cross_edge_indices)
        voxel_graph_cross_edge_indices = torch.tensor(voxel_graph_cross_edge_indices)
        voxel_and_local_cross_edge_indices = torch.stack(
            [local_graph_cross_edge_indices, voxel_graph_cross_edge_indices]
        )

        assert len(local_graph_cross_edge_indices) == len(voxel_graph_cross_edge_indices)

        return GraphData(
            {
                "far": far,
                "type_ratio": type_ratio,
                "local_graph_types": local_graph_types,
                "local_graph_floor_levels": local_graph_floor_levels,
                "local_graph_edge_indices": local_graph_edge_indices,
                "voxel_graph_labels": voxel_graph_labels,
                "voxel_graph_features": voxel_graph_features,
                "voxel_graph_edge_indices": voxel_graph_edge_indices,
                "voxel_and_local_cross_edge_indices": voxel_and_local_cross_edge_indices,
            }
        )


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

        os.makedirs(self.configuration.SAVE_DATA_PATH, exist_ok=True)

        for global_graph_path, local_graph_path, voxel_graph_path in tqdm(
            zip(global_graphs, local_graphs, voxel_graphs), total=len(voxel_graphs)
        ):
            with open(global_graph_path, "r") as f:
                global_graph_data = json.load(f)

            with open(local_graph_path, "r") as f:
                local_graph_data = json.load(f)

            with open(voxel_graph_path, "r") as f:
                voxel_graph_data = json.load(f)

            processed_data = self.process_data(
                global_graph_data,
                local_graph_data,
                voxel_graph_data,
                self.configuration,
            )

            processed_data_name = "".join([s for s in global_graph_path.split("/")[-1] if s.isdigit()]) + ".pt"

            torch.save(processed_data, os.path.join(self.configuration.SAVE_DATA_PATH, processed_data_name))
