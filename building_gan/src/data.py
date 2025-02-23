import os
import sys
import json
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Data, Dataset, Batch

if os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from building_gan.src.config import Configuration


class LocalGraphData:
    def __init__(self, local_graph_data: dict):
        local_graph_types_onehot = local_graph_data["local_graph_types_onehot"]
        local_graph_type_ratio_per_node = local_graph_types_onehot * local_graph_data["type_ratio"]
        local_graph_far_per_node = torch.zeros(local_graph_types_onehot.shape[0], 1) + local_graph_data["far"]
        local_graph_floor_levels_normalized = local_graph_data["local_graph_floor_levels_normalized"].unsqueeze(0).t()

        self.x = torch.cat(
            [
                local_graph_types_onehot,
                local_graph_type_ratio_per_node,
                local_graph_far_per_node,
                local_graph_floor_levels_normalized,
            ],
            dim=1,
        )

        self.data_number = local_graph_data["data_number"]
        self.local_graph_types_onehot = local_graph_types_onehot
        self.edge_index = local_graph_data["local_graph_edge_indices"]
        self.local_graph_floor_levels = local_graph_data["local_graph_floor_levels"]
        self.local_graph_type_ratio_per_node = local_graph_type_ratio_per_node
        self.local_graph_node_cluster = local_graph_data["local_graph_node_cluster"]
        self.local_graph_center = local_graph_data["local_graph_center"]
        self.local_graph_types = local_graph_data["local_graph_types"]
        self.local_graph_type_ids = local_graph_data["local_graph_type_ids"]


class VoxelGraphData:
    def __init__(self, voxel_graph_data: dict):
        voxel_graph_types_onehot = voxel_graph_data["voxel_graph_types_onehot"]
        voxel_graph_features = voxel_graph_data["voxel_graph_features"]
        voxel_graph_far_per_node = torch.zeros(voxel_graph_types_onehot.shape[0], 1) + voxel_graph_data["far"]
        voxel_graph_floor_levels_normalized = voxel_graph_data["voxel_graph_floor_levels_normalized"].unsqueeze(0).t()

        self.x = torch.cat(
            [
                voxel_graph_features,
                voxel_graph_far_per_node,
                voxel_graph_floor_levels_normalized,
            ],
            dim=1,
        )

        self.data_number = voxel_graph_data["data_number"]
        self.voxel_graph_types = voxel_graph_data["voxel_graph_types"]
        self.voxel_graph_types_onehot = voxel_graph_types_onehot
        self.edge_index = voxel_graph_data["voxel_graph_edge_indices"]
        self.voxel_graph_floor_levels = voxel_graph_data["voxel_graph_floor_levels"]
        self.voxel_graph_node_coordinate = voxel_graph_data["voxel_graph_node_coordinate"]
        self.voxel_graph_node_dimension = voxel_graph_data["voxel_graph_node_dimension"]
        self.voxel_graph_location = voxel_graph_data["voxel_graph_location"]
        self.voxel_graph_node_ratio = voxel_graph_types_onehot * voxel_graph_data["voxel_graph_node_ratio"]
        self.voxel_graph_node_ratio = self.voxel_graph_node_ratio.max(dim=1)[0].unsqueeze(1)


class GraphDataset(Dataset):
    def __init__(self, configuration: Configuration):
        super().__init__()
        self.configuration = configuration

        self.local_graph_data_files = [
            os.path.join(self.configuration.SAVE_DATA_PATH, d)
            for d in os.listdir(self.configuration.SAVE_DATA_PATH)
            if d.endswith(configuration.LOCAL_DATA_SUFFIX)
        ][: configuration.DATA_SLICER]

        self.voxel_graph_data_files = [
            os.path.join(self.configuration.SAVE_DATA_PATH, d)
            for d in os.listdir(self.configuration.SAVE_DATA_PATH)
            if d.endswith(configuration.VOXEL_DATA_SUFFIX)
        ][: configuration.DATA_SLICER]

        assert len(self.local_graph_data_files) == len(self.voxel_graph_data_files)

        self.local_graph_data = []
        self.voxel_graph_data = []
        for local_graph_file, voxel_graph_file in zip(self.local_graph_data_files, self.voxel_graph_data_files):
            local_graph = torch.load(local_graph_file)
            self.local_graph_data.append(
                Data(
                    x=local_graph.x,
                    edge_index=local_graph.edge_index,
                    node_cluster=local_graph.local_graph_node_cluster,
                    node_ratio=local_graph.local_graph_type_ratio_per_node,
                    types_onehot=local_graph.local_graph_types_onehot,
                    center=local_graph.local_graph_center,
                    type=local_graph.local_graph_types,
                    type_id=local_graph.local_graph_type_ids,
                    data_number=local_graph.data_number,
                    floor=local_graph.local_graph_floor_levels,
                )
            )

            voxel_graph = torch.load(voxel_graph_file)
            self.voxel_graph_data.append(
                Data(
                    x=voxel_graph.x,
                    edge_index=voxel_graph.edge_index,
                    voxel_level=voxel_graph.voxel_graph_floor_levels,
                    type=voxel_graph.voxel_graph_types,
                    types_onehot=voxel_graph.voxel_graph_types_onehot,
                    coordinate=voxel_graph.voxel_graph_node_coordinate,
                    dimension=voxel_graph.voxel_graph_node_dimension,
                    location=voxel_graph.voxel_graph_location,
                    node_ratio=voxel_graph.voxel_graph_node_ratio,
                    data_number=voxel_graph.data_number,
                )
            )

    def __getitem__(self, i):
        return self.local_graph_data[i], self.voxel_graph_data[i]

    def __len__(self):
        return len(self.local_graph_data)

    @staticmethod
    def collate_fn(batch):
        local_graphs, voxel_graphs = zip(*batch)

        local_batch = Batch.from_data_list(local_graphs)
        voxel_batch = Batch.from_data_list(voxel_graphs)

        return local_batch, voxel_batch


class GraphDataLoaders:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.sanity_checking = self.configuration.SANITY_CHECKING
        self.dataset, self.train_dataloader, self.validation_dataloader, self.test_dataloader = self._get_loaders()

    def _get_loaders(self):
        dataset = GraphDataset(configuration=self.configuration)

        train_dataset, validation_dataset, test_dataset = random_split(dataset, self.configuration.SPLIT_RATIOS)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.configuration.BATCH_SIZE,
            num_workers=self.configuration.NUM_WORKERS,
            shuffle=True,
            drop_last=True,
            collate_fn=GraphDataset.collate_fn,
        )

        validation_dataloader = (
            DataLoader(
                validation_dataset,
                batch_size=self.configuration.BATCH_SIZE,
                num_workers=self.configuration.NUM_WORKERS,
                shuffle=True,
                drop_last=True,
                collate_fn=GraphDataset.collate_fn,
            )
            if not self.sanity_checking
            else None
        )

        test_dataloader = (
            DataLoader(
                test_dataset,
                batch_size=self.configuration.BATCH_SIZE,
                num_workers=self.configuration.NUM_WORKERS,
                shuffle=True,
                drop_last=True,
                collate_fn=GraphDataset.collate_fn,
            )
            if not self.sanity_checking
            else None
        )

        return dataset, train_dataloader, validation_dataloader, test_dataloader


class DataCreatorHelper:
    @staticmethod
    def process_data(
        global_graph_data: dict,
        local_graph_data: dict,
        voxel_graph_data: dict,
        configuration: Configuration,
        data_number: str,
    ):
        # processes local graph data
        local_graph_unique_indices = {}
        local_graph_floor_levels = []
        local_graph_node_cluster = []
        local_graph_center = []
        local_graph_types = []
        local_graph_type_ids = []

        local_graph_nodes = local_graph_data["node"]
        for ni, local_node in enumerate(local_graph_nodes):
            unique_index = local_node["floor"], local_node["type"], local_node["type_id"]
            local_graph_unique_indices[unique_index] = ni
            local_graph_floor_levels.append(local_node["floor"])
            local_graph_node_cluster.append(local_node["type"])
            local_graph_center.append(local_node["center"])
            local_graph_types.append(local_node["type"])
            local_graph_type_ids.append(local_node["type_id"])

        local_graph_types_onehot = torch.nn.functional.one_hot(
            torch.tensor(local_graph_node_cluster), num_classes=configuration.NUM_CLASSES
        )

        local_graph_node_cluster = torch.tensor(local_graph_node_cluster)
        local_graph_floor_levels = torch.tensor(local_graph_floor_levels)
        local_graph_floor_levels_normalized = local_graph_floor_levels / configuration.NORMALIZATION_FACTOR
        local_graph_center = torch.tensor(local_graph_center)
        local_graph_types = torch.tensor(local_graph_types)
        local_graph_type_ids = torch.tensor(local_graph_type_ids)

        assert len(local_graph_floor_levels) == len(local_graph_nodes)
        assert len(local_graph_types_onehot) == len(local_graph_nodes)

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
        type_ratio = [0] * configuration.NUM_CLASSES
        for global_node in global_graph_nodes:
            type_ratio[global_node["type"]] = global_node["proportion"]

        type_ratio = torch.tensor(type_ratio)

        # process voxel graph data
        voxel_graph_unique_indices = {}
        voxel_graph_features = []
        voxel_graph_types = []
        voxel_graph_types_onehot = []
        voxel_graph_floor_levels = []
        voxel_graph_node_coordinate = []
        voxel_graph_node_dimension = []
        voxel_graph_location = []
        voxel_graph_node_ratio = [0] * configuration.NUM_CLASSES

        voxel_graph_nodes = voxel_graph_data["voxel_node"]
        for vni, voxel_node in enumerate(voxel_graph_nodes):
            voxel_graph_unique_indices[tuple(voxel_node["location"])] = vni

            floor_level, _, _ = voxel_node["location"]
            voxel_graph_floor_levels.append(floor_level)

            voxel_graph_features.append(
                [
                    *[coo / configuration.NORMALIZATION_FACTOR for coo in voxel_node["coordinate"]],
                    *[dim / configuration.NORMALIZATION_FACTOR for dim in voxel_node["dimension"]],
                    *[loc / configuration.NORMALIZATION_FACTOR for loc in voxel_node["location"]],
                ]
            )

            voxel_node_type = voxel_node["type"]
            if voxel_node_type == configuration.VOID_OLD:
                voxel_node_type = configuration.VOID
            elif voxel_node_type == configuration.NOT_ALLOWED_OLD:
                voxel_node_type = configuration.NOT_ALLOWED

            voxel_graph_node_ratio[voxel_node_type] += 1

            voxel_graph_node_coordinate.append(voxel_node["coordinate"])
            voxel_graph_node_dimension.append(voxel_node["dimension"])
            voxel_graph_location.append(voxel_node["location"])
            voxel_graph_types.append(voxel_node_type)

            voxel_graph_types_onehot.append(
                torch.nn.functional.one_hot(
                    torch.tensor(voxel_node_type), num_classes=configuration.NUM_CLASSES
                ).tolist()
            )

        voxel_graph_node_ratio = torch.tensor(voxel_graph_node_ratio) / len(voxel_graph_nodes)

        # compute voxel graph edge indices
        voxel_graph_adjacency_matrix = torch.zeros(size=(len(voxel_graph_nodes), len(voxel_graph_nodes)))
        for vni, voxel_node in enumerate(voxel_graph_nodes):
            ui = voxel_graph_unique_indices[tuple(voxel_node["location"])]

            for voxel_node_neighbor in voxel_node["neighbors"]:
                uj = voxel_graph_unique_indices[tuple(voxel_node_neighbor)]

                voxel_graph_adjacency_matrix[ui][uj] = 1

        voxel_graph_edge_indices = voxel_graph_adjacency_matrix.nonzero().t()

        voxel_graph_types = torch.tensor(voxel_graph_types)
        voxel_graph_types_onehot = torch.tensor(voxel_graph_types_onehot)
        voxel_graph_floor_levels = torch.tensor(voxel_graph_floor_levels)
        voxel_graph_floor_levels_normalized = voxel_graph_floor_levels / configuration.NORMALIZATION_FACTOR
        voxel_graph_features = torch.tensor(voxel_graph_features)
        voxel_graph_node_coordinate = torch.tensor(voxel_graph_node_coordinate)
        voxel_graph_node_dimension = torch.tensor(voxel_graph_node_dimension)
        voxel_graph_location = torch.tensor(voxel_graph_location)

        assert len(voxel_graph_types) == len(voxel_graph_nodes)
        assert len(voxel_graph_types_onehot) == len(voxel_graph_nodes)
        assert len(voxel_graph_floor_levels) == len(voxel_graph_nodes)
        assert len(voxel_graph_features) == len(voxel_graph_nodes)
        assert len(voxel_graph_node_coordinate) == len(voxel_graph_nodes)
        assert len(voxel_graph_node_dimension) == len(voxel_graph_nodes)
        assert len(voxel_graph_location) == len(voxel_graph_nodes)

        local_graph_node_indices_per_level = [[] for _ in range(max(local_graph_floor_levels) + 1)]
        for i, floor_level in enumerate(local_graph_floor_levels):
            local_graph_node_indices_per_level[floor_level].append(i)

        local_graph_data = {
            "far": far,
            "type_ratio": type_ratio,
            "local_graph_node_cluster": local_graph_node_cluster,
            "local_graph_types_onehot": local_graph_types_onehot,
            "local_graph_floor_levels": local_graph_floor_levels,
            "local_graph_floor_levels_normalized": local_graph_floor_levels_normalized,
            "local_graph_edge_indices": local_graph_edge_indices,
            "local_graph_center": local_graph_center,
            "local_graph_types": local_graph_types,
            "local_graph_type_ids": local_graph_type_ids,
            "data_number": data_number,
        }

        voxel_graph_data = voxel_graph_data = {
            "far": far,
            "voxel_graph_types": voxel_graph_types,
            "voxel_graph_types_onehot": voxel_graph_types_onehot,
            "voxel_graph_floor_levels": voxel_graph_floor_levels,
            "voxel_graph_floor_levels_normalized": voxel_graph_floor_levels_normalized,
            "voxel_graph_features": voxel_graph_features,
            "voxel_graph_edge_indices": voxel_graph_edge_indices,
            "voxel_graph_node_coordinate": voxel_graph_node_coordinate,
            "voxel_graph_node_dimension": voxel_graph_node_dimension,
            "voxel_graph_location": voxel_graph_location,
            "voxel_graph_node_ratio": voxel_graph_node_ratio,
            "data_number": data_number,
        }

        return LocalGraphData(local_graph_data), VoxelGraphData(voxel_graph_data)


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

            data_number = "".join([s for s in global_graph_path.split("/")[-1] if s.isdigit()])

            local_graph_data, voxel_graph_data = self.process_data(
                global_graph_data,
                local_graph_data,
                voxel_graph_data,
                self.configuration,
                data_number,
            )

            processed_data_name_voxel = f"{data_number}{self.configuration.VOXEL_DATA_SUFFIX}"
            processed_data_name_local = f"{data_number}{self.configuration.LOCAL_DATA_SUFFIX}"

            torch.save(local_graph_data, os.path.join(self.configuration.SAVE_DATA_PATH, processed_data_name_local))
            torch.save(voxel_graph_data, os.path.join(self.configuration.SAVE_DATA_PATH, processed_data_name_voxel))
