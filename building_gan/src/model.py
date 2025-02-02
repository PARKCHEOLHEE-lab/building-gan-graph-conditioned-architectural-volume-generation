import torch
import torch.nn as nn
import torch_geometric.nn as tgnn


class ProgramGNNBlock(tgnn.MessagePassing):
    def __init__(self):
        super().__init__(self, aggr="mean")

        self.message_mlp = nn.Linear()
        self.update_mlp = nn.Linear()

    def message(self, x_i, x_j):
        return

    def update(self):
        return


class ProgramGNN(nn.Module):
    def __init__(self, steps, program_x_dim, program_z_dim, hidden_dim):
        super().__init__()

        self.steps = steps
        self.program_x_dim = program_x_dim
        self.program_z_dim = program_z_dim
        self.hidden_dim = hidden_dim

        self.program_encoder = nn.Sequential(
            nn.Linear(self.program_x_dim + program_z_dim, hidden_dim),
            nn.LeakyReLU(),
        )

        self.program_blocks = []

    def forward(self, data, program_z, f):
        # (1) MLP^p_{enc}([x_i, z^p_i, F])
        x = torch.cat([data.local_graph_types, data.local_graph_floor_levels, program_z, f])
        x = self.program_encoder(x)

        for _ in range(self.steps):
            pass

        return x


class VoxelGNN(tgnn.MessagePassing):
    def __init__(self):
        super().__init__()


class BuildingGenerator:
    pass


class BuildingDiscriminator:
    pass


def collate_fn(batch):
    local_graphs, voxel_graphs = zip(*batch)

    # Local graph batch
    local_batch = Batch.from_data_list(
        [
            Data(
                x=local_graph.x,
                edge_index=local_graph.edge_index,
            )
            for local_graph in local_graphs
        ]
    )

    # Voxel graph batch
    voxel_batch = Batch.from_data_list(
        [
            Data(
                x=voxel_graph.x,
                edge_index=voxel_graph.edge_index,
                cross_edge_index=voxel_graph.voxel_and_local_cross_edge_indices,  # Rename for clarity
            )
            for voxel_graph in voxel_graphs
        ]
    )

    return local_batch, voxel_batch


if __name__ == "__main__":
    from torch_geometric.data import Batch, Data

    # from torch_geometric.loader import DataLoader
    from torch.utils.data import DataLoader
    from building_gan.src.data import GraphDataset
    from building_gan.src.config import Configuration

    torch.manual_seed(777)

    dataset = GraphDataset(Configuration(), slicer=128)
    dataloader = DataLoader(
        dataset,
        batch_size=12,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    try:
        for local_graph, voxel_graph in dataloader:
            print("\nSuccessful batch:")
            print(f"Local batch: x={local_graph.x.shape}")
            print(f"Voxel batch: x={voxel_graph.x.shape}")
            break
    except Exception as e:
        print(f"\nError during iteration: {str(e)}")

    program_gnn = ProgramGNN(
        steps=1,
        program_x_dim=10,
        program_z_dim=128,
        hidden_dim=128,
    )
