import math
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn

from torch_geometric import utils
from torch_geometric.data import Batch


def position_encoder(d_model, max_len=20):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class ProgramGNNBlock(tgnn.MessagePassing):
    def __init__(self, hidden_dim: int):
        super().__init__(aggr="mean")

        self.mlp_message = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU())

        self.mlp_update = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.LeakyReLU())

    def forward(self, x, edge_index, node_cluster, node_ratio):
        return self.propagate(edge_index, x=x, node_cluster=node_cluster, node_ratio=node_ratio)

    def message(self, x_i, x_j):
        # (2) \frac{1}/{|Ne(i)|} \sum_{j \in Ne(i)} MLP^p_{message}([x^t_i, x^t_j]) with aggr="mean"

        return self.mlp_message(torch.cat([x_i, x_j], dim=-1))

    def update(self, aggr_out, x, node_cluster=None, node_ratio=None):
        # (3) c^t_i = Mean_{j \in Cl(i)}({x^t_j})
        #     https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html

        c = utils.scatter(
            src=x,
            index=node_cluster,
            reduce="mean",
        )[node_cluster]

        # (4) MLP^p_{update}([x^t_i, m^t_i, r_{Cl(i)}c^t_i, F])

        c *= node_ratio.sum(dim=1).unsqueeze(0).t()

        return self.mlp_update(torch.cat([x, aggr_out, c], dim=-1))


class ProgramGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, noise_dim: int, num_steps: int):
        super().__init__()

        self.num_steps = num_steps
        self.mlp_encoder = nn.Sequential(nn.Linear(input_dim + noise_dim, hidden_dim), nn.LeakyReLU())

        self.layers = nn.ModuleList([ProgramGNNBlock(hidden_dim) for _ in range(num_steps)])

    def forward(self, local_graph, z):
        # (1) MLP^p_{enc}([x_i, z^p_i, f])
        x = torch.cat([local_graph.x, z], dim=-1)
        x = self.mlp_encoder(x)

        # [(2), (3), (4)] compute message passing T times
        for layer in self.layers:
            x = x + layer(x, local_graph.edge_index, local_graph.node_cluster, local_graph.node_ratio)

        return x


class VoxelGNNBlock(tgnn.MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr="mean")

        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim), nn.LeakyReLU()  # +3 for relative positions
        )

        self.update_mlp = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.LeakyReLU())

    def forward(self, x, edge_index, pos):
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_i, x_j, pos_i, pos_j):
        rel_pos = pos_i - pos_j
        return self.message_mlp(torch.cat([x_i, x_j, rel_pos], dim=-1))

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.attention = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, program_features, voxel_features, cross_edge_index):
        # Get features for connected nodes
        program_node_features = program_features[cross_edge_index[0]]
        voxel_node_features = voxel_features[cross_edge_index[1]]

        # Compute attention scores
        attention_input = torch.cat([program_node_features, voxel_node_features], dim=-1)
        attention_weights = torch.softmax(self.attention(attention_input), dim=0)

        # Apply attention
        attended_features = attention_weights * program_node_features

        # Aggregate for each voxel node
        output_features = torch.zeros_like(voxel_features)
        output_features.index_add_(0, cross_edge_index[1], attended_features)

        return output_features, attention_weights


class BuildingGenerator(nn.Module):
    def __init__(
        self,
        program_input_dim,
        program_noise_dim,
        voxel_input_dim,
        voxel_noise_dim,
        hidden_dim,
        num_steps,
    ):
        super().__init__()

        self.program_gnn = ProgramGNN(program_input_dim, hidden_dim, program_noise_dim, num_steps)
        self.voxel_gnn = VoxelGNNBlock(hidden_dim)
        self.cross_attention = CrossModalAttention(hidden_dim)

        self.voxel_encoder = nn.Sequential(nn.Linear(voxel_input_dim + voxel_noise_dim, hidden_dim), nn.LeakyReLU())

    def forward(self, local_graph, voxel_graph, program_noise, voxel_noise):
        # Program GNN
        program_features = self.program_gnn(local_graph, program_noise)

        # Initial voxel features
        voxel_features = self.voxel_encoder(voxel_graph.x, voxel_noise)

        # Cross attention
        attended_features, attention_weights = self.cross_attention(
            program_features, voxel_features, voxel_graph.cross_edge_index
        )

        # Update voxel features
        voxel_features = voxel_features + attended_features

        # Extract positions from voxel features (assuming first 3 dims are positions)
        positions = voxel_graph.x[:, :3]

        # Voxel GNN
        voxel_features = voxel_features + self.voxel_gnn(voxel_features, voxel_graph.edge_index, positions)

        return voxel_features, attention_weights


def collate_fn(batch):
    local_graphs, voxel_graphs = zip(*batch)

    original_cross_edges = []
    cumulative_sum_local_nodes = 0
    cumulative_sum_voxel_nodes = 0

    for i, (local_graph, voxel_graph) in enumerate(zip(local_graphs, voxel_graphs)):
        cross_edges = voxel_graph.cross_edge_index.clone()

        # adjust cross_edge_index for batch
        if i > 0:
            cross_edges[0] += cumulative_sum_local_nodes
            cross_edges[1] += cumulative_sum_voxel_nodes

        original_cross_edges.append(cross_edges)

        cumulative_sum_local_nodes += local_graph.num_nodes
        cumulative_sum_voxel_nodes += voxel_graph.num_nodes

    local_batch = Batch.from_data_list(local_graphs)
    voxel_batch = Batch.from_data_list(voxel_graphs)
    voxel_batch.cross_edge_index = torch.cat(original_cross_edges, dim=1)

    assert voxel_batch.cross_edge_index[0].max() + 1 == local_batch.num_nodes
    assert voxel_batch.cross_edge_index[1].max() + 1 == voxel_batch.num_nodes

    return local_batch, voxel_batch


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from building_gan.src.data import GraphDataset
    from building_gan.src.config import Configuration

    torch.manual_seed(777)
    torch.cuda.manual_seed_all(777)

    DEVICE = "cuda"

    configuration = Configuration()
    dataset = GraphDataset(configuration=configuration, slicer=128)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=configuration.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Training loop
    generator = BuildingGenerator(
        program_input_dim=dataloader.dataset[0][0].num_features,
        program_noise_dim=configuration.PROGRAM_NOISE_DIM,
        voxel_input_dim=dataloader.dataset[0][1].num_features,
        voxel_noise_dim=configuration.VOXEL_NOISE_DIM,
        hidden_dim=configuration.HIDDEN_DIM,
        num_steps=5,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(generator.parameters(), lr=configuration.LEARNING_RATE)

    for epoch in range(configuration.EPOCHS):
        for local_batch, voxel_batch in dataloader:
            local_batch = local_batch.to(DEVICE)
            voxel_batch = voxel_batch.to(DEVICE)

            # Generate noise
            batch_size = local_batch.num_graphs
            program_noise = torch.randn(local_batch.num_nodes, configuration.PROGRAM_NOISE_DIM).to(DEVICE)
            voxel_noise = torch.randn(voxel_batch.num_nodes, configuration.VOXEL_NOISE_DIM).to(DEVICE)

            # Forward pass
            voxel_features, attention_weights = generator(local_batch, voxel_batch, program_noise, voxel_noise)

            # Loss computation and backward pass would go here
            # ...
