import torch
import torch_scatter
import torch.nn as nn
import torch_geometric.nn as tgnn
import torch_geometric.utils as tgutils


class GumbelSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, e, cross_edge_index, tau=1.0):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(e)))
        y = torch.softmax((e + gumbel_noise) / tau, dim=0)

        num_voxels = cross_edge_index[1].max().item() + 1

        # compute gumbel-softmax per index
        y_max = torch_scatter.scatter_max(
            src=y.cpu(),
            index=cross_edge_index[1].cpu(),
            dim_size=num_voxels,
        )[1]

        y_hard = torch.zeros_like(y)
        y_hard.scatter_(-1, y_max.to(y.device), 1.0)
        y_hard = y_hard - y.detach() + y

        return y.unsqueeze(1), y_hard.unsqueeze(1)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_floor_level):
        super().__init__()

        self.d_model = d_model
        self.max_floor_level = max_floor_level

        positional_encoding_table = torch.zeros(self.max_floor_level, self.d_model)

        # PE_{pos, 2i} = sin(pos / 10000^{2i / d_{model}})
        # PE_{pos, 2i+1} = cos(pos / 10000^{2i / d_{model}})

        pos = torch.arange(0, self.max_floor_level).unsqueeze(1)
        _2i = torch.arange(0, self.d_model)[0::2]

        positional_encoding_table[:, 0::2] = torch.sin(pos / 10000 ** (_2i / self.d_model))
        positional_encoding_table[:, 1::2] = torch.cos(pos / 10000 ** (_2i / self.d_model))

        self.register_buffer("positional_encoding_table", positional_encoding_table)

    def forward(self, x, voxel_level):
        return x + self.positional_encoding_table[voxel_level]


class PointerBasedCrossModalModule(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.mlp_program = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_voxel = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_mask = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, 1))
        self.gumbel_softmax = GumbelSoftmax()
        self.theta = nn.Parameter(torch.Tensor(hidden_dim, 1))
        nn.init.xavier_normal_(self.theta)

    def forward(self, x, v, cross_edge_index):
        # Conceptually, these pointer modules should gradually improve the design.
        # Note that the outputattk indicates which program node is associated to the
        # program type of the voxel node, instead of merely the program type prediction.

        x_selected = x[cross_edge_index[0]]
        v_selected = v[cross_edge_index[1]]

        # (9) mask_k = \sigma(MLP(v^t_k))
        mask = self.mlp_mask(v).sigmoid()

        # (10) e_{k, i} = \theta^T \tanh(W_x x^T_i + W_v v^t_k) (10)
        e = self.theta.t() * torch.tanh(self.mlp_program(x_selected) + self.mlp_voxel(v_selected))
        e = e.sum(dim=1)

        # (11) att_k = \text{gumbel softmax}(e_k)
        attention_soft, attention_hard = self.gumbel_softmax(e, cross_edge_index)

        # (12) v^{t+1}_k = v^t_k + mask_k \sum_i att_{k,i} x^T_i
        summed = tgutils.scatter(src=x_selected * attention_soft, index=cross_edge_index[1], reduce="sum")
        v = v + mask * summed

        return v, mask, attention_soft, attention_hard


class ProgramGNNBlock(tgnn.MessagePassing):
    def __init__(self, hidden_dim: int):
        super().__init__(aggr="mean")

        self.mlp_message = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU())

        self.mlp_update = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.LeakyReLU())

    def forward(self, x, edge_index, node_cluster, node_ratio):
        return self.propagate(edge_index, x=x, node_cluster=node_cluster, node_ratio=node_ratio)

    def message(self, x_i, x_j):
        # (2) m^t_i = \frac{1}/{|Ne(i)|} \sum_{j \in Ne(i)} MLP^p_{message}([x^t_i, x^t_j]) with aggr="mean"
        return self.mlp_message(torch.cat([x_i, x_j], dim=-1))

    def update(self, aggr_out, x, node_cluster=None, node_ratio=None):
        # (3) c^t_i = Mean_{j \in Cl(i)}({x^t_j})
        #     https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html
        c = tgutils.scatter(
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
        # (1) x^0_i = MLP^p_{enc}([x_i, z^p_i, f])
        x = torch.cat([local_graph.x, z], dim=-1)
        x = self.mlp_encoder(x)

        # [(2), (3), (4)]
        for layer in self.layers:
            # (4) x^{t+1}_i = x^t_i + MLP^p_{update}([x^t_i, m^t_i, r_{Cl(i)}c^t_i, F])
            x = x + layer(x, local_graph.edge_index, local_graph.node_cluster, local_graph.node_ratio)

        return x


class VoxelGNNBlock(tgnn.MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr="sum")

        self.mlp_message = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.LeakyReLU())

        self.mlp_update = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.LeakyReLU())

    def forward(self, v, edge_index, pos):
        return self.propagate(edge_index, v=v, pos=pos)

    def message(self, v_i, v_j, pos_i, pos_j):
        # (6) n^t_k = \sum_{l \in Ne(k)} MLP^v_{message}([v^t_k, v^t_l, p_k - p_l])
        return self.mlp_message(torch.cat([v_i, v_j, pos_i - pos_j], dim=-1))

    def update(self, aggr_out, v):
        # (7) MLP_{update}(v^t_k, n^t_k)
        return self.mlp_update(torch.cat([v, aggr_out], dim=-1))


class VoxelGNN(tgnn.MessagePassing):
    def __init__(self, input_dim: int, hidden_dim: int, noise_dim: int, num_steps: int):
        super().__init__()

        self.num_steps = num_steps
        self.mlp_encoder = nn.Sequential(nn.Linear(input_dim + noise_dim, hidden_dim), nn.LeakyReLU())
        self.positional_encoder = PositionalEncoder(hidden_dim, 100)
        self.pointer_based_cross_modal_module = PointerBasedCrossModalModule(hidden_dim)

        self.layers = nn.ModuleList([VoxelGNNBlock(hidden_dim) for _ in range(num_steps)])

    def forward(self, voxel_graph, z, x, skip_pointer=False):
        # (5) v^0_k MLP^v_{enc}([v_k, z^v_k]) + PE(story_k)
        v = self.mlp_encoder(torch.cat([voxel_graph.x, z], dim=-1))
        v = self.positional_encoder(v, voxel_graph.voxel_level)

        # [(6), (7), (8), (9), (10), (11), (12)]
        for li, layer in enumerate(self.layers):
            v = v + layer(
                v,
                voxel_graph.edge_index,
                pos=self.positional_encoder.positional_encoding_table[voxel_graph.voxel_level],
            )

            # "baseline model uses 12 steps of message passing and call the pointer module once every 2 steps"
            if (li + 1) % 2 == 0 and not skip_pointer:
                v, mask, attention_soft, attention_hard = self.pointer_based_cross_modal_module(
                    x, v, voxel_graph.cross_edge_index
                )

        if skip_pointer:
            mask = None
            attention_soft = None
            attention_hard = None

        return v, mask, attention_soft, attention_hard


class Generator(nn.Module):
    def __init__(
        self,
        program_input_dim,
        voxel_input_dim,
        configuration,
    ):
        super().__init__()

        self.program_gnn = ProgramGNN(
            program_input_dim,
            configuration.HIDDEN_DIM,
            configuration.PROGRAM_NOISE_DIM,
            configuration.PROGRAM_MESSAGE_PASSING_STEPS,
        )

        self.voxel_gnn = VoxelGNN(
            voxel_input_dim,
            configuration.HIDDEN_DIM,
            configuration.VOXEL_NOISE_DIM,
            configuration.VOXEL_MESSAGE_PASSING_STEPS,
        )

        self.pointer_based_cross_model_module = PointerBasedCrossModalModule(configuration.HIDDEN_DIM)

        self.to(configuration.DEVICE)

    def forward(self, local_graph, voxel_graph, program_noise, voxel_noise):
        # computes equations (1), (2), (3), (4)
        x = self.program_gnn(local_graph, program_noise)

        # computes equations (5), (6), (7), (8), (9), (10), (11), (12)
        v, mask, attention_soft, attention_hard = self.voxel_gnn(voxel_graph, voxel_noise, x)

        print(mask, attention_soft, attention_hard)

        # mask_hard = (mask > 0.3).int()
        # attention_hard = attention * mask_hard

        return v


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from building_gan.src.data import GraphDataset
    from building_gan.src.config import Configuration

    torch.manual_seed(777)
    torch.cuda.manual_seed_all(777)

    configuration = Configuration()
    dataset = GraphDataset(configuration=configuration, slicer=128)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=configuration.NUM_WORKERS,
        collate_fn=GraphDataset.collate_fn,
    )

    # Training loop
    generator = Generator(
        program_input_dim=dataloader.dataset[0][0].num_features,
        voxel_input_dim=dataloader.dataset[0][1].num_features,
        configuration=configuration,
    )

    optimizer = torch.optim.Adam(generator.parameters(), lr=configuration.LEARNING_RATE)

    for epoch in range(configuration.EPOCHS):
        for local_batch, voxel_batch in dataloader:
            local_batch = local_batch.to(configuration.DEVICE)
            voxel_batch = voxel_batch.to(configuration.DEVICE)

            # Generate noise
            batch_size = local_batch.num_graphs
            program_noise = torch.randn(local_batch.num_nodes, configuration.PROGRAM_NOISE_DIM).to(configuration.DEVICE)
            voxel_noise = torch.randn(voxel_batch.num_nodes, configuration.VOXEL_NOISE_DIM).to(configuration.DEVICE)

            # Forward pass
            voxel_features, attention_weights = generator(local_batch, voxel_batch, program_noise, voxel_noise)

            # Loss computation and backward pass would go here
            # ...
