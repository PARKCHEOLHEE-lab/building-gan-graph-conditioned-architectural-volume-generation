import os
import sys
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
from torch_geometric.data import Batch

if os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from building_gan.src.config import Configuration


class VoxelGNNGenerator(nn.Module):
    def __init__(self, configuration: Configuration):
        super().__init__()

        if configuration.GENERATOR_CONV_TYPE == "GCNCONV":
            conv = tgnn.GCNConv
        elif configuration.GENERATOR_CONV_TYPE == "GRAPHCONV":
            conv = tgnn.GraphConv
        elif configuration.GENERATOR_CONV_TYPE == "GATCONV":
            conv = tgnn.GATConv
        elif configuration.GENERATOR_CONV_TYPE == "GATV2CONV":
            conv = tgnn.GATv2Conv
        else:
            raise ValueError(f"Invalid conv_type: {configuration.GENERATOR_CONV_TYPE}")

        self.mlp_encoder = nn.Sequential(
            nn.Linear(
                configuration.LOCAL_GRAPH_DIM + configuration.VOXEL_GRAPH_DIM + configuration.Z_DIM,
                configuration.GENERATOR_HIDDEN_DIM,
            ),
            nn.ReLU(True),
            nn.Linear(configuration.GENERATOR_HIDDEN_DIM, configuration.GENERATOR_HIDDEN_DIM),
            nn.ReLU(True),
        )

        self.encoder = []

        out_channels = configuration.GENERATOR_HIDDEN_DIM
        for _ in range(configuration.GENERATOR_ENCODER_REPEAT):
            self.encoder += [
                (conv(out_channels, out_channels // 2), f"{configuration.INPUT_ARGS} -> x"),
                tgnn.norm.GraphNorm(out_channels // 2),
                nn.ReLU(True),
                nn.Dropout(0.2),
            ]

            out_channels //= 2

        for _ in range(configuration.GENERATOR_ENCODER_REPEAT):
            self.encoder += [
                (conv(out_channels, out_channels * 2), f"{configuration.INPUT_ARGS} -> x"),
                tgnn.norm.GraphNorm(out_channels * 2),
                nn.ReLU(True),
                nn.Dropout(0.2),
            ]

            out_channels *= 2

        self.encoder = tgnn.Sequential(input_args=configuration.INPUT_ARGS, modules=self.encoder)

        self.decoder = nn.Sequential(
            nn.Linear(
                configuration.LOCAL_GRAPH_DIM
                + configuration.VOXEL_GRAPH_DIM
                + configuration.Z_DIM
                + out_channels
                + configuration.GENERATOR_HIDDEN_DIM,
                configuration.GENERATOR_HIDDEN_DIM,
            ),
            nn.BatchNorm1d(configuration.GENERATOR_HIDDEN_DIM),
            nn.ReLU(True),
            nn.Linear(configuration.GENERATOR_HIDDEN_DIM, configuration.GENERATOR_HIDDEN_DIM // 2),
            nn.BatchNorm1d(configuration.GENERATOR_HIDDEN_DIM // 2),
            nn.ReLU(True),
            nn.Linear(configuration.GENERATOR_HIDDEN_DIM // 2, configuration.GENERATOR_HIDDEN_DIM // 4),
            nn.BatchNorm1d(configuration.GENERATOR_HIDDEN_DIM // 4),
            nn.ReLU(True),
            nn.Linear(configuration.GENERATOR_HIDDEN_DIM // 4, configuration.GENERATOR_HIDDEN_DIM // 8),
            nn.BatchNorm1d(configuration.GENERATOR_HIDDEN_DIM // 8),
            nn.ReLU(True),
            nn.Linear(configuration.GENERATOR_HIDDEN_DIM // 8, configuration.NUM_CLASSES),
        )

        self.to(configuration.DEVICE)

    def forward(self, local_graph, voxel_graph, z):
        x_ = torch.cat([local_graph.x[voxel_graph.type], voxel_graph.x, z], dim=-1)
        x = self.mlp_encoder(x_)

        encoded = self.encoder(x=x, edge_index=voxel_graph.edge_index)
        encoded = torch.cat([encoded, x, x_], dim=-1)

        decoded = self.decoder(encoded)

        label_soft = torch.nn.functional.gumbel_softmax(decoded, tau=1.0)
        label_hard = torch.zeros_like(label_soft)
        label_hard.scatter_(-1, label_soft.argmax(dim=1, keepdim=True), 1.0)
        label_hard = label_hard - label_soft.detach() + label_soft

        return label_hard, label_soft


class VoxelGNNDiscriminator(nn.Module):
    def __init__(self, configuration: Configuration):
        super().__init__()

        if configuration.DISCRIMINATOR_CONV_TYPE == "GCNCONV":
            conv = tgnn.GCNConv
        elif configuration.DISCRIMINATOR_CONV_TYPE == "GRAPHCONV":
            conv = tgnn.GraphConv
        elif configuration.DISCRIMINATOR_CONV_TYPE == "GATCONV":
            conv = tgnn.GATConv
        elif configuration.DISCRIMINATOR_CONV_TYPE == "GATV2CONV":
            conv = tgnn.GATv2Conv
        else:
            raise ValueError(f"Invalid conv_type: {configuration.DISCRIMINATOR_CONV_TYPE}")

        self.mlp_encoder = nn.Sequential(
            nn.Linear(
                configuration.LOCAL_GRAPH_DIM + configuration.VOXEL_GRAPH_DIM + configuration.NUM_CLASSES,
                configuration.DISCRIMINATOR_HIDDEN_DIM,
            ),
            nn.ReLU(True),
            nn.Linear(configuration.DISCRIMINATOR_HIDDEN_DIM, configuration.DISCRIMINATOR_HIDDEN_DIM),
            nn.ReLU(True),
        )

        self.encoder = []

        out_channels = configuration.DISCRIMINATOR_HIDDEN_DIM
        for _ in range(configuration.DISCRIMINATOR_ENCODER_REPEAT):
            self.encoder += [
                (conv(out_channels, out_channels // 2), f"{configuration.INPUT_ARGS} -> x"),
                tgnn.norm.GraphNorm(out_channels // 2),
                nn.ReLU(True),
                nn.Dropout(0.2),
            ]

            out_channels //= 2

        for _ in range(configuration.DISCRIMINATOR_ENCODER_REPEAT):
            self.encoder += [
                (conv(out_channels, out_channels * 2), f"{configuration.INPUT_ARGS} -> x"),
                tgnn.norm.GraphNorm(out_channels * 2),
                nn.ReLU(True),
                nn.Dropout(0.2),
            ]

            out_channels *= 2

        self.encoder = tgnn.Sequential(input_args=configuration.INPUT_ARGS, modules=self.encoder)

        self.decoder = nn.Sequential(
            nn.Linear(configuration.DISCRIMINATOR_HIDDEN_DIM, configuration.DISCRIMINATOR_HIDDEN_DIM // 2),
            nn.ReLU(True),
            nn.Linear(configuration.DISCRIMINATOR_HIDDEN_DIM // 2, configuration.DISCRIMINATOR_HIDDEN_DIM // 4),
            nn.ReLU(True),
            nn.Linear(configuration.DISCRIMINATOR_HIDDEN_DIM // 4, configuration.DISCRIMINATOR_HIDDEN_DIM // 8),
            nn.ReLU(True),
            nn.Linear(configuration.DISCRIMINATOR_HIDDEN_DIM // 8, 1),
            nn.Sigmoid(),
        )

        self.to(configuration.DEVICE)

    def forward(self, local_graph, voxel_graph, label_hard):
        x_ = torch.cat([local_graph.x[voxel_graph.type], voxel_graph.x, label_hard], dim=-1)
        x = self.mlp_encoder(x_)

        encoded = self.encoder(x=x, edge_index=voxel_graph.edge_index)
        decoded = self.decoder(encoded)

        return decoded


def compute_gradient_penalty(
    discriminator: VoxelGNNDiscriminator,
    local_graph: Batch,
    voxel_graph: Batch,
    label_soft: Batch,
    lambda_gp: float,
):
    e = torch.rand(voxel_graph.types_onehot.shape[0], 1)
    e = e.to(label_soft.device)

    interpolated = (e * voxel_graph.types_onehot + ((1 - e) * label_soft)).requires_grad_(True)
    interpolated = interpolated.to(label_soft.device)

    interpolated_hard = torch.zeros_like(interpolated)
    interpolated_hard.scatter_(-1, interpolated.argmax(dim=1, keepdim=True), 1.0)

    d_loss_interpolated = discriminator(local_graph, voxel_graph, interpolated)

    gradients = torch.autograd.grad(
        outputs=d_loss_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_loss_interpolated).to(label_soft.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = ((gradients.norm(dim=1) - 1) ** 2).mean() * lambda_gp

    return gradient_penalty
