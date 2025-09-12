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
    def __init__(self, configuration: Configuration, local_graph_dim: int, voxel_graph_dim: int):
        super().__init__()
        
        self.configuration = configuration
        self.local_graph_dim = local_graph_dim
        self.voxel_graph_dim = voxel_graph_dim

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

        matched_features_encoder_modules = []
        matched_features_encoder_modules += [
            nn.Linear(local_graph_dim, configuration.LOCAL_ENCODER_HIDDEN_DIM),
            nn.LayerNorm(configuration.LOCAL_ENCODER_HIDDEN_DIM),
            nn.LeakyReLU(0.2),
        ]

        for _ in range(configuration.LOCAL_GRAPH_ENCODER_REPEAT):
            matched_features_encoder_modules += [
                nn.Linear(configuration.LOCAL_ENCODER_HIDDEN_DIM, configuration.LOCAL_ENCODER_HIDDEN_DIM),
                nn.LayerNorm(configuration.LOCAL_ENCODER_HIDDEN_DIM),
                nn.LeakyReLU(0.2),
            ]

        self.matched_features_encoder = nn.Sequential(*matched_features_encoder_modules)

        self.mlp_encoder_modules = []
        self.mlp_encoder_modules += [
            nn.Linear(
                configuration.LOCAL_ENCODER_HIDDEN_DIM + voxel_graph_dim + configuration.Z_DIM,
                configuration.GENERATOR_HIDDEN_DIM,
            ),
            nn.LayerNorm(configuration.GENERATOR_HIDDEN_DIM),
            nn.LeakyReLU(0.2),
        ]

        for _ in range(configuration.GENERATOR_MLP_ENCODER_REPEAT):
            self.mlp_encoder_modules += [
                nn.Linear(configuration.GENERATOR_HIDDEN_DIM, configuration.GENERATOR_HIDDEN_DIM),
                nn.LayerNorm(configuration.GENERATOR_HIDDEN_DIM),
                nn.LeakyReLU(0.2),
            ]

        self.mlp_encoder = nn.Sequential(*self.mlp_encoder_modules)

        self.encoder_modules = []
        out_channels = configuration.GENERATOR_HIDDEN_DIM
        for _ in range(configuration.GENERATOR_ENCODER_REPEAT):
            self.encoder_modules += [
                (conv(out_channels, out_channels // 2), f"{configuration.INPUT_ARGS} -> x"),
                tgnn.norm.GraphNorm(out_channels // 2),
                nn.ReLU(True),
                nn.Dropout(0.2),
            ]

            out_channels //= 2

        for _ in range(configuration.GENERATOR_ENCODER_REPEAT):
            self.encoder_modules += [
                (conv(out_channels, out_channels * 2), f"{configuration.INPUT_ARGS} -> x"),
                tgnn.norm.GraphNorm(out_channels * 2),
                nn.ReLU(True),
                nn.Dropout(0.2),
            ]

            out_channels *= 2

        self.encoder = tgnn.Sequential(input_args=configuration.INPUT_ARGS, modules=self.encoder_modules)

        self.decoder = nn.Sequential(
            nn.Linear(
                configuration.LOCAL_ENCODER_HIDDEN_DIM
                + voxel_graph_dim
                + configuration.Z_DIM
                + out_channels
                + configuration.GENERATOR_HIDDEN_DIM,
                configuration.GENERATOR_HIDDEN_DIM,
            ),
            nn.LayerNorm(configuration.GENERATOR_HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(configuration.GENERATOR_HIDDEN_DIM, configuration.GENERATOR_HIDDEN_DIM // 2),
            nn.LayerNorm(configuration.GENERATOR_HIDDEN_DIM // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(configuration.GENERATOR_HIDDEN_DIM // 2, configuration.GENERATOR_HIDDEN_DIM // 4),
            nn.LayerNorm(configuration.GENERATOR_HIDDEN_DIM // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(configuration.GENERATOR_HIDDEN_DIM // 4, configuration.GENERATOR_HIDDEN_DIM // 8),
            nn.LayerNorm(configuration.GENERATOR_HIDDEN_DIM // 8),
            nn.LeakyReLU(0.2),
            nn.Linear(configuration.GENERATOR_HIDDEN_DIM // 8, configuration.NUM_CLASSES),
        )

        self.configuration = configuration

        self.to(configuration.DEVICE)

    def forward(self, local_graph, voxel_graph, z):
        device = local_graph.x.device

        matched_x = torch.zeros((voxel_graph.x.shape[0], local_graph.x.shape[1]), device=device)

        for type_idx in torch.unique(voxel_graph.type):
            voxel_mask = voxel_graph.type == type_idx
            local_mask = local_graph.type == type_idx

            if local_mask.sum() > 0:
                matched_x[voxel_mask] = local_graph.x[local_mask].mean(dim=0)

        encoded_matched_features = self.matched_features_encoder(
            matched_x.to(self.matched_features_encoder[0].weight.device)
        )

        combined_features = torch.cat(
            [
                encoded_matched_features, 
                voxel_graph.x.to(encoded_matched_features.device), 
                z.squeeze(0).to(encoded_matched_features.device)
            ], dim=-1
        )

        x = self.mlp_encoder(combined_features)
        encoded = self.encoder(x=x, edge_index=voxel_graph.edge_index)

        final_features = torch.cat([encoded, x, encoded_matched_features, voxel_graph.x, z.squeeze(0)], dim=-1)

        logits = self.decoder(final_features)

        label_soft = torch.nn.functional.gumbel_softmax(logits, tau=1.0)
        label_hard = torch.zeros_like(label_soft)
        label_hard.scatter_(-1, label_soft.argmax(dim=1, keepdim=True), 1.0)
        label_hard = label_hard - label_soft.detach() + label_soft

        return logits, label_hard, label_soft


class VoxelGNNDiscriminator(nn.Module):
    def __init__(self, configuration: Configuration, local_graph_dim: int, voxel_graph_dim: int):
        super().__init__()
        
        self.configuration = configuration
        self.local_graph_dim = local_graph_dim
        self.voxel_graph_dim = voxel_graph_dim

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
                local_graph_dim + voxel_graph_dim + configuration.NUM_CLASSES,
                configuration.DISCRIMINATOR_HIDDEN_DIM,
            ),
            nn.ReLU(True),
            nn.Linear(configuration.DISCRIMINATOR_HIDDEN_DIM, configuration.DISCRIMINATOR_HIDDEN_DIM),
            nn.ReLU(True),
        )

        self.encoder_modules = []

        out_channels = configuration.DISCRIMINATOR_HIDDEN_DIM
        for _ in range(configuration.DISCRIMINATOR_ENCODER_REPEAT):
            self.encoder_modules += [
                (conv(out_channels, out_channels // 2), f"{configuration.INPUT_ARGS} -> x"),
                tgnn.norm.GraphNorm(out_channels // 2),
                nn.ReLU(True),
                nn.Dropout(0.2),
            ]

            out_channels //= 2

        for _ in range(configuration.DISCRIMINATOR_ENCODER_REPEAT):
            self.encoder_modules += [
                (conv(out_channels, out_channels * 2), f"{configuration.INPUT_ARGS} -> x"),
                tgnn.norm.GraphNorm(out_channels * 2),
                nn.ReLU(True),
                nn.Dropout(0.2),
            ]

            out_channels *= 2

        self.encoder = tgnn.Sequential(input_args=configuration.INPUT_ARGS, modules=self.encoder_modules)

        self.decoder_modules = [
            nn.Linear(configuration.DISCRIMINATOR_HIDDEN_DIM, configuration.DISCRIMINATOR_HIDDEN_DIM // 2),
            nn.ReLU(True),
            nn.Linear(configuration.DISCRIMINATOR_HIDDEN_DIM // 2, configuration.DISCRIMINATOR_HIDDEN_DIM // 4),
            nn.ReLU(True),
            nn.Linear(configuration.DISCRIMINATOR_HIDDEN_DIM // 4, configuration.DISCRIMINATOR_HIDDEN_DIM // 8),
            nn.ReLU(True),
            nn.Linear(configuration.DISCRIMINATOR_HIDDEN_DIM // 8, 1),
        ]

        if not configuration.USE_WGANGP:
            self.decoder_modules.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*self.decoder_modules)

        self.to(configuration.DEVICE)

    def forward(self, local_graph, voxel_graph, label_hard):
        matched_x = torch.zeros((voxel_graph.x.shape[0], local_graph.x.shape[1]), device=local_graph.x.device)

        for type_idx in torch.unique(voxel_graph.type):
            voxel_mask = voxel_graph.type == type_idx
            local_mask = local_graph.type == type_idx

            if local_mask.sum() > 0:
                matched_x[voxel_mask] = local_graph.x[local_mask].mean(dim=0)

        x_ = torch.cat([matched_x, voxel_graph.x, label_hard.squeeze(0)], dim=-1)
        x = self.mlp_encoder(x_)

        encoded = self.encoder(x=x, edge_index=voxel_graph.edge_index)
        decoded = self.decoder(encoded)

        return decoded
