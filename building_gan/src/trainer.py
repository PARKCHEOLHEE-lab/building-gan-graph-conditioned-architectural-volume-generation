import os
import sys
import torch
import matplotlib.pyplot as plt

from typing import Hashable
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch_geometric.data import Batch

if os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from building_gan.src.config import Configuration
from building_gan.src.data import GraphDataLoaders
from building_gan.src.models import Generator, Discriminator


class TrainerHelper:
    @staticmethod
    def compute_gradient_penalty(discriminator: Discriminator, voxel_batch: Batch, label_hard: Batch, lambda_gp: float):
        e = torch.rand(voxel_batch.types_onehot.shape[0], 1)
        e = e.to(label_hard.device)

        interpolated = (e * voxel_batch.types_onehot + ((1 - e) * label_hard)).requires_grad_(True)
        interpolated = interpolated.to(label_hard.device)

        interpolated_hard = torch.zeros_like(interpolated)
        interpolated_hard.scatter_(-1, interpolated.argmax(dim=1, keepdim=True), 1.0)

        d_loss_interpolated = discriminator(voxel_batch, interpolated)

        gradients = torch.autograd.grad(
            outputs=d_loss_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_loss_interpolated).to(label_hard.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = ((gradients.norm(dim=1) - 1) ** 2).mean() * lambda_gp

        return gradient_penalty


class Trainer(TrainerHelper):
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        dataloaders: GraphDataLoaders,
        optimizer_generator: torch.optim.Optimizer,
        optimizer_discriminator: torch.optim.Optimizer,
        configuration: Configuration,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.dataloaders = dataloaders
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.configuration = configuration
        self.sanity_checking = self.configuration.SANITY_CHECKING

    def visualize(self):
        return

    @torch.no_grad()
    def _visualize_one(self, local_graph, voxel_graph, program_noise, voxel_noise):
        self.generator.eval()

        voxel_types_generated = self.generator(
            local_graph, voxel_graph, program_noise, voxel_noise, onehot_to_type=True
        )

        accuracy = (voxel_types_generated == voxel_graph.type).float().mean().item()

        fig = plt.figure(figsize=(20, 5))

        ax_graph_local = fig.add_subplot(1, 5, 1, projection="3d")
        ax_voxel_grid = fig.add_subplot(1, 5, 2, projection="3d")
        ax_voxel_groundtruth = fig.add_subplot(1, 5, 3, projection="3d")
        ax_voxel_generated = fig.add_subplot(1, 5, 4, projection="3d")
        ax_legend = fig.add_subplot(1, 5, 5, projection="3d")

        ax_graph_local.set_title("Graph\n")
        ax_voxel_grid.set_title("Voxel Grid\n")
        ax_voxel_groundtruth.set_title("Ground Truth\n")
        ax_voxel_generated.set_title(f"Generated, (acc: {accuracy:.4f})\n")
        ax_legend.set_title("Legend\n")

        for src, trg in local_graph.edge_index.t():
            z_src, y_src, x_src = local_graph.center[src].tolist()
            z_trg, y_trg, x_trg = local_graph.center[trg].tolist()

            ax_graph_local.plot([x_src, x_trg], [y_src, y_trg], [z_src, z_trg], color="gray", alpha=0.3, linewidth=0.5)

        for li in range(local_graph.num_nodes):
            z_cen, y_cen, x_cen = local_graph.center[li].tolist()
            node_type = local_graph.type[li].item()

            ax_graph_local.scatter(x_cen, y_cen, z_cen, c=self.configuration.COLORS[node_type], s=10)

        for ni in range(voxel_graph.num_nodes):
            z_coo, y_coo, x_coo = voxel_graph.coordinate[ni].tolist()
            z_dim, y_dim, x_dim = voxel_graph.dimension[ni].tolist()
            voxel_type_real = voxel_graph.type[ni].item()
            voxel_type_generated = voxel_types_generated[ni].int().item()

            voxel_vertices = [
                [x_coo, y_coo, z_coo],
                [x_coo + x_dim, y_coo, z_coo],
                [x_coo + x_dim, y_coo + y_dim, z_coo],
                [x_coo, y_coo + y_dim, z_coo],
                [x_coo, y_coo, z_coo + z_dim],
                [x_coo + x_dim, y_coo, z_coo + z_dim],
                [x_coo + x_dim, y_coo + y_dim, z_coo + z_dim],
                [x_coo, y_coo + y_dim, z_coo + z_dim],
            ]

            voxel_faces = [
                [voxel_vertices[0], voxel_vertices[1], voxel_vertices[2], voxel_vertices[3]],
                [voxel_vertices[4], voxel_vertices[5], voxel_vertices[6], voxel_vertices[7]],
                [voxel_vertices[0], voxel_vertices[1], voxel_vertices[5], voxel_vertices[4]],
                [voxel_vertices[2], voxel_vertices[3], voxel_vertices[7], voxel_vertices[6]],
                [voxel_vertices[1], voxel_vertices[2], voxel_vertices[6], voxel_vertices[5]],
                [voxel_vertices[0], voxel_vertices[3], voxel_vertices[7], voxel_vertices[4]],
            ]

            voxel_grid = Poly3DCollection(voxel_faces, alpha=0.2)
            voxel_grid.set_facecolor("white")
            voxel_grid.set_edgecolor("gray")
            ax_voxel_grid.add_collection3d(voxel_grid)

            voxel_groundtruth = Poly3DCollection(voxel_faces, alpha=0.035 if voxel_type_real in (-1, -2) else 1.0)
            voxel_groundtruth.set_facecolor(self.configuration.COLORS[voxel_type_real])
            ax_voxel_groundtruth.add_collection3d(voxel_groundtruth)

            voxel_generated = Poly3DCollection(voxel_faces, alpha=0.035 if voxel_type_generated in (-1, -2) else 1.0)
            voxel_generated.set_facecolor(self.configuration.COLORS[voxel_type_generated])
            ax_voxel_generated.add_collection3d(voxel_generated)

        program_map_reversed = {
            v: k for k, v in self.configuration.to_dict(class_name=["ProgramMap"]).items() if isinstance(v, Hashable)
        }

        ax_legend.legend(
            handles=[
                Patch(
                    facecolor=self.configuration.COLORS[program],
                    label=program_map_reversed[program].replace("_", " ").title(),
                )
                for program in self.configuration.COLORS.keys()
            ],
            fontsize=7,
            frameon=False,
            loc="upper center",
        )

        max_coords = torch.max(voxel_graph.coordinate + voxel_graph.dimension, dim=0).values.tolist()
        min_coords = torch.min(voxel_graph.coordinate, dim=0).values.tolist()

        for ax in [ax_graph_local, ax_voxel_grid, ax_voxel_groundtruth, ax_voxel_generated, ax_legend]:
            ax.set_box_aspect([1, 1, 1])
            ax.set_proj_type("ortho")
            ax._axis3don = False
            ax.set_xlim(min_coords[2], max_coords[2])
            ax.set_ylim(min_coords[1], max_coords[1])
            ax.set_zlim(min_coords[0], max_coords[0])

        self.generator.train()

    def _train_each_epoch(self):
        g_loss_total = []
        d_loss_total = []

        for i, (local_graph, voxel_graph) in enumerate(self.dataloaders.train_dataloader):
            # Train Discriminator
            self.optimizer_discriminator.zero_grad()

            local_graph = local_graph.to(self.configuration.DEVICE)
            voxel_graph = voxel_graph.to(self.configuration.DEVICE)

            program_noise = torch.randn(local_graph.num_nodes, self.configuration.PROGRAM_NOISE_DIM).to(
                self.configuration.DEVICE
            )

            voxel_noise = torch.randn(voxel_graph.num_nodes, self.configuration.VOXEL_NOISE_DIM).to(
                self.configuration.DEVICE
            )

            label_hard = self.generator(local_graph, voxel_graph, program_noise, voxel_noise)

            d_loss_real = -self.discriminator(voxel_graph, voxel_graph.types_onehot)
            d_loss_fake = self.discriminator(voxel_graph, label_hard)

            gradient_penalty = self.compute_gradient_penalty(
                self.discriminator, voxel_graph, label_hard, self.configuration.LAMBDA_GP
            )

            d_loss = d_loss_real + d_loss_fake + gradient_penalty
            d_loss.backward(retain_graph=True)
            self.optimizer_discriminator.step()

            if (i + 1) % self.configuration.N_CRITIC == 0 or self.sanity_checking:
                # Train Generator
                self.optimizer_generator.zero_grad()

                g_loss = -self.discriminator(voxel_graph, label_hard)
                g_loss.backward()
                self.optimizer_generator.step()

                g_loss_total.append(g_loss.item())
                d_loss_total.append(d_loss.item())

            if self.sanity_checking:
                self._visualize_one(local_graph, voxel_graph, program_noise, voxel_noise)

        return torch.tensor(g_loss_total).mean(), torch.tensor(d_loss_total).mean()

    @torch.no_grad()
    def _validate_each_epoch(self):
        # generator: Generator,
        # discriminator: Discriminator,
        # validation_dataloader: DataLoader,

        return 0, 0

    def train(self):
        for epoch in range(1, self.configuration.EPOCHS + 1):
            g_loss_train, d_loss_train = self._train_each_epoch()

            if not self.sanity_checking:
                g_loss_validation, d_loss_validation = self._validate_each_epoch()
                pass

        return


if __name__ == "__main__":
    configuration = Configuration()
    configuration.set_seed()

    dataloaders = GraphDataLoaders(configuration=configuration)

    generator = Generator(
        program_input_dim=dataloaders.dataset[0][0].num_features,
        voxel_input_dim=dataloaders.dataset[0][1].num_features,
        configuration=configuration,
    )

    discriminator = Discriminator(
        voxel_input_dim=dataloaders.dataset[0][1].num_features,
        configuration=configuration,
    )

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=configuration.LEARNING_RATE_GENERATOR)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=configuration.LEARNING_RATE_DISCRIMINATOR)

    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        dataloaders=dataloaders,
        optimizer_generator=optimizer_generator,
        optimizer_discriminator=optimizer_discriminator,
        configuration=configuration,
    )

    trainer.train()
