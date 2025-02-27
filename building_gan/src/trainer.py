import os
import sys
import time
import pytz
import torch
import datetime
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Hashable, Callable
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch_geometric.data import Batch
from IPython.display import clear_output
from torch.utils.tensorboard import SummaryWriter

if os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from building_gan.src.config import Configuration
from building_gan.src.data import GraphDataLoaders
from building_gan.src.models import VoxelGNNGenerator, VoxelGNNDiscriminator


class TrainerHelper:
    @staticmethod
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

    @staticmethod
    def runtime_calculator(func: Callable) -> Callable:
        """A decorator function for measuring the runtime of another function.

        Args:
            func (Callable): Function to measure

        Returns:
            Callable: Decorator
        """

        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            runtime = end_time - start_time
            print(f"The function {func.__name__} took {runtime} seconds to run.")
            return result

        return wrapper


class Trainer(TrainerHelper):
    def __init__(
        self,
        generator: VoxelGNNGenerator,
        discriminator: VoxelGNNDiscriminator,
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

        self.generator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_generator, patience=5, verbose=True, factor=0.1
        )

        self.has_multiple_gpus = not self.sanity_checking and torch.cuda.device_count() > 1
        if self.has_multiple_gpus:
            self.generator = torch.nn.DataParallel(self.generator)
            self.discriminator = torch.nn.DataParallel(self.discriminator)

        self.summary_writer = SummaryWriter(
            log_dir=os.path.join(
                self.configuration.LOG_DIR,
                datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%Y__%H-%M-%S"),
            )
        )

    @torch.no_grad()
    def _visualize_one(
        self,
        local_graph,
        voxel_graph,
        epoch,
        show=False,
    ):
        self.generator.eval()
        self.discriminator.eval()

        z = torch.randn(voxel_graph.num_nodes, self.configuration.Z_DIM).to(self.configuration.DEVICE)
        label_hard, _ = self.generator(local_graph, voxel_graph, z)

        voxel_types_generated = label_hard.argmax(dim=1)
        accuracy = (voxel_types_generated == voxel_graph.type).float().mean().item()

        fig = plt.figure(figsize=(20, 5))

        ax_graph_local = fig.add_subplot(1, 5, 1, projection="3d")
        ax_voxel_grid = fig.add_subplot(1, 5, 2, projection="3d")
        ax_voxel_groundtruth = fig.add_subplot(1, 5, 3, projection="3d")
        ax_voxel_generated = fig.add_subplot(1, 5, 4, projection="3d")
        ax_legend = fig.add_subplot(1, 5, 5, projection="3d")

        ax_graph_local.set_title("Graph\n")
        ax_voxel_grid.set_title(f"Irregular Voxel Grid (nodes: {voxel_graph.num_nodes})\n")
        ax_voxel_groundtruth.set_title("Ground Truth\n")
        ax_voxel_generated.set_title(f"{epoch}, Generated, (acc: {accuracy:.4f})\n")
        ax_legend.set_title("Legend\n")

        local_graph = local_graph.to("cpu")
        voxel_graph = voxel_graph.to("cpu")
        voxel_types_generated = voxel_types_generated.to("cpu")

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

            voxel_groundtruth = Poly3DCollection(voxel_faces, alpha=0.035 if voxel_type_real in (6, 7) else 1.0)
            voxel_groundtruth.set_facecolor(self.configuration.COLORS[voxel_type_real])
            ax_voxel_groundtruth.add_collection3d(voxel_groundtruth)

            voxel_generated = Poly3DCollection(voxel_faces, alpha=0.035 if voxel_type_generated in (6, 7) else 1.0)
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

        if show:
            plt.show()

        self.generator.train()
        self.discriminator.train()

        return fig

    # @TrainerHelper.runtime_calculator
    def _train_each_epoch(self):
        torch.cuda.empty_cache()

        g_loss_total_train = []
        d_loss_total_train = []
        accuracy_total_train = []

        for _, (local_graph, voxel_graph) in enumerate(self.dataloaders.train_dataloader):
            # Set device
            local_graph = local_graph.to(self.configuration.DEVICE)
            voxel_graph = voxel_graph.to(self.configuration.DEVICE)

            # Train discriminator
            for _ in range(self.configuration.N_CRITIC):
                self.optimizer_discriminator.zero_grad()

                # Generate fake data
                z = torch.randn(voxel_graph.num_nodes, self.configuration.Z_DIM).to(self.configuration.DEVICE)
                label_hard, label_soft = self.generator(local_graph, voxel_graph, z)

                # Compute discriminator loss
                d_real = self.discriminator(local_graph, voxel_graph, voxel_graph.types_onehot)
                d_fake = self.discriminator(local_graph, voxel_graph, label_hard.detach())

                d_loss_real = torch.nn.functional.binary_cross_entropy(d_real, torch.ones_like(d_real))
                d_loss_fake = torch.nn.functional.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))

                gp = self.compute_gradient_penalty(
                    self.discriminator, local_graph, voxel_graph, label_soft, self.configuration.LAMBDA_GP
                )

                d_loss = d_loss_fake + d_loss_real + gp
                d_loss.backward(retain_graph=True)
                self.optimizer_discriminator.step()

                d_loss_total_train.append(d_loss.item())

            # 2. Train Generator
            self.optimizer_generator.zero_grad()
            z = torch.randn(voxel_graph.num_nodes, self.configuration.Z_DIM).to(self.configuration.DEVICE)

            # Generate new fake data
            d_fake = self.discriminator(local_graph, voxel_graph, label_hard)
            g_loss_adv = torch.nn.functional.binary_cross_entropy(d_fake, torch.ones_like(d_fake))

            g_loss_label = torch.nn.functional.cross_entropy(label_soft, voxel_graph.types_onehot.float())
            g_loss_label *= self.configuration.LAMBDA_LABEL

            label_ratio_g = label_hard.sum(dim=0) / voxel_graph.num_nodes
            label_ratio = voxel_graph.types_onehot.sum(dim=0) / voxel_graph.num_nodes

            g_loss_ratio = torch.nn.functional.l1_loss(label_ratio_g, label_ratio)
            g_loss_ratio *= self.configuration.LAMBDA_RATIO

            g_loss_ratio_voids = torch.nn.functional.l1_loss(label_ratio_g[-2:], label_ratio[-2:])
            g_loss_ratio_voids *= self.configuration.LAMBDA_RATIO_VOID
            g_loss = g_loss_adv + g_loss_ratio + g_loss_label + g_loss_ratio_voids
            g_loss_total_train.append(g_loss.item())

            g_loss.backward()
            self.optimizer_generator.step()

            voxel_types_generated = label_hard.argmax(dim=1)
            accuracy = (voxel_types_generated == voxel_graph.type).float().mean().item()
            accuracy_total_train.append(accuracy)

        g_loss_mean_train = torch.tensor(g_loss_total_train).mean().item()
        d_loss_mean_train = torch.tensor(d_loss_total_train).mean().item()
        accuracy_mean_train = torch.tensor(accuracy_total_train).mean().item()

        return g_loss_mean_train, d_loss_mean_train, accuracy_mean_train

    @torch.no_grad()
    def _validate_each_epoch(self):
        if self.sanity_checking:
            return 0, 0

        self.generator.eval()
        self.discriminator.eval()

        g_loss_total_validation = []
        accuracy_total_validation = []

        for _, (local_graph, voxel_graph) in enumerate(self.dataloaders.validation_dataloader):
            local_graph = local_graph.to(self.configuration.DEVICE)
            voxel_graph = voxel_graph.to(self.configuration.DEVICE)

            z = torch.randn(voxel_graph.num_nodes, self.configuration.Z_DIM).to(self.configuration.DEVICE)
            label_hard, label_soft = self.generator(local_graph, voxel_graph, z)

            d_fake = self.discriminator(local_graph, voxel_graph, label_hard)
            g_loss_adv = torch.nn.functional.binary_cross_entropy(d_fake, torch.ones_like(d_fake))

            g_loss_label = torch.nn.functional.cross_entropy(label_soft, voxel_graph.types_onehot.float())
            g_loss_label *= self.configuration.LAMBDA_LABEL

            label_ratio_g = label_hard.sum(dim=0) / voxel_graph.num_nodes
            label_ratio = voxel_graph.types_onehot.sum(dim=0) / voxel_graph.num_nodes

            g_loss_ratio = torch.nn.functional.l1_loss(label_ratio_g, label_ratio)
            g_loss_ratio *= self.configuration.LAMBDA_RATIO

            g_loss_ratio_voids = torch.nn.functional.l1_loss(label_ratio_g[-2:], label_ratio[-2:])
            g_loss_ratio_voids *= self.configuration.LAMBDA_RATIO_VOID
            g_loss = g_loss_adv + g_loss_ratio + g_loss_label + g_loss_ratio_voids
            g_loss_total_validation.append(g_loss.item())

            voxel_types_generated = label_hard.argmax(dim=1)
            accuracy = (voxel_types_generated == voxel_graph.type).float().mean().item()
            accuracy_total_validation.append(accuracy)

        g_loss_mean_validation = torch.tensor(g_loss_total_validation).mean().item()
        accuracy_mean_validation = torch.tensor(accuracy_total_validation).mean().item()

        self.generator.train()
        self.discriminator.train()

        return g_loss_mean_validation, accuracy_mean_validation

    def train(self):
        config_dict = self.configuration.to_dict()
        for key, value in config_dict.items():
            self.summary_writer.add_text(f"configuration/{key}", str(value))

        epoch_start = 1
        epoch_end = self.configuration.EPOCHS + 1
        best_accuracy = 0

        clear_output(wait=True)

        for epoch in tqdm(range(epoch_start, epoch_end), desc="Training..."):
            # Train each epoch
            g_loss_mean_train, d_loss_mean_train, accuracy_mean_train = self._train_each_epoch()

            # Validate each epoch
            g_loss_mean_validation, accuracy_mean_validation = self._validate_each_epoch()

            # Log to tensorboard
            self.summary_writer.add_scalar("g_loss_train", g_loss_mean_train, epoch)
            self.summary_writer.add_scalar("d_loss_train", d_loss_mean_train, epoch)
            self.summary_writer.add_scalar("g_loss_validation", g_loss_mean_validation, epoch)
            self.summary_writer.add_scalar("accuracy_train", accuracy_mean_train, epoch)
            self.summary_writer.add_scalar("accuracy_validation", accuracy_mean_validation, epoch)

            if self.sanity_checking:
                fig = self._visualize_one(
                    self.dataloaders.train_dataloader.dataset[0][0].to(self.configuration.DEVICE),
                    self.dataloaders.train_dataloader.dataset[0][1].to(self.configuration.DEVICE),
                    epoch,
                    show=epoch == 1 or epoch % 1000 == 0,
                )

                self.summary_writer.add_figure(f"epoch_{epoch}", fig, epoch)

            else:
                current_accuracy = accuracy_mean_train * 0.5 + accuracy_mean_validation
                if best_accuracy < current_accuracy:
                    best_accuracy = current_accuracy

                    torch.save(
                        self.generator.state_dict(),
                        os.path.join(self.configuration.CHECKPOINT_DIR, "generator_best.pth"),
                    )

                    torch.save(
                        self.discriminator.state_dict(),
                        os.path.join(self.configuration.CHECKPOINT_DIR, "discriminator_best.pth"),
                    )


if __name__ == "__main__":
    configuration = Configuration()
    configuration.set_seed()

    dataloaders = GraphDataLoaders(configuration=configuration)
    generator = VoxelGNNGenerator(configuration)
    discriminator = VoxelGNNDiscriminator(configuration)

    optimizer_generator = torch.optim.AdamW(generator.parameters(), lr=configuration.LEARNING_RATE_GENERATOR)
    optimizer_discriminator = torch.optim.AdamW(
        discriminator.parameters(), lr=configuration.LEARNING_RATE_DISCRIMINATOR
    )

    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        dataloaders=dataloaders,
        optimizer_generator=optimizer_generator,
        optimizer_discriminator=optimizer_discriminator,
        configuration=configuration,
    )

    trainer.train()
