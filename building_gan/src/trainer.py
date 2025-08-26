import os
import io
import gc
import sys
import time
import pytz
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from matplotlib.patches import Patch
from typing import Hashable, Callable
from torch_geometric.data import Batch
from IPython.display import clear_output
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

if os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from building_gan.src.config import Configuration
from building_gan.src.data import GraphDataLoaders
from building_gan.src.models import VoxelGNNGenerator, VoxelGNNDiscriminator


class TrainerHelper:
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

    @torch.no_grad()
    def _visualize_one(
        self,
        local_graph,
        voxel_graph,
        epoch,
        show=False,
        title=None,
        to_pil=False,
    ):
        self.generator.eval()
        self.discriminator.eval()

        z = torch.randn(1, voxel_graph.num_nodes, self.configuration.Z_DIM).to(self.configuration.DEVICE)
        _, label_hard, _ = self.generator(local_graph, voxel_graph, z)

        voxel_types_generated = label_hard.argmax(dim=1)
        accuracy = (voxel_types_generated == voxel_graph.type).float().mean().item()

        fig = plt.figure(figsize=(20, 5))
        if title is not None:
            fig.suptitle(title, fontsize=16)

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

        if to_pil:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            fig = Image.open(buf)

        self.generator.train()
        self.discriminator.train()

        return fig

    @runtime_calculator
    def evaluate_qualitatively(self, epoch, num_samples=2, to_tensor=False):
        plt.close("all")
        
        train_figs = []
        validation_figs = []
        train_random_indices = torch.randint(len(self.dataloaders.train_dataloader.dataset), size=(num_samples,))
        validation_random_indices = torch.randint(
            len(self.dataloaders.validation_dataloader.dataset), size=(num_samples,)
        )

        for ti, vi in zip(train_random_indices, validation_random_indices):
            local_graph, voxel_graph = self.dataloaders.train_dataloader.dataset[ti]
            local_graph = Batch.from_data_list((local_graph,))
            voxel_graph = Batch.from_data_list((voxel_graph,))

            local_graph_validation, voxel_graph_validation = self.dataloaders.validation_dataloader.dataset[vi]
            local_graph_validation = Batch.from_data_list((local_graph_validation,))
            voxel_graph_validation = Batch.from_data_list((voxel_graph_validation,))

            assert [set(d) for d in local_graph.data_number] == [set(d) for d in voxel_graph.data_number]
            assert [set(d) for d in local_graph_validation.data_number] == [
                set(d) for d in voxel_graph_validation.data_number
            ]

            train_fig = self._visualize_one(
                local_graph.to(self.configuration.DEVICE),
                voxel_graph.to(self.configuration.DEVICE),
                epoch,
                title=f"train at epoch: {epoch}\n",
                to_pil=True,
            )

            validation_fig = self._visualize_one(
                local_graph_validation.to(self.configuration.DEVICE),
                voxel_graph_validation.to(self.configuration.DEVICE),
                epoch,
                title=f"validation at epoch: {epoch}\n",
                to_pil=True,
            )

            train_figs.append(train_fig)
            validation_figs.append(validation_fig)

        figs = train_figs + validation_figs

        width, height = figs[0].size
        merged_fig = Image.new("RGB", (width, height * len(figs)))

        for i, fig in enumerate(figs):
            merged_fig.paste(fig, (0, i * height))

        if to_tensor:
            merged_fig = np.array(merged_fig)
            merged_fig = np.transpose(merged_fig, (2, 0, 1))
            merged_fig = torch.tensor(merged_fig, dtype=torch.uint8)

        return merged_fig

    @runtime_calculator
    def _train_each_epoch(self):
        torch.cuda.empty_cache()
        gc.collect()

        g_loss_total_train = []
        d_loss_total_train = []
        accuracy_total_train = []

        accumulation_step = 1

        for _, (local_graph, voxel_graph) in enumerate(self.dataloaders.train_dataloader):
            # Set device
            local_graph = local_graph.to(self.configuration.DEVICE)
            voxel_graph = voxel_graph.to(self.configuration.DEVICE)

            assert [set(d) for d in local_graph.data_number] == [set(d) for d in voxel_graph.data_number]

            # Train discriminator
            for _ in range(self.configuration.N_CRITIC):
                # Generate fake data

                z = torch.randn(1, voxel_graph.num_nodes, self.configuration.Z_DIM).to(self.configuration.DEVICE)
                logits, label_hard, label_soft = self.generator(local_graph, voxel_graph, z)
                label_hard = label_hard.unsqueeze(0)
                label_soft = label_soft.unsqueeze(0)

                # Compute discriminator loss
                d_real = self.discriminator(local_graph, voxel_graph, voxel_graph.types_onehot.unsqueeze(0))
                d_fake = self.discriminator(local_graph, voxel_graph, label_hard.detach())

                d_loss_real = torch.nn.functional.binary_cross_entropy(d_real, torch.ones_like(d_real))
                d_loss_fake = torch.nn.functional.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))

                d_loss = d_loss_fake + d_loss_real
                d_loss /= self.configuration.ACCUMULATION_STEPS
                d_loss.backward(retain_graph=True)

                if (accumulation_step % self.configuration.ACCUMULATION_STEPS == 0) or self.sanity_checking:
                    self.optimizer_discriminator.step()
                    self.optimizer_discriminator.zero_grad()

                d_loss_total_train.append(d_loss.item() * self.configuration.ACCUMULATION_STEPS)

            # Train generator
            d_fake = self.discriminator(local_graph, voxel_graph, label_hard)
            g_loss_adv = torch.nn.functional.binary_cross_entropy(d_fake, torch.ones_like(d_fake))
            g_loss_adv *= self.configuration.LAMBDA_ADV

            g_loss_label = torch.nn.functional.cross_entropy(logits, voxel_graph.type)
            g_loss_label *= self.configuration.LAMBDA_LABEL

            label_ratio_g = label_hard.squeeze(0).sum(dim=0) / voxel_graph.num_nodes
            label_ratio = voxel_graph.types_onehot.sum(dim=0) / voxel_graph.num_nodes

            g_loss_ratio = torch.nn.functional.l1_loss(label_ratio_g, label_ratio)
            g_loss_ratio *= self.configuration.LAMBDA_RATIO

            g_loss_ratio_voids = torch.nn.functional.l1_loss(label_ratio_g[-2:], label_ratio[-2:])
            g_loss_ratio_voids *= self.configuration.LAMBDA_RATIO_VOID

            g_loss = g_loss_adv + g_loss_ratio + g_loss_label + g_loss_ratio_voids
            g_loss /= self.configuration.ACCUMULATION_STEPS
            g_loss.backward()

            if (accumulation_step % self.configuration.ACCUMULATION_STEPS == 0) or self.sanity_checking:
                self.optimizer_generator.step()
                self.optimizer_generator.zero_grad()

            g_loss_total_train.append(g_loss.item() * self.configuration.ACCUMULATION_STEPS)  # Scale back for logging

            voxel_types_generated = label_hard.squeeze(0).argmax(dim=1)
            accuracy = (voxel_types_generated == voxel_graph.type).float().mean().item()
            accuracy_total_train.append(accuracy)

            accumulation_step += 1

        g_loss_mean_train = torch.tensor(g_loss_total_train).mean().item()
        d_loss_mean_train = torch.tensor(d_loss_total_train).mean().item()
        accuracy_mean_train = torch.tensor(accuracy_total_train).mean().item()

        return g_loss_mean_train, d_loss_mean_train, accuracy_mean_train

    @runtime_calculator
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

            assert [set(d) for d in local_graph.data_number] == [set(d) for d in voxel_graph.data_number]

            z = torch.randn(1, voxel_graph.num_nodes, self.configuration.Z_DIM).to(self.configuration.DEVICE)
            logits, label_hard, label_soft = self.generator(local_graph, voxel_graph, z)
            label_hard = label_hard.unsqueeze(0)
            label_soft = label_soft.unsqueeze(0)

            d_fake = self.discriminator(local_graph, voxel_graph, label_hard)
            g_loss_adv = torch.nn.functional.binary_cross_entropy(d_fake, torch.ones_like(d_fake))
            g_loss_adv *= self.configuration.LAMBDA_ADV

            g_loss_label = torch.nn.functional.cross_entropy(logits, voxel_graph.type)
            g_loss_label *= self.configuration.LAMBDA_LABEL

            label_ratio_g = label_hard.squeeze(0).sum(dim=0) / voxel_graph.num_nodes
            label_ratio = voxel_graph.types_onehot.sum(dim=0) / voxel_graph.num_nodes

            g_loss_ratio = torch.nn.functional.l1_loss(label_ratio_g, label_ratio)
            g_loss_ratio *= self.configuration.LAMBDA_RATIO

            g_loss_ratio_voids = torch.nn.functional.l1_loss(label_ratio_g[-2:], label_ratio[-2:])
            g_loss_ratio_voids *= self.configuration.LAMBDA_RATIO_VOID
            g_loss = g_loss_adv + g_loss_ratio + g_loss_label + g_loss_ratio_voids
            g_loss_total_validation.append(g_loss.item())

            voxel_types_generated = label_hard.squeeze(0).argmax(dim=1)
            accuracy = (voxel_types_generated == voxel_graph.type).float().mean().item()
            accuracy_total_validation.append(accuracy)

        g_loss_mean_validation = torch.tensor(g_loss_total_validation).mean().item()
        accuracy_mean_validation = torch.tensor(accuracy_total_validation).mean().item()

        self.generator.train()
        self.discriminator.train()

        return g_loss_mean_validation, accuracy_mean_validation


class Trainer(TrainerHelper):
    def __init__(
        self,
        generator: VoxelGNNGenerator,
        discriminator: VoxelGNNDiscriminator,
        dataloaders: GraphDataLoaders,
        optimizer_generator: torch.optim.Optimizer,
        optimizer_discriminator: torch.optim.Optimizer,
        scheduler_generator: torch.optim.lr_scheduler,
        configuration: Configuration,
        log_dir: str = None,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.dataloaders = dataloaders
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.scheduler_generator = scheduler_generator
        self.configuration = configuration
        self.sanity_checking = self.configuration.SANITY_CHECKING
        self.log_dir = log_dir

        if not self.sanity_checking and torch.cuda.device_count() > 1:
            self.generator = torch.nn.DataParallel(self.generator)
            self.discriminator = torch.nn.DataParallel(self.discriminator)

        if self.log_dir is None:
            self.log_dir = os.path.join(
                self.configuration.LOG_DIR,
                datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%Y__%H-%M-%S"),
            )

        self.states = {
            "epoch_start": 1,
            "epoch_end": self.configuration.EPOCHS + 1,
            "best_accuracy": 0,
            "generator": None,
            "discriminator": None,
            "optimizer_generator": None,
            "optimizer_discriminator": None,
            "scheduler_generator": None,
        }

        if os.path.exists(os.path.join(self.log_dir, "states.pt")):
            self.states = torch.load(os.path.join(self.log_dir, "states.pt"))

            if torch.cuda.device_count() >= 2:
                self.generator.load_state_dict(self.states["generator"])
                self.discriminator.load_state_dict(self.states["discriminator"])
            else:
                generator_state = {k.replace("module.", ""): v for k, v in self.states["generator"].items()}
                discriminator_state = {k.replace("module.", ""): v for k, v in self.states["discriminator"].items()}
                self.generator.load_state_dict(generator_state)
                self.discriminator.load_state_dict(discriminator_state)

            self.optimizer_generator.load_state_dict(self.states["optimizer_generator"])
            self.optimizer_discriminator.load_state_dict(self.states["optimizer_discriminator"])
            self.scheduler_generator.load_state_dict(self.states["scheduler_generator"])

            print(f"Loaded states from {self.log_dir}")

        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i).name)

    def train(self):
        self.summary_writer = SummaryWriter(log_dir=self.log_dir)

        config_dict = self.configuration.to_dict()
        for key, value in config_dict.items():
            self.summary_writer.add_text(f"configuration/{key}", str(value))

        epoch_start = self.states["epoch_start"]
        epoch_end = self.states["epoch_end"]
        best_accuracy = self.states["best_accuracy"]

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
                # Visualize train data
                train_fig = self._visualize_one(
                    self.dataloaders.train_dataloader.dataset[0][0].to(self.configuration.DEVICE),
                    self.dataloaders.train_dataloader.dataset[0][1].to(self.configuration.DEVICE),
                    epoch,
                    show=epoch == 1 or epoch % 1000 == 0,
                )

                self.summary_writer.add_figure(f"epoch_{epoch}", train_fig, epoch)

            else:
                # Save the best states
                current_accuracy = accuracy_mean_train * 0.5 + accuracy_mean_validation
                if best_accuracy < current_accuracy:
                    print(f"Best accuracy updated: {best_accuracy} -> {current_accuracy}")
                    best_accuracy = current_accuracy

                    torch.save(
                        {
                            "epoch_start": epoch,
                            "epoch_end": epoch_end,
                            "best_accuracy": best_accuracy,
                            "generator": self.generator.state_dict(),
                            "discriminator": self.discriminator.state_dict(),
                            "optimizer_generator": self.optimizer_generator.state_dict(),
                            "optimizer_discriminator": self.optimizer_discriminator.state_dict(),
                            "scheduler_generator": self.scheduler_generator.state_dict(),
                        },
                        os.path.join(self.log_dir, "states.pt"),
                    )

                    merged_fig = self.evaluate_qualitatively(epoch, num_samples=2, to_tensor=True)
                    self.summary_writer.add_image(f"epoch_{epoch}", merged_fig, epoch)
                                
            self.scheduler_generator.step()
