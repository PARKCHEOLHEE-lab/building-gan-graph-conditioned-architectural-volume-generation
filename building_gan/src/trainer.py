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
from sklearn import metrics
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
        plt.close("all")
        
        self.generator.eval()
        self.discriminator.eval()

        z = torch.randn(1, voxel_graph.num_nodes, self.configuration.Z_DIM).to(self.configuration.DEVICE)
        _, label_hard, _ = self.generator(local_graph, voxel_graph, z)

        voxel_types_generated = label_hard.argmax(dim=1)
        
        f1_score = metrics.f1_score(
            voxel_graph.type.cpu(), 
            voxel_types_generated.cpu(), 
            average=self.configuration.METRICS_AVERAGE,
        )
        
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
        ax_voxel_generated.set_title(f"{epoch}, Generated, (f1: {f1_score:.4f})\n")
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

        self.generator.train()
        self.discriminator.train()

        if show:
            plt.show()

        if to_pil:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            fig = Image.open(buf)
            
            return fig

    @runtime_calculator
    def evaluate_qualitatively(self, epoch, num_samples=2, to_tensor=False, use_test_dataset=False):
        
        train_figs = []
        validation_figs = []
        train_random_indices = torch.randint(len(self.dataloaders.train_dataloader.dataset), size=(num_samples,))

        if use_test_dataset:
            validation_random_indices = torch.randint(
                len(self.dataloaders.test_dataloader.dataset), size=(num_samples,)
            )
        else:
            validation_random_indices = torch.randint(
                len(self.dataloaders.validation_dataloader.dataset), size=(num_samples,)
            )
            
        title_train = None
        if epoch is not None:
            title_train = f"train at epoch: {epoch}\n"
            
        title_validation = None
        if epoch is not None:
            title_validation = f"test at epoch: {epoch}\n" if use_test_dataset else f"validation at epoch: {epoch}\n"

        for ti, vi in zip(train_random_indices, validation_random_indices):
            local_graph, voxel_graph = self.dataloaders.train_dataloader.dataset[ti]
            local_graph = Batch.from_data_list((local_graph,))
            voxel_graph = Batch.from_data_list((voxel_graph,))
            
            if use_test_dataset:
                local_graph_validation, voxel_graph_validation = self.dataloaders.test_dataloader.dataset[vi]
            else:
                local_graph_validation, voxel_graph_validation = self.dataloaders.validation_dataloader.dataset[vi]
            local_graph_validation = Batch.from_data_list((local_graph_validation,))
            voxel_graph_validation = Batch.from_data_list((voxel_graph_validation,))

            assert [set(d) for d in local_graph.data_number] == [set(d) for d in voxel_graph.data_number]
            assert [set(d) for d in local_graph_validation.data_number] == [
                set(d) for d in voxel_graph_validation.data_number
            ]

            if not use_test_dataset:
                train_fig = self._visualize_one(
                    local_graph.to(self.configuration.DEVICE),
                    voxel_graph.to(self.configuration.DEVICE),
                    epoch,
                    title=title_train,
                    to_pil=True,
                )

                train_figs.append(train_fig)

            validation_fig = self._visualize_one(
                local_graph_validation.to(self.configuration.DEVICE),
                voxel_graph_validation.to(self.configuration.DEVICE),
                epoch,
                title=title_validation, 
                to_pil=True,
            )

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
    
    def _compute_gradient_penalty(
        self,
        local_graph: Batch,
        voxel_graph: Batch,
        label_soft: Batch,
    ):
        
        e = torch.rand(voxel_graph.types_onehot.shape[0], 1)
        e = e.to(label_soft.device)

        interpolated = (e * voxel_graph.types_onehot + ((1 - e) * label_soft.squeeze(0))).requires_grad_(True)
        interpolated = interpolated.to(label_soft.device)

        d_loss_interpolated = self.discriminator(local_graph, voxel_graph, interpolated.unsqueeze(0))

        gradients = torch.autograd.grad(
            outputs=d_loss_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_loss_interpolated).to(label_soft.device),
            create_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = ((gradients.norm(dim=1) - 1) ** 2).mean() * self.configuration.LAMBDA_GP

        return gradient_penalty
    
    def _compute_discriminator_loss(self, local_graph, voxel_graph, label_hard, label_soft):
        d_real = self.discriminator(local_graph, voxel_graph, voxel_graph.types_onehot.unsqueeze(0))
        d_fake = self.discriminator(local_graph, voxel_graph, label_hard)
        
        if self.configuration.USE_WGANGP:
            d_loss = d_fake.mean() - d_real.mean()
            d_loss += self._compute_gradient_penalty(local_graph, voxel_graph, label_soft)
        
        else:
            d_loss_real = torch.nn.functional.binary_cross_entropy(d_real, torch.ones_like(d_real))
            d_loss_fake = torch.nn.functional.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))

            d_loss = d_loss_fake + d_loss_real
            
        return d_loss
    
    def _compute_generator_loss(self, local_graph, voxel_graph, logits, label_hard):

        d_fake = self.discriminator(local_graph, voxel_graph, label_hard)
        if self.configuration.USE_WGANGP:
            g_loss_adv = -d_fake.mean()
        
        else:
            g_loss_adv = torch.nn.functional.binary_cross_entropy(d_fake, torch.ones_like(d_fake))
            
        g_loss_adv *= self.configuration.LAMBDA_ADV
        
        g_loss_label = torch.nn.functional.cross_entropy(logits, voxel_graph.type)
        g_loss_label *= self.configuration.LAMBDA_LABEL

        label_ratio_g = label_hard.squeeze(0).sum(dim=0) / voxel_graph.num_nodes
        label_ratio = voxel_graph.types_onehot.sum(dim=0) / voxel_graph.num_nodes

        g_loss_ratio = torch.nn.functional.l1_loss(label_ratio_g[:-2], label_ratio[:-2])
        g_loss_ratio *= self.configuration.LAMBDA_RATIO

        g_loss_ratio_voids = torch.nn.functional.l1_loss(label_ratio_g[-2:], label_ratio[-2:])
        g_loss_ratio_voids *= self.configuration.LAMBDA_RATIO_VOID
        
        voxel_types_generated = label_hard.squeeze(0).argmax(dim=1)
            
        far_unique = []
        far_unique_generated = []

        si = 0
        for gi in range(voxel_graph.num_graphs):
            each_voxel_graph = voxel_graph[gi]
            each_far = each_voxel_graph.x[0][9]
            each_dimension = each_voxel_graph.x[:, 3:6] * self.configuration.NORMALIZATION_FACTOR_DIMENSION
            
            ei = si + each_voxel_graph.num_nodes
            each_voxel_types_generated = voxel_types_generated[si:ei]
            each_dimension_to_use = each_dimension[each_voxel_types_generated != self.configuration.VOID]
            
            each_gfa = (each_dimension_to_use[:, 1] * each_dimension_to_use[:, 2]).sum()
            each_far_generated = each_gfa / each_voxel_graph.site_area[0]
            
            far_unique.append(each_far)
            far_unique_generated.append(each_far_generated)
            
            si = ei
            
        g_loss_far = torch.nn.functional.l1_loss(torch.tensor(far_unique_generated), torch.tensor(far_unique))
        g_loss_far *= self.configuration.LAMBDA_FAR
        
        g_loss = g_loss_adv + g_loss_ratio + g_loss_label + g_loss_ratio_voids + g_loss_far
        
        return g_loss
    
    def _compute_metrics(self, voxel_graph, label_hard):
        
        voxel_types = voxel_graph.type.cpu()
        voxel_types_generated = label_hard.squeeze(0).argmax(dim=1).cpu()
            
        f1_score = metrics.f1_score(
            voxel_types, 
            voxel_types_generated, 
            average=self.configuration.METRICS_AVERAGE,
        )
        
        precision_score = metrics.precision_score(
            voxel_types, 
            voxel_types_generated,
            average=self.configuration.METRICS_AVERAGE,
        )
        
        recall_score = metrics.recall_score(
            voxel_types, 
            voxel_types_generated,
            average=self.configuration.METRICS_AVERAGE,
        )
        
        accuracy_score = metrics.accuracy_score(
            voxel_types,
            voxel_types_generated,
        )
        
        return f1_score, precision_score, recall_score, accuracy_score

    @runtime_calculator
    def _train_each_epoch(self):
        torch.cuda.empty_cache()
        gc.collect()

        g_loss_total_train = []
        d_loss_total_train = []

        f1_score_total_train = []
        precision_score_total_train = []
        recall_score_total_train = []
        accuracy_score_total_train = []

        for local_graph, voxel_graph in self.dataloaders.train_dataloader:
            # Set device
            local_graph = local_graph.to(self.configuration.DEVICE)
            voxel_graph = voxel_graph.to(self.configuration.DEVICE)

            assert [set(d) for d in local_graph.data_number] == [set(d) for d in voxel_graph.data_number]

            # Train discriminator
            for _ in range(self.configuration.N_CRITIC):

                with torch.no_grad():
                    z = torch.randn(1, voxel_graph.num_nodes, self.configuration.Z_DIM).to(self.configuration.DEVICE)
                    logits, label_hard, label_soft = self.generator(local_graph, voxel_graph, z)
                    label_hard = label_hard.unsqueeze(0)
                    label_soft = label_soft.unsqueeze(0)
                
                self.optimizer_discriminator.zero_grad()

                d_loss = self._compute_discriminator_loss(local_graph, voxel_graph, label_hard, label_soft)
                d_loss.backward()
                d_loss_total_train.append(d_loss.item())

                self.optimizer_discriminator.step()
                
            # Train generator
            z = torch.randn(1, voxel_graph.num_nodes, self.configuration.Z_DIM).to(self.configuration.DEVICE)
            logits, label_hard, label_soft = self.generator(local_graph, voxel_graph, z)
            label_hard = label_hard.unsqueeze(0)
            label_soft = label_soft.unsqueeze(0)
            
            self.optimizer_generator.zero_grad()

            g_loss = self._compute_generator_loss(local_graph, voxel_graph, logits, label_hard)
            g_loss.backward()
            g_loss_total_train.append(g_loss.item())

            self.optimizer_generator.step()
            
            f1_score, precision_score, recall_score, accuracy_score = self._compute_metrics(voxel_graph, label_hard)
            f1_score_total_train.append(f1_score)
            precision_score_total_train.append(precision_score)
            recall_score_total_train.append(recall_score)
            accuracy_score_total_train.append(accuracy_score)
            
        g_loss_train = torch.tensor(g_loss_total_train).mean().item()
        d_loss_train = torch.tensor(d_loss_total_train).mean().item()

        f1_score_train = torch.tensor(f1_score_total_train).mean().item()
        precision_score_train = torch.tensor(precision_score_total_train).mean().item()
        recall_score_train = torch.tensor(recall_score_total_train).mean().item()
        accuracy_score_train = torch.tensor(accuracy_score_total_train).mean().item()

        return g_loss_train, d_loss_train, f1_score_train, precision_score_train, recall_score_train, accuracy_score_train

    @runtime_calculator
    @torch.no_grad()
    def _validate_each_epoch(self):
        if self.sanity_checking:
            return 0, 0, 0, 0, 0

        self.generator.eval()
        self.discriminator.eval()

        g_loss_total_validation = []

        f1_score_total_validation = []
        precision_score_total_validation = []
        recall_score_total_validation = []
        accuracy_score_total_validation = []

        for local_graph, voxel_graph in self.dataloaders.validation_dataloader:
            local_graph = local_graph.to(self.configuration.DEVICE)
            voxel_graph = voxel_graph.to(self.configuration.DEVICE)

            assert [set(d) for d in local_graph.data_number] == [set(d) for d in voxel_graph.data_number]

            z = torch.randn(1, voxel_graph.num_nodes, self.configuration.Z_DIM).to(self.configuration.DEVICE)
            logits, label_hard, label_soft = self.generator(local_graph, voxel_graph, z)
            label_hard = label_hard.unsqueeze(0)
            label_soft = label_soft.unsqueeze(0)
            
            g_loss = self._compute_generator_loss(local_graph, voxel_graph, logits, label_hard)
            g_loss_total_validation.append(g_loss.item())
            
            f1_score, precision_score, recall_score, accuracy_score = self._compute_metrics(voxel_graph, label_hard)
            f1_score_total_validation.append(f1_score)
            precision_score_total_validation.append(precision_score)
            recall_score_total_validation.append(recall_score)
            accuracy_score_total_validation.append(accuracy_score)
            
        g_loss_mean_validation = torch.tensor(g_loss_total_validation).mean().item()

        f1_score_validation = torch.tensor(f1_score_total_validation).mean().item()
        precision_score_validation = torch.tensor(precision_score_total_validation).mean().item()
        recall_score_validation = torch.tensor(recall_score_total_validation).mean().item()
        accuracy_score_validation = torch.tensor(accuracy_score_total_validation).mean().item()

        self.generator.train()
        self.discriminator.train()

        return g_loss_mean_validation, f1_score_validation, precision_score_validation, recall_score_validation, accuracy_score_validation


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

        if self.log_dir is None:
            self.log_dir = os.path.join(
                self.configuration.LOG_DIR,
                datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%Y__%H-%M-%S"),
            )

        self.states = {
            "epoch_start": 1,
            "epoch_end": self.configuration.EPOCHS + 1,
            "best_f1_score": 0,
            "f1_score_train": 0,
            "f1_score_validation": 0,
            "precision_score_train": 0,
            "precision_score_validation": 0,
            "recall_score_train": 0,
            "recall_score_validation": 0,
            "accuracy_score_train": 0,
            "accuracy_score_validation": 0,
            "generator": None,
            "discriminator": None,
            "optimizer_generator": None,
            "optimizer_discriminator": None,
            "scheduler_generator": None,

        }

        if os.path.exists(os.path.join(self.log_dir, "states.pt")):
            self.states = torch.load(os.path.join(self.log_dir, "states.pt"))
            self.generator.load_state_dict(self.states["generator"])
            self.discriminator.load_state_dict(self.states["discriminator"])
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
        epoch_end = self.configuration.EPOCHS + 1
        
        best_f1_score = self.states["best_f1_score"]

        clear_output(wait=True)

        for epoch in tqdm(range(epoch_start, epoch_end), desc="Training..."):
            (
                g_loss_train, 
                d_loss_train, 
                f1_score_train, 
                precision_score_train, 
                recall_score_train, 
                accuracy_score_train
            ) = self._train_each_epoch()

            (
                g_loss_mean_validation, 
                f1_score_validation, 
                precision_score_validation, 
                recall_score_validation, 
                accuracy_score_validation
            ) = self._validate_each_epoch()

            self.summary_writer.add_scalar("g_loss_train", g_loss_train, epoch)
            self.summary_writer.add_scalar("d_loss_train", d_loss_train, epoch)
            self.summary_writer.add_scalar("g_loss_validation", g_loss_mean_validation, epoch)
            self.summary_writer.add_scalar("f1_score_train", f1_score_train, epoch)
            self.summary_writer.add_scalar("f1_score_validation", f1_score_validation, epoch)
            self.summary_writer.add_scalar("precision_score_train", precision_score_train, epoch)
            self.summary_writer.add_scalar("precision_score_validation", precision_score_validation, epoch)
            self.summary_writer.add_scalar("recall_score_train", recall_score_train, epoch)
            self.summary_writer.add_scalar("recall_score_validation", recall_score_validation, epoch)
            self.summary_writer.add_scalar("accuracy_score_train", accuracy_score_train, epoch)
            self.summary_writer.add_scalar("accuracy_score_validation", accuracy_score_validation, epoch)

            current_f1_score = (
                f1_score_train * self.configuration.F1_SCORE_TRAIN_WEIGHT
                + f1_score_validation * self.configuration.F1_SCORE_VALIDATION_WEIGHT
            )
            
            if best_f1_score < current_f1_score:
                print(f"Best f1 score updated: {best_f1_score} -> {current_f1_score}")
                best_f1_score = current_f1_score

                if self.sanity_checking:
                    train_fig = self._visualize_one(
                        self.dataloaders.train_dataloader.dataset[0][0].to(self.configuration.DEVICE),
                        self.dataloaders.train_dataloader.dataset[0][1].to(self.configuration.DEVICE),
                        epoch,
                        to_pil=True,
                    )
                    
                    train_fig = np.array(train_fig)
                    train_fig = np.transpose(train_fig, (2, 0, 1))
                    train_fig = torch.tensor(train_fig, dtype=torch.uint8)

                    self.summary_writer.add_image(f"epoch_{epoch}", train_fig, epoch)

                else:
                    torch.save(
                        {
                            "epoch_start": epoch,
                            "epoch_end": epoch_end,
                            "best_f1_score": best_f1_score,
                            "f1_score_train": f1_score_train,
                            "f1_score_validation": f1_score_validation,
                            "recall_score_train": recall_score_train,
                            "recall_score_validation": recall_score_validation,
                            "accuracy_score_train": accuracy_score_train,
                            "accuracy_score_validation": accuracy_score_validation,
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
                    
            else:
                if not self.sanity_checking:
                    states = torch.load(os.path.join(self.log_dir, "states.pt"))
                    states["epoch_start"] = epoch
                    torch.save(states, os.path.join(self.log_dir, "states.pt"))
                                
            self.scheduler_generator.step()
            
    @runtime_calculator
    @torch.no_grad()
    def test(self):
        
        # f1_score_total_test = []
        # precision_score_total_test = []
        # recall_score_total_test = []
        # accuracy_score_total_test = []

        # # test f1 score
        # for local_graph, voxel_graph in self.dataloaders.test_dataloader:
            
        #     local_graph = local_graph.to(self.configuration.DEVICE)
        #     voxel_graph = voxel_graph.to(self.configuration.DEVICE)

        #     assert [set(d) for d in local_graph.data_number] == [set(d) for d in voxel_graph.data_number]

        #     z = torch.randn(1, voxel_graph.num_nodes, self.configuration.Z_DIM).to(self.configuration.DEVICE)
        #     logits, label_hard, label_soft = self.generator(local_graph, voxel_graph, z)
        #     label_hard = label_hard.unsqueeze(0)
        #     label_soft = label_soft.unsqueeze(0)
            
        #     f1_score, precision_score, recall_score, accuracy_score = self._compute_metrics(voxel_graph, label_hard)
        #     f1_score_total_test.append(f1_score)
        #     precision_score_total_test.append(precision_score)
        #     recall_score_total_test.append(recall_score)
        #     accuracy_score_total_test.append(accuracy_score)

        return