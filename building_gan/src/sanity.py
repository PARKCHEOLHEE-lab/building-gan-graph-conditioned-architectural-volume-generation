import os
import sys
import torch

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

from building_gan.src.config import Configuration
from building_gan.src.data import GraphDataLoaders
from building_gan.src.models import VoxelGNNGenerator, VoxelGNNDiscriminator
from building_gan.src.trainer import Trainer

configuration = Configuration(sanity_checking=True)
configuration.EPOCHS = 5000
configuration.DATA_POINT = 4001
configuration.set_seed()

dataloaders = GraphDataLoaders(configuration=configuration)

_local_graph, _voxel_graph = dataloaders.train_dataloader.dataset[0]

generator = VoxelGNNGenerator(
    configuration=configuration, 
    local_graph_dim=_local_graph.x.shape[1], 
    voxel_graph_dim=_voxel_graph.x.shape[1]
)

discriminator = VoxelGNNDiscriminator(
    configuration=configuration, 
    local_graph_dim=_local_graph.x.shape[1], 
    voxel_graph_dim=_voxel_graph.x.shape[1]
)

optimizer_generator = torch.optim.AdamW(generator.parameters(), lr=configuration.LEARNING_RATE_GENERATOR)
optimizer_discriminator = torch.optim.AdamW(discriminator.parameters(), lr=configuration.LEARNING_RATE_DISCRIMINATOR)
scheduler_generator = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_generator, T_max=configuration.EPOCHS)

trainer = Trainer(
    generator=generator,
    discriminator=discriminator,
    dataloaders=dataloaders,
    optimizer_generator=optimizer_generator,
    optimizer_discriminator=optimizer_discriminator,
    scheduler_generator=scheduler_generator,
    configuration=configuration,
    log_dir=os.path.join(configuration.LOG_DIR, "sanity-checking"),
)

trainer.train()