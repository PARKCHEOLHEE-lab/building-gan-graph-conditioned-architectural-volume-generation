import os
import torch
import random
import numpy as np


class ProgramMap:
    VOID = -1
    NOT_ALLOWED = -2

    LOBBY_CORRIDOR = 0
    RESTROOM = 1
    STAIRS = 2
    ELEVATOR = 3
    OFFICE = 4
    MECHANICAL_ROOM = 5

    COLORS = {
        VOID: "gray",
        NOT_ALLOWED: "white",
        LOBBY_CORRIDOR: "brown",
        RESTROOM: "red",
        STAIRS: "yellow",
        ELEVATOR: "green",
        OFFICE: "blue",
        MECHANICAL_ROOM: "orange",
    }

    NUM_CLASSES = 6


class DataConfiguration:
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/6types-raw_data"))
    GLOBAL_GRAPH_DATA_PATH = os.path.join(DATA_PATH, "global_graph_data")
    LOCAL_GRAPH_DATA_PATH = os.path.join(DATA_PATH, "local_graph_data")
    VOXEL_GRAPH_DATA_PATH = os.path.join(DATA_PATH, "voxel_data")

    SAVE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/6types-processed_data"))

    NEGATIVE_SAMPLING_MULTIPLIER = 2
    NORMALIZATION_FACTOR = 100
    FLOOR_LEVEL_NORM_FACTOR = 100

    LOCAL_DATA_SUFFIX = "_local.pt"
    VOXEL_DATA_SUFFIX = "_voxel.pt"


class ModelConfiguration:
    BATCH_SIZE = 128
    NUM_WORKERS = 3
    EPOCHS = 1

    HIDDEN_DIM = 128
    PROGRAM_NOISE_DIM = 128
    VOXEL_NOISE_DIM = 128

    PROGRAM_MESSAGE_PASSING_STEPS = 5
    VOXEL_MESSAGE_PASSING_STEPS = 12

    NUM_CLASSES = 6
    BATCH_SIZE = 8

    LEARNING_RATE_GENERATOR = 0.0002
    LEARNING_RATE_DISCRIMINATOR = 0.0002
    LAMBDA = 10

    DEVICE = "cuda"


class Configuration(ProgramMap, DataConfiguration, ModelConfiguration):
    """Configuration for the plan generator"""

    def __init__(self):
        pass

    def to_dict(self):
        raw_config = {**vars(Configuration), **vars(ModelConfiguration), **vars(DataConfiguration)}
        config = {}
        for key, value in raw_config.items():
            if not key.startswith("__") and not callable(value):
                config[key] = value

        return config

    LOG_DIR = os.path.abspath(os.path.join(__file__, "../../runs"))

    @staticmethod
    def set_seed(seed: int = ModelConfiguration.SEED):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

        print("CUDA status")
        print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"  DEVICE: {Configuration.DEVICE} \n")

        print("Seeds status:")
        print(f"  Seeds set for torch        : {torch.initial_seed()}")
        print(f"  Seeds set for torch on GPU : {torch.cuda.initial_seed()}")
        print(f"  Seeds set for numpy        : {seed}")
        print(f"  Seeds set for random       : {seed} \n")

        Configuration.SEED = seed
