import os
import torch
import random
import numpy as np

from typing import List


class ProgramMap:
    VOID_OLD = -1
    NOT_ALLOWED_OLD = -2

    LOBBY_CORRIDOR = 0
    RESTROOM = 1
    STAIRS = 2
    ELEVATOR = 3
    OFFICE = 4
    MECHANICAL_ROOM = 5
    VOID = 6
    NOT_ALLOWED = 7

    COLORS = {
        LOBBY_CORRIDOR: "brown",
        RESTROOM: "red",
        STAIRS: "yellow",
        ELEVATOR: "green",
        OFFICE: "blue",
        MECHANICAL_ROOM: "orange",
        VOID: "gray",
        NOT_ALLOWED: "white",
    }

    NUM_CLASSES = 8


class DataConfiguration:
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/6types-raw_data"))
    GLOBAL_GRAPH_DATA_PATH = os.path.join(DATA_PATH, "global_graph_data")
    LOCAL_GRAPH_DATA_PATH = os.path.join(DATA_PATH, "local_graph_data")
    VOXEL_GRAPH_DATA_PATH = os.path.join(DATA_PATH, "voxel_data")

    SAVE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/6types-processed_data"))

    NEGATIVE_SAMPLING_MULTIPLIER = 2
    NORMALIZATION_FACTOR = 50

    LOCAL_DATA_SUFFIX = "_local.pt"
    VOXEL_DATA_SUFFIX = "_voxel.pt"


class ModelConfiguration:
    NUM_WORKERS = 3
    EPOCHS = 3000
    SEED = 777

    HIDDEN_DIM = 128
    PROGRAM_NOISE_DIM = 32
    VOXEL_NOISE_DIM = 32

    PROGRAM_MESSAGE_PASSING_STEPS = 4
    VOXEL_MESSAGE_PASSING_STEPS = 12

    TRAIN_SPLIT_RATIO = 0.70
    VALIDATION_SPLIT_RATIO = 0.20
    TEST_SPLIT_RATIO = 0.10
    SPLIT_RATIOS = [TRAIN_SPLIT_RATIO, VALIDATION_SPLIT_RATIO, TEST_SPLIT_RATIO]

    DATA_POINT = None
    BATCH_SIZE = 32
    SANITY_CHECKING = True
    if SANITY_CHECKING:
        DATA_POINT = 777
        BATCH_SIZE = 1

    N_CRITIC = 5
    LEARNING_RATE_GENERATOR = 0.0002
    LEARNING_RATE_DISCRIMINATOR = 0.0002
    BETAS = (0.5, 0.999)

    LAMBDA_GP = 10.0
    LAMBDA_RATIO = 1.0
    LAMBDA_LABEL = 1.0

    DEVICE = "cuda"

    LOG_DIR = os.path.abspath(os.path.join(__file__, "../../runs"))

    GENERATOR_CONV_TYPE = "GATCONV"
    GENERATOR_ENCODER_REPEAT = 5
    GENERATOR_HIDDEN_DIM = 128

    DISCRIMINATOR_CONV_TYPE = "GATCONV"
    DISCRIMINATOR_ENCODER_REPEAT = 3
    DISCRIMINATOR_HIDDEN_DIM = 64

    Z_DIM = 128
    LOCAL_GRAPH_DIM = 18
    VOXEL_GRAPH_DIM = 11
    ENCODER_DROPOUT_RATE = 0.2

    INPUT_ARGS = "x, edge_index"


class Configuration(ProgramMap, DataConfiguration, ModelConfiguration):
    """Configuration for the plan generator"""

    def __init__(self):
        pass

    def to_dict(self, class_name: List[str] = ["ProgramMap", "DataConfiguration", "ModelConfiguration"]):
        config_dict = {
            "ProgramMap": ProgramMap,
            "DataConfiguration": DataConfiguration,
            "ModelConfiguration": ModelConfiguration,
        }

        raw_config = {}
        for class_name in class_name:
            raw_config.update(vars(config_dict[class_name]))

        config = {}
        for key, value in raw_config.items():
            if not key.startswith("__") and not callable(value):
                config[key] = value

        return config

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
