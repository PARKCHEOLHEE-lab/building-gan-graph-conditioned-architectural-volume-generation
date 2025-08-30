import os
import torch
import random
import numpy as np

from typing import List


class ProgramMap:
    VOID_OLD = -1

    LOBBY_CORRIDOR = 0
    RESTROOM = 1
    STAIRS = 2
    ELEVATOR = 3
    OFFICE = 4
    MECHANICAL_ROOM = 5
    VOID = 6

    COLORS = {
        LOBBY_CORRIDOR: "brown",
        RESTROOM: "red",
        STAIRS: "yellow",
        ELEVATOR: "green",
        OFFICE: "blue",
        MECHANICAL_ROOM: "orange",
        VOID: "gray",
    }

    NUM_CLASSES = len(COLORS)


class DataConfiguration:
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/6types-raw_data"))
    GLOBAL_GRAPH_DATA_PATH = os.path.join(DATA_PATH, "global_graph_data")
    LOCAL_GRAPH_DATA_PATH = os.path.join(DATA_PATH, "local_graph_data")
    VOXEL_GRAPH_DATA_PATH = os.path.join(DATA_PATH, "voxel_data")

    SAVE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/6types-processed_data"))

    NORMALIZATION_FACTOR_FLOOR_LEVEL = 10
    NORMALIZATION_FACTOR_DIMENSION = 11
    NORMALIZATION_FACTOR_LOCATION = 11
    NORMALIZATION_FACTOR_COORDINATE = 42
    NORMALIZATION_FACTOR_SITE = 1600

    LOCAL_DATA_SUFFIX = "_local.pt"
    VOXEL_DATA_SUFFIX = "_voxel.pt"


class ModelConfiguration:
    NUM_WORKERS = 3
    EPOCHS = 5000
    SEED = 777

    TRAIN_SPLIT_RATIO = 0.70
    VALIDATION_SPLIT_RATIO = 0.20
    TEST_SPLIT_RATIO = 0.10
    SPLIT_RATIOS = [TRAIN_SPLIT_RATIO, VALIDATION_SPLIT_RATIO, TEST_SPLIT_RATIO]

    DATA_POINT = None
    DATA_SLICER = int(1e10)
    BATCH_SIZE = 512

    N_CRITIC = 5
    LEARNING_RATE_GENERATOR = 0.0002
    LEARNING_RATE_DISCRIMINATOR = 0.0002
    
    SCHEDULER_ETA_MIN_MULTIPLIER = 0.0001

    LAMBDA_RATIO = 10.0
    LAMBDA_RATIO_VOID = 7.0
    LAMBDA_LABEL = 0.0
    LAMBDA_ADV = 1.0
    LAMBDA_FAR = 5.0
    LAMBDA_GP = 10.0
    
    BETAS = (0.5, 0.999)
    
    F1_SCORE_TRAIN_WEIGHT = 0.5
    F1_SCORE_VALIDATION_WEIGHT = 1.0
    
    METRICS_AVERAGE = "weighted"

    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"

    LOG_DIR = os.path.abspath(os.path.join(__file__, "../../runs"))

    GENERATOR_CONV_TYPE = "GATCONV"
    GENERATOR_ENCODER_REPEAT = 7
    GENERATOR_HIDDEN_DIM = 128

    DISCRIMINATOR_CONV_TYPE = "GATCONV"
    DISCRIMINATOR_ENCODER_REPEAT = 3
    DISCRIMINATOR_HIDDEN_DIM = 64

    Z_DIM = 128
    LOCAL_GRAPH_ENCODER_REPEAT = 4
    LOCAL_ENCODER_HIDDEN_DIM = 128
    ENCODER_DROPOUT_RATE = 0.2

    GENERATOR_MLP_ENCODER_REPEAT = 4

    INPUT_ARGS = "x, edge_index"
    
    USE_WGANGP = True


class Configuration(ProgramMap, DataConfiguration, ModelConfiguration):
    """Configuration for the plan generator"""

    def __init__(self, sanity_checking: bool = False):
        self.SANITY_CHECKING = sanity_checking
        if sanity_checking:
            self.BATCH_SIZE = 1
            self.DATA_SLICER = int(1e10)
            self.DATA_POINT = 77

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
