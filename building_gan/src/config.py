import os


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

    LEARNING_RATE_GENERATOR = 0.0002
    LEARNING_RATE_DISCRIMINATOR = 0.0002

    DEVICE = "cuda"


class Configuration(ProgramMap, DataConfiguration, ModelConfiguration):
    def __init__(self):
        pass
