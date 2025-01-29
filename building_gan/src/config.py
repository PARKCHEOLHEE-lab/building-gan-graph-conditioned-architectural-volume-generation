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


class DataConfiguration:
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/6types-raw_data"))
    GLOBAL_GRAPH_DATA_PATH = os.path.join(DATA_PATH, "global_graph_data")
    LOCAL_GRAPH_DATA_PATH = os.path.join(DATA_PATH, "local_graph_data")
    VOXEL_DATA_PATH = os.path.join(DATA_PATH, "voxel_data")

    NEGATIVE_SAMPLING_MULTIPLIER = 2
    NORMALIZATION_FACTOR = 100


class Configuration(ProgramMap, DataConfiguration):
    def __init__(self):
        pass
