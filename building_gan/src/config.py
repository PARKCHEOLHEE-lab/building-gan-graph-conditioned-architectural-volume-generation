from enum import Enum, auto


class ProgramMap(Enum):
    VOID = -1
    NOT_ALLOWED = -2

    LOBBY_CORRIDOR = 0
    RESTROOM = auto()
    STAIRS = auto()
    ELEVATOR = auto()
    OFFICE = auto()
    MECHANICAL_ROOM = auto()

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
