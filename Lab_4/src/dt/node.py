
from dataclasses import dataclass
from enum import Enum

__all__ = ["NodeType", "Node"]


class NodeType(Enum):
    INTERNAL = 0
    LEAF = 1


@dataclass
class Node:
    classification_class: int = None
    type: NodeType = None
    feature_split: int = None
    threshold: float = None
    depth: int = 0

    left: "Node" = None
    right: "Node" = None
