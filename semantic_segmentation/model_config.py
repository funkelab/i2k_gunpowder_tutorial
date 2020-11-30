from omegaconf import MISSING

from dataclasses import dataclass
from typing import List


@dataclass
class SemanticModelConf:
    num_classes: int = MISSING
    in_channels: int = MISSING
    num_fmaps: int = MISSING
    fmap_inc_factor: int = MISSING
    downsample_factors: List = MISSING
    activation: str = MISSING
    voxel_size: List = MISSING
