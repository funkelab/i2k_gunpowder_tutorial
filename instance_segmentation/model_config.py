from dataclasses import dataclass, MISSING

from typing import List

@dataclass
class ModelConf:
    num_classes: int = MISSING
    in_channels: int = MISSING
    num_fmaps: int = MISSING
    fmap_inc_factor: int = MISSING
    downsample_factors: List = MISSING
    activation: str = MISSING
    voxel_size: List = MISSING