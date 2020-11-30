from model import AffinitiesUnet
from model_config import ModelConf
from predict_pipeline import predict_pipeline

import numpy as np
import gunpowder as gp
import torch
import zarr

import waterz
from watershed_helpers import watershed_from_affinities

from omegaconf import OmegaConf

overrides = OmegaConf.load("model_configs/test.yaml")
model_config: ModelConf = OmegaConf.structured(ModelConf(**overrides))
OmegaConf.set_struct(model_config, True)

# model = AffinitiesUnet(model_config)

# pipeline, request = predict_pipeline(model, "model.checkpoint")

# with gp.build(pipeline):
    # pipeline.request_batch(request)
    # pass

def watershed(results, name):
    results = zarr.open(f"{results}", "r+")
    print(f"results shape: {results[name].shape}")
    predictions = results[name]

    thresholds = [0.5]
    
    voxel_size = predictions.attrs["resolution"]
    offset = predictions.attrs["offset"]
    shape = predictions.shape[-len(voxel_size):]

    spec = gp.ArraySpec(gp.Roi(offset, gp.Coordinate(voxel_size)*shape), voxel_size=voxel_size)
    predictions = gp.Array(predictions, spec)
    
    fragments = watershed_from_affinities(predictions.data)[0]
    thresholds = [0.5]
    segmentations = waterz.agglomerate(
        affs=predictions.data.astype(np.float32),
        fragments=fragments,
        thresholds=thresholds,
    )

    segmentation = next(segmentations)


    results[f"{name}_watershed"] = segmentation
    results[f"{name}_watershed"].attrs["offset"] = predictions.spec.roi.get_offset()
    results[f"{name}_watershed"].attrs["resolution"] = predictions.spec.voxel_size

watershed("model.checkpoint.zarr", "volumes/predictions")