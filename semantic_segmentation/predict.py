from model import SemanticUnet
from model_config import SemanticModelConf
from predict_pipeline import predict_pipeline

import gunpowder as gp
import zarr
import numpy as np

from omegaconf import OmegaConf

overrides = OmegaConf.load("model_configs/test.yaml")
model_config: SemanticModelConf = OmegaConf.structured(SemanticModelConf(**overrides))
OmegaConf.set_struct(model_config, True)

model = SemanticUnet(model_config)

pipeline, request = predict_pipeline(model, "model_checkpoint_168000")

with gp.build(pipeline):
    pipeline.request_batch(request)


def argmax(results, name):
    results = zarr.open(f"{results}", "r+")
    print(f"results shape: {results[name].shape}")
    predictions = results[name]

    voxel_size = predictions.attrs["resolution"]
    offset = predictions.attrs["offset"]
    shape = predictions.shape[-len(voxel_size) :]

    spec = gp.ArraySpec(
        gp.Roi(offset, gp.Coordinate(voxel_size) * shape), voxel_size=voxel_size
    )
    predictions = gp.Array(predictions, spec)

    semantic_segmentation = np.argmax(predictions.data, axis=0)

    results[f"{name}_argmax"] = semantic_segmentation
    results[f"{name}_argmax"].attrs["offset"] = predictions.spec.roi.get_offset()
    results[f"{name}_argmax"].attrs["resolution"] = predictions.spec.voxel_size


argmax("model.checkpoint.zarr", "volumes/predictions")