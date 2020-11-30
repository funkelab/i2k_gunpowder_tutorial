import gunpowder as gp
import torch
import numpy as np
import daisy

from helpers import Squeeze, Unsqueeze

def predict_pipeline(model, checkpoint):
    grow = (36, 36, 36)
    voxel_size = gp.Coordinate((4, 4, 4))
    input_shape = gp.Coordinate((76, 76, 76)) + grow
    input_size = input_shape * voxel_size
    output_shape = gp.Coordinate((36, 36, 36)) + grow
    output_size = output_shape * voxel_size
    context = (input_size - output_size) / 2

    model.load_state_dict(torch.load(checkpoint)["model"])
    model.eval()
    raw = gp.ArrayKey("RAW")
    predictions = gp.ArrayKey("PREDICTIONS")

    reference_request = gp.BatchRequest()
    reference_request.add(raw, input_size)
    reference_request.add(predictions, output_size)

    source = gp.ZarrSource(
                filename="../MB-Z1213-56.zarr",
                datasets={
                    raw: f"TEST/raw",
                },)

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
    total_output_roi = total_input_roi.grow(-context, -context)

    daisy.prepare_ds(f"{checkpoint}.zarr", "volumes/predictions", daisy.Roi(total_output_roi.get_offset(), total_output_roi.get_shape()), (4, 4, 4), np.float32, write_size=output_size, num_channels=3)

    pipeline = (
        source
        + gp.Normalize(raw)
        + Unsqueeze([raw])
        + Unsqueeze([raw])
        + gp.torch.Predict(
            model=model,
            inputs={"raw": raw},
            outputs={0: predictions},
        )
        + Squeeze([raw, predictions])
        + Squeeze([raw])
        + gp.ZarrWrite(
            output_dir="./",
            output_filename=f"{checkpoint}.zarr",
            dataset_names={
                raw: "volumes/raw",
                predictions: "volumes/predictions",
            },
            dataset_dtypes={predictions: gp.ArraySpec(roi=total_output_roi)}
        )
        + gp.Scan(reference_request)
    )

    total_request = gp.BatchRequest()
    total_request[raw] = gp.ArraySpec(roi=total_input_roi)
    total_request[predictions] = gp.ArraySpec(roi=total_output_roi)

    return pipeline, total_request