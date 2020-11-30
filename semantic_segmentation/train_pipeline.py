import gunpowder as gp
import torch

from add_class_labels import AddClassLabels

import math


def train_pipeline(model):

    voxel_size = gp.Coordinate((4, 4, 4))
    input_shape = gp.Coordinate((76, 76, 76))
    input_size = input_shape * voxel_size
    output_shape = gp.Coordinate((36, 36, 36))
    output_size = output_shape * voxel_size
    label_proportions = [
        (0, 0.9419152666666667),
        (1, 0.0516876),
        (2, 0.000764),
        (3, 0.005633133333333333),
    ]
    batch_size = 10

    raw = gp.ArrayKey("RAW")
    predictions = gp.ArrayKey("PREDICTIONS")
    prediction_gradients = gp.ArrayKey("PREDICTION_GRADIENTS")
    intracellular = gp.ArrayKey("INTRACELLULAR")
    labels = gp.ArrayKey("LABELS")

    sources = tuple(
        [
            gp.ZarrSource(
                filename="../MB-Z1213-56.zarr",
                datasets={
                    raw: f"{volume}/raw",
                    intracellular: f"{volume}/intracellular_semantic",
                },
            )
            + AddClassLabels(intracellular, labels)
            + gp.RandomLocation()
            for volume in ["A", "B", "C"]
        ]
    )
    pipeline = (
        sources
        + gp.RandomProvider()
        + gp.Normalize(raw)
        + gp.ElasticAugment((10, 10, 10), (0.1, 0.1, 0.1), (math.pi / 2, math.pi / 2))
        + gp.SimpleAugment(transpose_only=(1, 2))
        + gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1)
        + gp.PreCache()
        + gp.Stack(batch_size)
        + gp.Unsqueeze([raw], axis=1)
        + gp.torch.Train(
            model=model,
            loss=torch.nn.CrossEntropyLoss(
                weight=torch.tensor([1 / p for l, p in label_proportions])
            ),
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-5),
            inputs={"raw": raw},
            outputs={0: predictions},
            loss_inputs={0: predictions, 1: labels},
            gradients={0: prediction_gradients},
            save_every=500,
        )
        + gp.Squeeze([raw], axis=1)
        + gp.Snapshot(
            output_filename="{iteration}.hdf",
            dataset_names={
                raw: "volumes/raw",
                labels: "volumes/labels",
                predictions: "volumes/predictions",
                prediction_gradients: "volumes/prediction_gradients",
            },
            every=500,
        )
    )

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(predictions, output_size)
    request.add(prediction_gradients, output_size)
    return pipeline, request
