import daisy
import neuroglancer
import numpy as np
import zarr
import h5py
import torch

import logging

logger = logging.getLogger(__name__)


def add_layer(context, array, name, voxel_size, array_offset, visible=True, **kwargs):
    print(f"array.shape: {array.shape}, voxel_size: {voxel_size}, array_offset: {array_offset}")
    array_dims = len(array.shape)
    spatial_dims = len(voxel_size)
    spatial_dims = min(spatial_dims, array_dims)
    voxel_size = voxel_size[:spatial_dims]
    array_offset = array_offset[:spatial_dims]
    # assert spatial_dims == 3
    channel_dims = array_dims - spatial_dims
    print(array_dims, spatial_dims, voxel_size, array.shape)
    attrs = {
        1: {"names": ["x"], "units": "nm", "scales": voxel_size},
        2: {"names": ["y", "x"], "units": "nm", "scales": voxel_size},
        3: {"names": ["z", "y", "x"], "units": "nm", "scales": voxel_size},
        4: {
            "names": ["c^", "z", "y", "x"],
            "units": ["", "nm", "nm", "nm"],
            "scales": [1, *voxel_size],
        },
        5: {
            "names": ["c^", "b^", "z", "y", "x"],
            "units": ["", "", "nm", "nm", "nm"],
            "scales": [1, 1, *voxel_size],
        },
    }
    dimensions = neuroglancer.CoordinateSpace(**attrs[array_dims])
    offset = np.array((0,) * (channel_dims) + array_offset)
    offset = offset // attrs[array_dims]["scales"]

    channels = ",".join(
        [
            f"toNormalized(getDataValue({i}))" if i < array.shape[0] else "0"
            for i in range(3)
        ]
    )
    shader_4d = (
        """
void main() {
  emitRGB(vec3(%s));
}
"""
        % channels
    )
    shader_3d = """
void main () {
  emitGrayscale(toNormalized(getDataValue()));
}"""

    layer = neuroglancer.LocalVolume(
        data=array, dimensions=dimensions, voxel_offset=tuple(offset)
    )

    if array.dtype == np.dtype(np.uint64):
        context.layers.append(name=name, layer=layer, visible=visible)
    else:
        context.layers.append(
            name=name,
            layer=layer,
            visible=visible,
            shader=shader_4d if array_dims == 4 else shader_3d,
            **kwargs,
        )


def get_volumes(h5_file, path=None):
    datasets = []
    try:
        if path is None:
            for key in h5_file.keys():
                datasets += get_volumes(h5_file, f"{key}")
            return datasets
        else:
            for key in h5_file.get(path, {}).keys():
                datasets += get_volumes(h5_file, f"{path}/{key}")
            return datasets
    except AttributeError:
        return [path]


def add_container(
    context,
    snapshot_file,
    name_prefix="",
    volume_paths=[None],
    graph_paths=[None],
    graph_node_attrs=None,
    graph_edge_attrs=None,
    # mst=["embedding", "fg_maxima"],
    mst=None,
    roi=None,
    modify=None,
    dims=3,
    array_offset=None,
    voxel_size=None,
):
    if snapshot_file.name.endswith(".zarr") or snapshot_file.name.endswith(".n5"):
        f = zarr.open(str(snapshot_file.absolute()), "r")
    elif snapshot_file.name.endswith(".h5") or snapshot_file.name.endswith(".hdf"):
        f = h5py.File(str(snapshot_file.absolute()), "r")
    with f as dataset:
        volumes = []
        for volume in volume_paths:
            volumes += get_volumes(dataset, volume)

        v = None
        for volume in volumes:
            v = daisy.open_ds(str(snapshot_file.absolute()), f"{volume}")
            if roi is not None:
                v = v.intersect(roi)
            if v.dtype == np.int64:
                v.materialize()
                v.data = v.data.astype(np.uint64)
            if v.dtype == np.dtype(bool):
                v.materialize()
                v.data = v.data.astype(np.float32)

            v.materialize()
            if modify is not None:
                data = modify(v.data, volume)
            else:
                data = v.data

            print(f"v.voxel_size: {v.voxel_size}, v.roi: {v.roi}")
            voxel_size = v.voxel_size[-dims:]
            array_offset = v.roi.get_offset()[-dims:]
            add_layer(
                context,
                data,
                f"{name_prefix}_{volume}",
                visible=False,
                voxel_size=voxel_size,
                array_offset=array_offset,
            )


def add_dacapo_snapshot(
    context,
    snapshot_file,
    name_prefix="snapshot",
    volume_paths=[None],
    graph_paths=[None],
    graph_node_attrs=None,
    graph_edge_attrs=None,
    # mst=["embedding", "fg_maxima"],
    mst=None,
    roi=None,
):
    raw = daisy.open_ds(str(snapshot_file.absolute()), f"volumes/raw")
    raw_shape = raw.shape[-3:]
    voxel_size = raw.voxel_size[-3:]
    array_offset = raw.roi.get_offset()[-3:]
    def modify(v, name):
        if name == "prediction":
            v = reshape_batch_channel(v, 1, 3, raw_shape)
        elif name == "raw":
            v = reshape_batch_channel(v, 0, 2, raw_shape)
        elif name == "target":
            v = reshape_batch_channel(v, 1, 3, raw_shape)
        elif name == "weights":
            v = reshape_batch_channel(v, 1, 3, raw_shape)
        elif name == "volumes/labels":
            v = reshape_batch_channel(v, 0, 2, raw_shape)
        elif name == "volumes/raw":
            v = reshape_batch_channel(v, 0, 2, raw_shape)
        elif name == "volumes/predictions":
            v = reshape_batch_channel(v, 0, 3, raw_shape)
        elif name == "volumes/prediction_gradients":
            v = reshape_batch_channel(v, 0, 3, raw_shape)
        else:
            print(f"Modifying unknown array {name} with shape {v.shape}")
            v = reshape_batch_channel(v, 1, 3, raw_shape)
        return v

    add_container(
        context,
        snapshot_file,
        name_prefix,
        volume_paths,
        graph_paths,
        graph_node_attrs,
        graph_edge_attrs,
        mst,
        roi,
        modify,
        voxel_size = voxel_size,
        array_offset = array_offset,
    )


def reshape_batch_channel(array, batch_dim=0, concat_dim=0, raw_shape=None):
    # Given shape (a0, a1, ..., am) and batch dim k:
    # First remove the dim ak: new_shape = (a0, a1, ..., ak-1, ak+1, ..., am)
    # Next replace concat_dim with -1
    def pad_volumes(volumes, shape):
        for v in volumes:
            v_shape = np.array(v.shape[-3:], dtype=np.uint32)
            diff = shape - v_shape
            pad = diff // 2
            assert all(np.isclose(pad * 2, diff))
            pad = ((0,0),)*(len(v.shape)-3) + tuple((p, p) for p in pad)
            yield np.pad(v, pad, mode="constant", constant_values=np.nan)

    if batch_dim is not None:
        array = np.concatenate(list(pad_volumes(np.rollaxis(array, batch_dim), raw_shape)), concat_dim)
    return array
