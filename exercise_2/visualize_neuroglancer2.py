import neuroglancer
import sys
from pathlib import Path

from neuroglancer2 import add_container, add_dacapo_snapshot


neuroglancer.set_server_bind_address("0.0.0.0")



if __name__ == "__main__":
    args = sys.argv[1:]
    snapshot_file = args[0]
    
    voxel_size = [4, 4, 4]

    dimensions = neuroglancer.CoordinateSpace(
        names=["z", "y", "x"], units="nm", scales=voxel_size
    )

    snapshot_files = snapshot_file.split(",")

    viewer = neuroglancer.Viewer()
    viewer.dimensions = dimensions


    with viewer.txn() as s:
        for snapshot_file in snapshot_files:
            path = snapshot_file.split("/")
            if len(path) > 2 and path[-2] == "snapshots":
                add_dacapo_snapshot(s, Path(snapshot_file))
            else:
                add_container(s, Path(snapshot_file))

    print(viewer)
    input("Hit ENTER to quit!")
