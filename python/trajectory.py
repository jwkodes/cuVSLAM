# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.
"""Export timestamp-associated camera poses without changing tracker state."""

import csv
from dataclasses import dataclass
import math
from numbers import Integral
import os
from pathlib import Path
import tempfile
from typing import Optional


@dataclass(frozen=True)
class TrajectoryImage:
    """An image associated with a rig-pose timestamp and a calibrated camera.

    IDs and image names are retained verbatim. ``timestamp_us`` can preserve an
    existing source timestamp; otherwise nanoseconds are rounded to microseconds
    using integer arithmetic (ties to even). An omitted synchronized sample ID
    becomes ``sample_<timestamp_ns>``; explicit IDs cannot be empty or "0",
    which pyCuSFM reserves for automatic synchronization. No interpolation occurs.
    """

    timestamp_ns: int
    camera_index: int
    image_name: str
    image_id: str
    camera_params_id: str
    synced_sample_id: Optional[str] = None
    timestamp_us: Optional[int] = None


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer')
    return int(value)


def _components(pose):
    q = tuple(float(v) for v in pose.rotation)
    t = tuple(float(v) for v in pose.translation)
    if len(q) != 4 or len(t) != 3 or not all(math.isfinite(v) for v in q + t):
        raise ValueError('Pose must have a finite xyzw quaternion and xyz translation')
    norm = math.sqrt(sum(v * v for v in q))
    if not math.isclose(norm, 1., rel_tol=0., abs_tol=1e-3):
        raise ValueError('Pose quaternion must be normalized')
    return tuple(v / norm for v in q), t


def _multiply(a, b):
    x, y, z, w = a
    u, v, s, t = b
    return (w * u + x * t + y * s - z * v, w * v - x * s + y * t + z * u, w * s + x * v - y * u + z * t,
            w * t - x * u - y * v - z * s)


def _camera_pose(world_from_rig, rig_from_camera):
    q, translation = _components(world_from_rig)
    r, offset = _components(rig_from_camera)
    rotated = _multiply(_multiply(q, (*offset, 0.)), (-q[0], -q[1], -q[2], q[3]))
    result = _multiply(q, r), tuple(translation[i] + rotated[i] for i in range(3))
    if not all(math.isfinite(v) for v in result[0] + result[1]):
        raise ValueError('Composed camera pose is not finite')
    return result


def _microseconds(timestamp_ns):
    whole, remainder = divmod(timestamp_ns, 1000)
    return whole + int(remainder > 500 or (remainder == 500 and whole % 2))


def export_trajectory_csv(output_path, poses, rig, images, *, overwrite=False):
    """Write camera-to-world poses in the CSV accepted by pyCuSFM's metadata tool.

    ``poses`` is an iterable of PoseStamped objects with world-from-rig poses,
    e.g. ``tracker.slam.get_all_slam_poses()`` for final optimized poses.
    ``images`` is an iterable of TrajectoryImage records. Every image must match
    a pose timestamp exactly; missing poses and duplicate IDs are errors. An
    image's camera pose is ``world_from_rig @ rig_from_camera``. Translation
    units are unchanged, so callers using cuSfM must supply meters.

    Calibration remains a separate input to cuSfM's metadata generator. This
    function does not select keyframes, align trajectories or invent poses.
    Returns the number of image rows written. Existing output is preserved
    unless overwrite=True; invalid input never replaces it.
    """
    if not isinstance(overwrite, bool):
        raise ValueError('overwrite must be boolean')
    by_time = {}
    for sample in poses:
        stamp = _integer(sample.timestamp_ns, 'pose timestamp_ns')
        if stamp in by_time:
            raise ValueError(f'Duplicate pose timestamp: {stamp}')
        _components(sample.pose)
        by_time[stamp] = sample.pose
    if not by_time:
        raise ValueError('No poses to export')
    rows, ids, observed = [], set(), set()
    for image in images:
        if not isinstance(image, TrajectoryImage):
            raise TypeError('images must contain TrajectoryImage records')
        stamp = _integer(image.timestamp_ns, 'image timestamp_ns')
        index = _integer(image.camera_index, 'camera_index')
        if index < 0 or index >= len(rig.cameras):
            raise ValueError(f'Unknown camera_index: {index}')
        if stamp not in by_time:
            raise ValueError(f'Image {image.image_id} has no pose at {stamp}')
        if image.image_id is None or image.camera_params_id is None:
            raise ValueError('Image ID and camera calibration ID are required')
        identifier = str(image.image_id)
        if not identifier or identifier in ids or (stamp, index) in observed:
            raise ValueError('Duplicate or empty image ID, or duplicate camera sample')
        if not image.image_name or not str(image.camera_params_id):
            raise ValueError('Image name and camera calibration ID are required')
        ids.add(identifier)
        observed.add((stamp, index))
        q, t = _camera_pose(by_time[stamp], rig.cameras[index].rig_from_camera)
        microseconds = _microseconds(stamp) if image.timestamp_us is None else _integer(
            image.timestamp_us, 'timestamp_us')
        sync_id = f'sample_{stamp}' if image.synced_sample_id is None else str(image.synced_sample_id)
        if sync_id in ('', '0'):
            raise ValueError('synced_sample_id cannot be empty or "0" (reserved by pyCuSFM)')
        rows.append((identifier, q[3], q[0], q[1], q[2], *t, str(image.camera_params_id), sync_id, microseconds,
                     str(image.image_name)))
    if not rows:
        raise ValueError('No images to export')
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w',
                                         newline='',
                                         encoding='utf-8',
                                         dir=path.parent,
                                         prefix=path.name + '.',
                                         delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(('image_id', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'camera_params_id',
                             'synced_sample_id', 'timestamp_us', 'image_name'))
            writer.writerows(rows)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        return len(rows)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
