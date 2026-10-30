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

"""Declarative startup configuration for stereo and RGB-D tracking.

Calibration and frame I/O stay outside this helper. Native configuration values
not supplied here retain their SDK defaults. Native option names are validated
by their constructors before a Tracker is created.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


def _freeze(value):
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError('Configuration keys must be strings')
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Configuration numbers must be finite')
        return value
    raise TypeError('Configuration values must be JSON-compatible; calibration is a separate input')


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _number(value, name, *, allow_auto=False):
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{name} must be a number')
    if not math.isfinite(value) or (value <= 0 and not (allow_auto and value < 0)):
        raise ValueError(f'{name} must be finite and positive' + (' or negative for automatic selection' if allow_auto else ''))


def _enum(enum_type, value, name):
    if not isinstance(value, str):
        return value
    members = getattr(enum_type, '__members__', {})
    if not members:
        members = {key: getattr(enum_type, key) for key in dir(enum_type)
                   if not key.startswith('_') and isinstance(getattr(enum_type, key), enum_type)}
    for key, member in members.items():
        if key.lower() == value.lower():
            return member
    raise ValueError(f'{name} must be one of: {", ".join(members)}')


@dataclass(frozen=True)
class NativeCuVSLAMConfig:
    """Resolved SDK constructor arguments; constructing these does not create a tracker."""

    rig: Any
    odometry: Any
    slam: Any
    tracker_mode: Any


@dataclass(frozen=True)
class CuVSLAMConfig:
    """Serializable constructor settings, independent of camera calibration.

    ``input_mode`` is ``stereo`` or ``rgbd``. ``execution_mode`` is ``realtime``
    (background optimization) or ``offline`` (blocking optimization). These pick
    the native odometry/Tracker modes and the two threading flags together.
    Explicit conflicting modes or threading flags are errors, never overrides.

    ``rig``, ``odometry`` and ``slam`` use the native constructors' option names.
    Omitted/null options inherit SDK defaults, except for the fields derived
    from the mode. In RGB-D, the default depth camera is camera zero; callers
    must set a depth scale matching their depth encoding. Mode selection does
    not generate depth, rectify images, read calibration, or run an input loop.
    """

    input_mode: str = 'stereo'
    execution_mode: str = 'realtime'
    enable_slam: bool = True
    rig: Mapping = field(default_factory=dict)
    odometry: Mapping = field(default_factory=dict)
    slam: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if self.input_mode not in ('stereo', 'rgbd'):
            raise ValueError('input_mode must be stereo or rgbd')
        if self.execution_mode not in ('realtime', 'offline'):
            raise ValueError('execution_mode must be realtime or offline')
        if type(self.enable_slam) is not bool:
            raise TypeError('enable_slam must be a bool')
        for name in ('rig', 'odometry', 'slam'):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f'{name} must be an object')
            object.__setattr__(self, name, _freeze(value))
        if set(self.rig) - {'min_landmark_depth'}:
            raise ValueError('rig only accepts min_landmark_depth; pass calibrated cameras to create_tracker')
        if 'odometry_mode' in self.odometry:
            raise ValueError('Use input_mode instead of odometry.odometry_mode')
        if 'multisensor_settings' in self.odometry:
            raise ValueError('This helper supports stereo and rgbd, not multisensor tracking')
        if not self.enable_slam and self.slam:
            raise ValueError('slam settings require enable_slam=True')
        asynchronous = self.execution_mode == 'realtime'
        for group, key, expected in ((self.odometry, 'async_sba', asynchronous),
                                     (self.slam, 'sync_mode', not asynchronous)):
            value = group.get(key)
            if value is not None and type(value) is not bool:
                raise TypeError(f'{key} must be a bool')
            if value is not None and value != expected:
                raise ValueError(f'{self.execution_mode} requires {key}={expected}')
        _number(self.rig.get('min_landmark_depth'), 'rig.min_landmark_depth')
        _number(self.odometry.get('min_depth'), 'odometry.min_depth', allow_auto=True)
        _number(self.odometry.get('max_depth'), 'odometry.max_depth', allow_auto=True)
        if self.input_mode != 'rgbd' and self.odometry.get('rgbd_settings') is not None:
            raise ValueError('rgbd_settings require input_mode=rgbd')
        depth = self.odometry.get('rgbd_settings')
        if depth is not None:
            if not isinstance(depth, Mapping):
                raise TypeError('rgbd_settings must be an object')
            _number(depth.get('depth_scale_factor'), 'rgbd_settings.depth_scale_factor')
            camera_id = depth.get('depth_camera_id')
            if camera_id is not None and (type(camera_id) is not int or camera_id < 0):
                raise ValueError('depth_camera_id must be a nonnegative integer')

    @classmethod
    def from_dict(cls, value):
        """Load structured settings; unknown top-level fields are errors."""
        if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
            raise TypeError('Configuration must be an object with string keys')
        unknown = set(value) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown configuration fields: {", ".join(sorted(unknown))}')
        return cls(**value)

    @classmethod
    def from_file(cls, path):
        """Load a YAML or JSON file. No tracker or GPU work is started."""
        path = Path(path)
        text = path.read_text(encoding='utf-8')
        if path.suffix.lower() == '.json':
            value = json.loads(text)
        elif path.suffix.lower() in ('.yaml', '.yml'):
            value = yaml.safe_load(text)
        else:
            raise ValueError('Configuration file must be .yaml, .yml or .json')
        return cls.from_dict(value)

    def to_dict(self):
        """Return a detached JSON-compatible copy of the requested settings."""
        return {item.name: _thaw(getattr(self, item.name)) for item in fields(self)}

    def to_file(self, path, *, overwrite=False):
        """Save requested settings, preserving an existing file unless explicitly replaced.

        This is not a snapshot of every native default. Pin the SDK version when
        reproducing runs, and inspect to_native() for resolved SDK objects.
        """
        path = Path(path)
        if path.suffix.lower() == '.json':
            text = json.dumps(self.to_dict(), indent=2, allow_nan=False) + '\n'
        elif path.suffix.lower() in ('.yaml', '.yml'):
            text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        else:
            raise ValueError('Configuration file must be .yaml, .yml or .json')
        with path.open('w' if overwrite else 'x', encoding='utf-8') as stream:
            stream.write(text)

    def to_native(self, cameras):
        """Build fresh Rig/Config objects and a Tracker mode, without creating a Tracker.

        Cameras are calibrated SDK Camera objects. Native option names/types are
        checked here by the SDK constructors. Input settings and cameras are not
        modified. Use the direct SDK APIs for other sensor modes.
        """
        from . import pycuvslam as sdk

        cameras = list(cameras)
        if not cameras or (self.input_mode == 'stereo' and len(cameras) < 2):
            raise ValueError('stereo needs at least two cameras; rgbd needs at least one')
        odometry = {key: _thaw(value) for key, value in self.odometry.items() if value is not None}
        slam = {key: _thaw(value) for key, value in self.slam.items() if value is not None}
        rig = {key: _thaw(value) for key, value in self.rig.items() if value is not None}
        asynchronous = self.execution_mode == 'realtime'
        odometry['async_sba'] = asynchronous
        odometry['odometry_mode'] = (sdk.Odometry.OdometryMode.Multicamera if self.input_mode == 'stereo'
                                     else sdk.Odometry.OdometryMode.RGBD)
        if 'multicam_mode' in odometry:
            odometry['multicam_mode'] = _enum(sdk.Odometry.MulticameraMode, odometry['multicam_mode'], 'multicam_mode')
        if self.input_mode == 'rgbd':
            depth = odometry.pop('rgbd_settings', {})
            depth = {key: value for key, value in depth.items() if value is not None}
            depth.setdefault('depth_camera_id', 0)
            if depth['depth_camera_id'] >= len(cameras):
                raise ValueError('depth_camera_id is outside the supplied camera list')
            if odometry.get('rectified_stereo_camera') and len(cameras) < 2:
                raise ValueError('rectified_stereo_camera requires a stereo pair, not a single RGB-D camera')
            try:
                odometry['rgbd_settings'] = sdk.Odometry.RGBDSettings(**depth)
            except TypeError as error:
                raise ValueError(f'Invalid rgbd_settings: {error}') from error
        slam['sync_mode'] = not asynchronous
        try:
            native_rig = sdk.Rig(cameras=cameras, **rig)
            native_odometry = sdk.Odometry.Config(**odometry)
            native_slam = sdk.Slam.Config(**slam) if self.enable_slam else None
        except TypeError as error:
            raise ValueError(f'Invalid native configuration options: {error}') from error
        prefix = 'OdometryWithSlam' if self.enable_slam else 'OdometryOnly'
        suffix = 'Realtime' if asynchronous else 'Offline'
        return NativeCuVSLAMConfig(native_rig, native_odometry, native_slam,
                                  getattr(sdk.Tracker.Mode, prefix + suffix))

    def create_tracker(self, cameras):
        """Create a fresh Tracker from these settings and calibrated cameras."""
        from . import pycuvslam as sdk

        settings = self.to_native(cameras)
        return sdk.Tracker(settings.rig, settings.tracker_mode, settings.odometry, settings.slam)
