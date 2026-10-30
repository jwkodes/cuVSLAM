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

import csv
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from cuvslam.trajectory import TrajectoryImage, export_trajectory_csv


def pose(rotation=(0, 0, 0, 1), translation=(0, 0, 0)):
    return SimpleNamespace(rotation=rotation, translation=translation)


class TestTrajectoryExport(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'poses.csv'
        self.stamp = 1700000000000000500
        self.samples = [SimpleNamespace(timestamp_ns=self.stamp, pose=pose())]
        self.rig = SimpleNamespace(cameras=[
            SimpleNamespace(rig_from_camera=pose()),
            SimpleNamespace(rig_from_camera=pose(translation=(.004, 0, 0)))
        ])
        self.frames = [
            TrajectoryImage(self.stamp, 0, 'left/a.png', '0', 'L', 'pair0'),
            TrajectoryImage(self.stamp, 1, 'right/a.png', '1', 'R', 'pair0')
        ]

    def read(self):
        with self.path.open(newline='') as stream:
            return list(csv.DictReader(stream))

    def test_stereo_export_uses_camera_to_world_not_inverse(self):
        h = math.sqrt(.5)
        self.samples[0].pose = pose((0, 0, h, h), (1, 2, 3))
        self.assertEqual(export_trajectory_csv(self.path, self.samples, self.rig, self.frames), 2)
        rows = self.read()
        self.assertAlmostEqual(float(rows[0]['tx']), 1)
        self.assertAlmostEqual(float(rows[1]['tx']), 1)
        self.assertAlmostEqual(float(rows[1]['ty']), 2.004)
        self.assertAlmostEqual(float(rows[1]['tz']), 3)
        self.assertAlmostEqual(float(rows[1]['qw']), h)
        self.assertAlmostEqual(float(rows[1]['qz']), h)
        self.assertEqual(rows[1]['image_name'], 'right/a.png')
        self.assertEqual(rows[1]['camera_params_id'], 'R')
        self.assertEqual(rows[1]['synced_sample_id'], 'pair0')
        self.assertEqual(rows[0]['timestamp_us'], '1700000000000000')

    def test_explicit_source_microseconds_is_retained(self):
        frames = [replace(self.frames[0], timestamp_us=123456)]
        export_trajectory_csv(self.path, self.samples, self.rig, frames)
        self.assertEqual(self.read()[0]['timestamp_us'], '123456')

    def test_zero_timestamp_uses_nonreserved_sync_id(self):
        sample = SimpleNamespace(timestamp_ns=0, pose=pose())
        frame = replace(self.frames[0], timestamp_ns=0, synced_sample_id=None)
        export_trajectory_csv(self.path, [sample], self.rig, [frame])
        self.assertEqual(self.read()[0]['synced_sample_id'], 'sample_0')
        self.assertEqual(self.read()[0]['timestamp_us'], '0')
        for invalid in ('', '0', 0):
            with self.assertRaisesRegex(ValueError, 'reserved'):
                export_trajectory_csv(self.path, [sample],
                                      self.rig, [replace(frame, synced_sample_id=invalid)],
                                      overwrite=True)
        self.assertEqual(self.read()[0]['synced_sample_id'], 'sample_0')

    def test_missing_pose_does_not_create_output(self):
        frames = [replace(self.frames[0], timestamp_ns=self.stamp + 1)]
        with self.assertRaisesRegex(ValueError, 'no pose'):
            export_trajectory_csv(self.path, self.samples, self.rig, frames)
        self.assertFalse(self.path.exists())

    def test_duplicate_pose_or_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate pose'):
            export_trajectory_csv(self.path, self.samples * 2, self.rig, self.frames)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            export_trajectory_csv(self.path, self.samples, self.rig, self.frames * 2)

    def test_invalid_camera_and_pose_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'camera_index'):
            export_trajectory_csv(self.path, self.samples, self.rig, [replace(self.frames[0], camera_index=2)])
        for bad in [pose(translation=(float('nan'), 0, 0)), pose(rotation=(0, 0, 0, 0))]:
            self.samples[0].pose = bad
            with self.assertRaises(ValueError):
                export_trajectory_csv(self.path, self.samples, self.rig, self.frames)

    def test_existing_file_requires_explicit_overwrite(self):
        self.path.write_text('preserve me')
        with self.assertRaises(FileExistsError):
            export_trajectory_csv(self.path, self.samples, self.rig, self.frames)
        self.assertEqual(self.path.read_text(), 'preserve me')
        export_trajectory_csv(self.path, self.samples, self.rig, self.frames, overwrite=True)
        self.assertEqual(len(self.read()), 2)
        self.samples[0].pose = pose(rotation=(0, 0, 0, 0))
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            export_trajectory_csv(self.path, self.samples, self.rig, self.frames, overwrite=True)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob('poses.csv.*')), [])
