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

"""Configuration validation and conversion against the real SDK bindings; no tracking."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
import unittest

import cuvslam as vslam
from cuvslam import CuVSLAMConfig


class TestCuVSLAMConfig(unittest.TestCase):
    def setUp(self):
        self.left = vslam.Camera(size=(640, 480), focal=(500., 500.), principal=(320., 240.))
        self.right = vslam.Camera(size=(640, 480), focal=(500., 500.), principal=(320., 240.))
        self.right.rig_from_camera = vslam.Pose((0., 0., 0., 1.), (.005, 0., 0.))
        self.cameras = [self.left, self.right]

    def test_defaults_preserve_native_values(self):
        settings = CuVSLAMConfig().to_native(self.cameras)
        default = vslam.Odometry.Config()
        self.assertEqual(settings.tracker_mode, vslam.Tracker.Mode.OdometryWithSlamRealtime)
        self.assertEqual(settings.odometry.async_sba, default.async_sba)
        self.assertEqual(settings.odometry.min_depth, default.min_depth)
        self.assertEqual(settings.odometry.rectified_stereo_camera, default.rectified_stereo_camera)
        self.assertEqual(settings.odometry.use_motion_model, default.use_motion_model)
        self.assertEqual(settings.slam.sync_mode, vslam.Slam.Config().sync_mode)
        self.assertAlmostEqual(settings.rig.min_landmark_depth, vslam.Rig().min_landmark_depth)

    def test_offline_near_depth_and_rectification(self):
        config = CuVSLAMConfig(execution_mode='offline', rig={'min_landmark_depth': .01},
                              odometry={'min_depth': .01, 'rectified_stereo_camera': True})
        settings = config.to_native(self.cameras)
        self.assertEqual(settings.tracker_mode, vslam.Tracker.Mode.OdometryWithSlamOffline)
        self.assertFalse(settings.odometry.async_sba)
        self.assertTrue(settings.slam.sync_mode)
        self.assertTrue(settings.odometry.rectified_stereo_camera)
        self.assertAlmostEqual(settings.rig.min_landmark_depth, .01)
        self.assertAlmostEqual(settings.odometry.min_depth, .01)

    def test_rgbd_scale_camera_and_mode(self):
        config = CuVSLAMConfig(input_mode='rgbd', execution_mode='offline',
                              rig={'min_landmark_depth': .01}, odometry={'rgbd_settings': {
                                  'depth_camera_id': 0, 'depth_scale_factor': 10000.,
                                  'enable_depth_stereo_tracking': False}})
        settings = config.to_native([self.left])
        self.assertEqual(settings.odometry.odometry_mode, vslam.Odometry.OdometryMode.RGBD)
        self.assertFalse(settings.odometry.rectified_stereo_camera)
        self.assertEqual(settings.odometry.rgbd_settings.depth_camera_id, 0)
        self.assertEqual(settings.odometry.rgbd_settings.depth_scale_factor, 10000.)
        self.assertFalse(settings.odometry.rgbd_settings.enable_depth_stereo_tracking)
        self.assertEqual(len(settings.rig.cameras), 1)

    def test_rgbd_defaults_to_first_camera(self):
        settings = CuVSLAMConfig(input_mode='rgbd').to_native([self.left])
        self.assertEqual(settings.odometry.rgbd_settings.depth_camera_id, 0)
        self.assertEqual(settings.odometry.rgbd_settings.depth_scale_factor,
                         vslam.Odometry.RGBDSettings().depth_scale_factor)

    def test_odometry_only_modes(self):
        for execution, expected in [('offline', vslam.Tracker.Mode.OdometryOnlyOffline),
                                    ('realtime', vslam.Tracker.Mode.OdometryOnlyRealtime)]:
            with self.subTest(execution=execution):
                settings = CuVSLAMConfig(execution_mode=execution, enable_slam=False).to_native(self.cameras)
                self.assertIsNone(settings.slam)
                self.assertEqual(settings.tracker_mode, expected)

    def test_explicit_matching_thread_settings(self):
        settings = CuVSLAMConfig(execution_mode='offline', odometry={'async_sba': False},
                                slam={'sync_mode': True}).to_native(self.cameras)
        self.assertFalse(settings.odometry.async_sba)
        self.assertTrue(settings.slam.sync_mode)

    def test_contradictory_thread_settings(self):
        for execution, group, values in [('offline', 'odometry', {'async_sba': True}),
                                         ('offline', 'slam', {'sync_mode': False}),
                                         ('realtime', 'odometry', {'async_sba': False}),
                                         ('realtime', 'slam', {'sync_mode': True})]:
            with self.subTest(execution=execution, group=group):
                with self.assertRaisesRegex(ValueError, 'requires'):
                    CuVSLAMConfig(execution_mode=execution, **{group: values})

    def test_modes_are_not_duplicated_in_native_options(self):
        with self.assertRaisesRegex(ValueError, 'Use input_mode'):
            CuVSLAMConfig(odometry={'odometry_mode': 'RGBD'})
        with self.assertRaises(ValueError):
            CuVSLAMConfig(input_mode='mono')
        with self.assertRaises(ValueError):
            CuVSLAMConfig(execution_mode='live_camera')

    def test_unused_mode_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'rgbd_settings require'):
            CuVSLAMConfig(odometry={'rgbd_settings': {}})
        with self.assertRaisesRegex(ValueError, 'multisensor'):
            CuVSLAMConfig(odometry={'multisensor_settings': {}})
        with self.assertRaisesRegex(ValueError, 'enable_slam'):
            CuVSLAMConfig(enable_slam=False, slam={'use_gpu': True})

    def test_calibration_is_not_embedded_in_config(self):
        with self.assertRaisesRegex(ValueError, 'calibrated cameras'):
            CuVSLAMConfig(rig={'cameras': []})

    def test_unknown_native_options_fail_before_tracker_creation(self):
        with self.assertRaisesRegex(ValueError, 'Invalid native configuration'):
            CuVSLAMConfig(odometry={'min_depht': .01}).to_native(self.cameras)
        with self.assertRaisesRegex(ValueError, 'Invalid native configuration'):
            CuVSLAMConfig(slam={'syn_mode': True}).to_native(self.cameras)
        with self.assertRaisesRegex(ValueError, 'Invalid rgbd_settings'):
            CuVSLAMConfig(input_mode='rgbd', odometry={'rgbd_settings': {
                'depth_scale_factr': 10000.}}).to_native([self.left])

    def test_unknown_top_level_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'execution_mod'):
            CuVSLAMConfig.from_dict({'execution_mod': 'offline'})
        for value in [None, [], 'stereo', {1: True}]:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    CuVSLAMConfig.from_dict(value)

    def test_json_and_yaml_roundtrip(self):
        config = CuVSLAMConfig(input_mode='rgbd', execution_mode='offline',
                              rig={'min_landmark_depth': .01}, odometry={
                                  'rgbd_settings': {'depth_scale_factor': 10000.}})
        with tempfile.TemporaryDirectory() as folder:
            for extension in ['json', 'yaml', 'yml']:
                path = Path(folder) / ('pipeline.' + extension)
                config.to_file(path)
                loaded = CuVSLAMConfig.from_file(path)
                self.assertEqual(loaded.to_dict(), config.to_dict())
                settings = loaded.to_native([self.left])
                self.assertFalse(settings.odometry.async_sba)
                self.assertEqual(settings.odometry.rgbd_settings.depth_scale_factor, 10000.)
                self.assertAlmostEqual(settings.rig.min_landmark_depth, .01)

    def test_existing_config_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'pipeline.json'
            path.write_text('{"input_mode":"rgbd"}', encoding='utf-8')
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                CuVSLAMConfig().to_file(path)
            self.assertEqual(path.read_bytes(), original)
            CuVSLAMConfig().to_file(path, overwrite=True)
            self.assertEqual(json.loads(path.read_text())['input_mode'], 'stereo')

    def test_unsupported_file_format(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'pipeline.txt'
            path.write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                CuVSLAMConfig.from_file(path)
            with self.assertRaises(ValueError):
                CuVSLAMConfig().to_file(path)

    def test_empty_yaml_is_not_a_config(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'pipeline.yaml'
            path.write_text('', encoding='utf-8')
            with self.assertRaises(TypeError):
                CuVSLAMConfig.from_file(path)

    def test_boolean_types_are_not_coerced(self):
        for kwargs in [{'enable_slam': 'false'}, {'enable_slam': 1},
                       {'odometry': {'async_sba': 'false'}}, {'slam': {'sync_mode': 1}}]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(TypeError):
                    CuVSLAMConfig(**kwargs)

    def test_depth_limits_and_units(self):
        for number in [0., -0.01, float('nan'), float('inf'), True]:
            with self.subTest(number=number):
                with self.assertRaises((ValueError, TypeError)):
                    CuVSLAMConfig(rig={'min_landmark_depth': number})
                with self.assertRaises((ValueError, TypeError)):
                    CuVSLAMConfig(input_mode='rgbd', odometry={'rgbd_settings': {
                        'depth_scale_factor': number}})
        self.assertEqual(CuVSLAMConfig(odometry={'min_depth': -1.}).to_native(self.cameras).odometry.min_depth, -1.)

    def test_rgbd_camera_ids(self):
        for camera_id in [-1, True, 0.5, '0']:
            with self.subTest(camera_id=camera_id):
                with self.assertRaises(ValueError):
                    CuVSLAMConfig(input_mode='rgbd', odometry={'rgbd_settings': {'depth_camera_id': camera_id}})
        with self.assertRaisesRegex(ValueError, 'outside'):
            CuVSLAMConfig(input_mode='rgbd', odometry={'rgbd_settings': {
                'depth_camera_id': 1}}).to_native([self.left])

    def test_camera_counts(self):
        with self.assertRaises(ValueError):
            CuVSLAMConfig().to_native([self.left])
        with self.assertRaises(ValueError):
            CuVSLAMConfig(input_mode='rgbd').to_native([])
        with self.assertRaises(ValueError):
            CuVSLAMConfig().create_tracker([])
        with self.assertRaisesRegex(ValueError, 'stereo pair'):
            CuVSLAMConfig(input_mode='rgbd', odometry={
                'rectified_stereo_camera': True}).to_native([self.left])

    def test_enum_strings(self):
        settings = CuVSLAMConfig(odometry={'multicam_mode': 'precision'},
                                slam={'loop_closure_mode': 'lsi'}).to_native(self.cameras)
        self.assertEqual(settings.odometry.multicam_mode, vslam.Odometry.MulticameraMode.Precision)
        self.assertEqual(settings.slam.loop_closure_mode, vslam.Slam.LoopClosureMode.LSI)
        with self.assertRaisesRegex(ValueError, 'multicam_mode'):
            CuVSLAMConfig(odometry={'multicam_mode': 'unknown'}).to_native(self.cameras)

    def test_null_inherits_defaults(self):
        settings = CuVSLAMConfig(execution_mode='offline', rig={'min_landmark_depth': None},
                                odometry={'min_depth': None, 'async_sba': None}).to_native(self.cameras)
        self.assertFalse(settings.odometry.async_sba)
        self.assertEqual(settings.odometry.min_depth, vslam.Odometry.Config().min_depth)
        self.assertAlmostEqual(settings.rig.min_landmark_depth, .1)

    def test_detached_immutable_settings(self):
        values = {'rgbd_settings': {'depth_scale_factor': 10000.}}
        config = CuVSLAMConfig(input_mode='rgbd', odometry=values)
        values['rgbd_settings']['depth_scale_factor'] = 1000.
        self.assertEqual(config.odometry['rgbd_settings']['depth_scale_factor'], 10000.)
        with self.assertRaises(TypeError):
            config.odometry['rgbd_settings']['depth_scale_factor'] = 1.
        with self.assertRaises(FrozenInstanceError):
            config.input_mode = 'stereo'
        detached = config.to_dict()
        detached['odometry']['rgbd_settings']['depth_scale_factor'] = 1.
        self.assertEqual(config.odometry['rgbd_settings']['depth_scale_factor'], 10000.)

    def test_each_resolution_is_fresh(self):
        config = CuVSLAMConfig(execution_mode='offline')
        first = config.to_native(self.cameras)
        first.odometry.min_depth = .25
        second = config.to_native(self.cameras)
        self.assertEqual(second.odometry.min_depth, vslam.Odometry.Config().min_depth)
        self.assertEqual(len(self.cameras), 2)
        self.assertEqual(tuple(self.left.size), (640, 480))


if __name__ == '__main__':
    unittest.main()
