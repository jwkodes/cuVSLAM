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

"""Check native settings are collected when NumPy is also imported."""

import subprocess
import sys
import unittest


class TestReferenceLeaks(unittest.TestCase):
    def assert_clean_exit(self, code):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("nanobind: leaked", result.stderr)

    def test_numpy_before_sdk(self):
        self.assert_clean_exit("import numpy; import cuvslam")

    def test_numpy_after_sdk(self):
        self.assert_clean_exit("import cuvslam; import numpy")

    def test_configuration_defaults(self):
        self.assert_clean_exit("""
import numpy
import cuvslam
first = cuvslam.Odometry.Config()
second = cuvslam.Odometry.Config()
original = second.rgbd_settings.depth_scale_factor
first.rgbd_settings.depth_scale_factor = original + 100
first.multisensor_settings.depth_camera_ids = [0]
assert second.rgbd_settings.depth_scale_factor == original
assert list(second.multisensor_settings.depth_camera_ids) == []
""")
