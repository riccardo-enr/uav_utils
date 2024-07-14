import unittest
import numpy as np
from numpy.linalg import norm
from control.nmpc_utils import acc2quaternion, acceleration_sp_to_thrust_q

class GeometricControllerTest(unittest.TestCase):

    def test_acceleration_sp_to_thrust_q(self):
        # Condition
        acceleration_sp = np.array([0.0, 0.0, - 10.0])
        yaw = 0.0
        # Expected outcome
        ref_thrust = - 0.58

        ref_quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        thrust, quaternion, _ = acceleration_sp_to_thrust_q(None, acceleration_sp, yaw)

        print(f"Test 1 - Expected: {ref_thrust}, Got: {thrust}")
        self.assertTrue(np.allclose(thrust, ref_thrust), f"Expected {ref_thrust}, but got {thrust}")

    def test_acc2quaternion(self):
        # Condition
        acceleration = np.array([0.0, 0.0, 1.0])
        yaw = 0.0
        # Expected outcome
        ref_attitude = np.array([1.0, 0.0, 0.0, 0.0])

        attitude = acc2quaternion(acceleration, yaw)

        print(f"Test 1 - Expected: {ref_attitude}, Got: {attitude}")
        self.assertTrue(np.allclose(attitude, ref_attitude), f"Expected {ref_attitude}, but got {attitude}")

        # Condition
        acceleration = np.array([0.0, 0.0, 1.0])
        yaw = 1.5714
        # Expected outcome
        ref_attitude = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])

        attitude = acc2quaternion(acceleration, yaw)

        print(f"Test 2 - Expected: {ref_attitude}, Got: {attitude}")
        self.assertTrue(np.allclose(attitude, ref_attitude), f"Expected {ref_attitude}, but got {attitude}")

if __name__ == '__main__':
    unittest.main()
