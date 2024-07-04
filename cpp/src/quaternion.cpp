#include "uav_utils/quaternion.hpp"

namespace uav_utils {

/**
 * Converts a quaternion to Euler angles.
 *
 * @param q The quaternion to convert.
 * @return An array of Euler angles [roll, pitch, yaw].
 */
std::array<double, 3> quaternionToEuler(const Eigen::Quaterniond &q) {
  std::array<double, 3> euler;
  Eigen::Vector3d euler_angles = q.toRotationMatrix().eulerAngles(0, 1, 2);

  euler[0] = euler_angles[0]; // Roll
  euler[1] = euler_angles[1]; // Pitch
  euler[2] = euler_angles[2]; // Yaw

  return euler;
}

} // namespace uav_utils