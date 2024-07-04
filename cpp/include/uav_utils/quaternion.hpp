#ifndef QUATERNION_HPP
#define QUATERNION_HPP

#include <Eigen/Dense>

namespace uav_utils {

std::array<double, 3> quaternionToEuler(const Eigen::Quaterniond &q);

} // namespace uav_utils

#endif // QUATERNION_HPP