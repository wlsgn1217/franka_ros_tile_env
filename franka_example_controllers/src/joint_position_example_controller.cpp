// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/joint_position_example_controller.h>

#include <cmath>
#include <memory>
#include <iostream>

#include <controller_interface/controller_base.h>
#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <franka/robot_state.h>
#include <franka_example_controllers/pseudo_inversion.h>

namespace franka_example_controllers {

bool JointPositionExampleController::init(hardware_interface::RobotHW* robot_hw,
                                          ros::NodeHandle& node_handle) {

  sub_equilibrium_pose_ = node_handle.subscribe(
      "equilibrium_pose", 20, &JointPositionExampleController::equilibriumPoseCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  
  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("JointPositionExampleController: Could not read parameter arm_id");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names)) {
    ROS_ERROR("JointPositionExampleController: Could not parse joint names");
  }
  if (joint_names.size() != 7) {
    ROS_ERROR_STREAM("JointPositionExampleController: Wrong number of joint names, got "
                     << joint_names.size() << " instead of 7 names!");
    return false;
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "JointPositionExampleController:ExampleController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "JointPositionExampleController:ExampleController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
    if (state_interface == nullptr) {
      ROS_ERROR_STREAM(
          "JointPositionExampleController:ExampleController: Error getting state interface from hardware");
      return false;
    }
    try {
      state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
          state_interface->getHandle(arm_id + "_robot"));
    } catch (hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "JointPositionExampleController:ExampleController: Exception getting state handle from interface: "
          << ex.what());
      return false;
    }

  position_joint_interface_ = robot_hw->get<hardware_interface::PositionJointInterface>();
  if (position_joint_interface_ == nullptr) {
    ROS_ERROR(
        "JointPositionExampleController: Error getting position joint interface from hardware!");
    return false;
  }
  position_joint_handles_.resize(7);
  for (size_t i = 0; i < 7; ++i) {
    try {
      position_joint_handles_[i] = position_joint_interface_->getHandle(joint_names[i]);
    } catch (const hardware_interface::HardwareInterfaceException& e) {
      ROS_ERROR_STREAM(
          "JointPositionExampleController: Exception getting joint handles: " << e.what());
      return false;
    }
  }

  dynamic_reconfigure_compliance_param_node_ =
    ros::NodeHandle(node_handle.getNamespace() + "/dynamic_reconfigure_compliance_param_node");

  dynamic_server_compliance_param_ = std::make_unique<
      dynamic_reconfigure::Server<franka_example_controllers::compliance_paramConfig>>(

      dynamic_reconfigure_compliance_param_node_);
  dynamic_server_compliance_param_->setCallback(
      boost::bind(&JointPositionExampleController::complianceParamCallback, this, _1, _2));

  position_d_target_.setZero();
  orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;

  cartesian_stiffness_.setZero();
  cartesian_damping_.setZero();

  return true;
}

void JointPositionExampleController::starting(const ros::Time& /* time */) {
  for (size_t i = 0; i < 7; ++i) {
    initial_pose_[i] = position_joint_handles_[i].getPosition();
    std::cout<<"intial_pose "<< i<< " : " << initial_pose_[i] << std::endl;
  }
  elapsed_time_ = ros::Duration(0.0);
  franka::RobotState initial_state = state_handle_->getRobotState();
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(initial_state.q.data());
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));
  
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());
  q_d_nullspace_ = q_initial;

  // set equilibrium point to current state
  position_d_target_ = initial_transform.translation();
  orientation_d_target_ = Eigen::Quaterniond(initial_transform.rotation());
}

void JointPositionExampleController::update(const ros::Time& /*time*/,
                                            const ros::Duration& period) {
  // elapsed_time_ += period;
  // double delta_angle = M_PI / 16 * (1 - std::cos(M_PI / 5.0 * elapsed_time_.toSec())) * 0.2;
  // for (size_t i = 0; i < 7; ++i) {
  //   if (i == 4) {
  //     position_joint_handles_[i].setCommand(initial_pose_[i]);
  //   } else {
  //     position_joint_handles_[i].setCommand(initial_pose_[i]);
  //   }
  // }

  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);                                       

  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_d(robot_state.q_d.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.rotation());
  Eigen::Map<Eigen::Matrix<double, 6, 1>> f_ext(robot_state.K_F_ext_hat_K.data());

  Eigen::Matrix<double, 6, 1> error;
  error.head(3) << position - position_d_target_;

  // orientation error
  if (orientation_d_target_.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  // "difference" quaternion
  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_target_);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  // Transform to base frame
  error.tail(3) << -transform.rotation() * error.tail(3);

  Eigen::MatrixXd jacobian_transpose_pinv;
  pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

  Eigen::VectorXd dq_d_task(7), dq_d_null(7), dq_d(7);
  dq_d_task << jacobian.transpose() *
                  (-cartesian_stiffness_ * error - cartesian_damping_ * (jacobian * dq));

  dq_d_null << (Eigen::MatrixXd::Identity(7, 7) -
                    jacobian.transpose() * jacobian_transpose_pinv) *
                       (nullspace_stiffness_ * (q_d_nullspace_ - q) -
                        (2.0 * sqrt(nullspace_stiffness_)) * dq);
  dq_d << dq_d_task + dq_d_null;
  
  Eigen::VectorXd target_joint_position(7);
  target_joint_position << q + 0.1*period.toSec() * dq_d;

  Eigen::VectorXd target_pose(6);
  target_pose << jacobian * target_joint_position;
  // for (size_t i = 0; i < 7; ++i) {
  //   pose_[i] = position_joint_handles_[i].getPosition();
  // }

  // ROS_INFO_STREAM("q is: " << q);
  // ROS_INFO_STREAM("target_joint_position is: " << target_joint_position);
  // ROS_INFO_STREAM("target_pose is: " << target_pose);

  
  for (size_t i = 0; i < 7; ++i) {
    position_joint_handles_[i].setCommand(target_joint_position[i]);
  }

}

// void JointPositionExampleController::starting(const ros::Time& /* time */) {
//   franka::RobotState initial_state = state_handle_->getRobotState();
//   std::array<double, 42> jacobian_array =
//       model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
//   Eigen::Map<Eigen::Matrix<double, 7, 1>> q(initial_state.q.data());
//   Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));
  
//   // set equilibrium point to current state
//   position_d_target_ = initial_transform.translation();
//   orientation_d_target_ = Eigen::Quaterniond(initial_transform.rotation());

// }

// void JointPositionExampleController::update(const ros::Time& /*time*/,
//                                             const ros::Duration& period) {
  
//   franka::RobotState robot_state = state_handle_->getRobotState();
//   std::array<double, 42> jacobian_array =
//       model_handle_->getZeroJacobian(franka::Frame::kEndEffector);                                       

//   Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
//   Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
//   Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
//   Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
//   Eigen::Vector3d position(transform.translation());
//   Eigen::Quaterniond orientation(transform.rotation());
//   Eigen::Map<Eigen::Matrix<double, 6, 1>> f_ext(robot_state.K_F_ext_hat_K.data());

//   Eigen::Matrix<double, 6, 1> error;
//   error.head(3) << position - position_d_target_;

//   // orientation error
//   if (orientation_d_target_.coeffs().dot(orientation.coeffs()) < 0.0) {
//     orientation.coeffs() << -orientation.coeffs();
//   }
//   // "difference" quaternion
//   Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_target_);
//   error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
//   // Transform to base frame
//   error.tail(3) << -transform.rotation() * error.tail(3);

//   Eigen::MatrixXd jacobian_transpose_pinv;
//   pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

//   Eigen::VectorXd joint_position_d;
//   joint_position_d << jacobian_transpose_pinv * error;

//   Eigen::VectorXd target_joint_position;
//   target_joint_position << q;

//   std::lock_guard<std::mutex> position_d_target_mutex_lock(
//       position_and_orientation_d_target_mutex_);
  
//   for (size_t i = 0; i < 7; ++i) {
//     position_joint_handles_[i].setCommand(target_joint_position(i));
//   }
// }

Eigen::Matrix<double, 7, 1> JointPositionExampleController::saturateVelocityRate(
    const Eigen::Matrix<double, 7, 1>& dq_d_calculated,
    const Eigen::Matrix<double, 7, 1>& dq_J_d) {  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> dq_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = dq_d_calculated[i] - dq_J_d[i];
    dq_d_saturated[i] =
        dq_J_d[i] + std::max(std::min(difference, delta_dq_max_), -delta_dq_max_);
  }
  return dq_d_saturated;
}

void JointPositionExampleController::complianceParamCallback(
    franka_example_controllers::compliance_paramConfig& config,
    uint32_t /*level*/) {
  cartesian_stiffness_target_.setIdentity();
  cartesian_stiffness_target_.topLeftCorner(3, 3)
      << translational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_stiffness_target_.bottomRightCorner(3, 3)
      << rotational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.setIdentity();
  // Damping ratio = 1
  cartesian_damping_target_.topLeftCorner(3, 3)
      << 2.0 * sqrt(translational_stiffness) * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.bottomRightCorner(3, 3)
      << 2.0 * sqrt(rotational_stiffness) * Eigen::Matrix3d::Identity();
}

void JointPositionExampleController::equilibriumPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& msg) {
  std::lock_guard<std::mutex> position_d_target_mutex_lock(
      position_and_orientation_d_target_mutex_);
  position_d_target_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
  orientation_d_target_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
      msg->pose.orientation.z, msg->pose.orientation.w;
  if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
    orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
  }
}

}  // namespace franka_example_controllers




PLUGINLIB_EXPORT_CLASS(franka_example_controllers::JointPositionExampleController,
                       controller_interface::ControllerBase)
