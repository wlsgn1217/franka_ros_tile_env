// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#pragma once

#include <array>
#include <string>
#include <vector>
#include <mutex>

#include <controller_interface/multi_interface_controller.h>
#include <dynamic_reconfigure/server.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>

#include <Eigen/Dense>
#include <geometry_msgs/PoseStamped.h>

#include <franka_example_controllers/compliance_paramConfig.h>
#include <franka_hw/franka_state_interface.h>
#include <franka_hw/franka_model_interface.h>

namespace franka_example_controllers {

class JointPositionExampleController : public controller_interface::MultiInterfaceController<
                                           franka_hw::FrankaModelInterface,
                                           hardware_interface::PositionJointInterface,
                                           franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:  
  Eigen::Matrix<double, 7, 1> saturateVelocityRate(
    const Eigen::Matrix<double, 7, 1>& dq_d_calculated,
    const Eigen::Matrix<double, 7, 1>& dq_J_d);

  hardware_interface::PositionJointInterface* position_joint_interface_;
  std::vector<hardware_interface::JointHandle> position_joint_handles_;
  ros::Duration elapsed_time_;
  std::array<double, 7> initial_pose_{};
  std::array<double, 7> pose_{};

  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::mutex position_and_orientation_d_target_mutex_;

  double filter_params_{0.005};
  double translational_stiffness{200.0};
  double rotational_stiffness{10.0};
  double nullspace_stiffness_{20.0};
  double nullspace_stiffness_target_{0.5};
  //joint vel limit: 150 degree /s = 2.617 rad /s -> 1.0 rad/s
  const double delta_dq_max_{1.0};

  Eigen::Matrix<double, 6, 6> cartesian_stiffness_;
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_target_;
  Eigen::Vector3d position_d_target_;
  Eigen::Quaterniond orientation_d_target_;

  Eigen::Matrix<double, 7, 1> q_d_nullspace_;

  ros::Subscriber sub_equilibrium_pose_;
  void equilibriumPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  std::unique_ptr<dynamic_reconfigure::Server<franka_example_controllers::compliance_paramConfig>>
      dynamic_server_compliance_param_;
  ros::NodeHandle dynamic_reconfigure_compliance_param_node_;
  void complianceParamCallback(franka_example_controllers::compliance_paramConfig& config,
                               uint32_t level);

};

}  // namespace franka_example_controllers
