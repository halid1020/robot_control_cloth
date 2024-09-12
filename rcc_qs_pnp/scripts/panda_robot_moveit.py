#!/usr/bin/env python
import sys
import rospy
import moveit_commander
from moveit_commander import PlanningSceneInterface, RobotCommander
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive
import time
from rcc_utils import prepare_posestamped, MyPos

class FrankaRobotMoveit:
    def __init__(self):
        # rospy.init_node('franka_robot_moveit', anonymous=True)
        
        # Initialize MoveIt commander
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("bob_arm")
        self.group.set_max_velocity_scaling_factor(0.3)
        self.group.set_max_acceleration_scaling_factor(0.1)
        self.group.set_planning_pipeline_id('pilz_industrial_motion_planner')
        self.group.set_planner_id('PTP')
        # self.arm_group.set_planning_pipeline_id('ompl')
        # self.arm_group.set_planner_id('RRTstar')
        self.group.set_pose_reference_frame("bob_link0")
        self.group.set_end_effector_link("bob_link8")
        self.planning_frame = self.group.get_planning_frame()

        # Logger
        self.logger = rospy.loginfo


        self.logger('Finished initializing Franka Robot MoveIt interface')

    def get_current_pose(self):
        """Get the current pose of the end-effector."""
        return self.group.get_current_pose().pose

    def go(self, pose=None, name=None, joint_states=None, straight = False):
        # Clear previously set targets
        self.group.clear_pose_targets()

        if name:
            self.logger(f'Setting named target: {name}')
            self.group.set_named_target(name)
            self.plan_and_execute()
        elif pose:
            self.logger(f'Setting pose target: {pose}')
            pose_stamped = prepare_posestamped(pose, frame_id=self.planning_frame)
            self.group.set_pose_target(pose_stamped)
            if straight:
                (plan, fraction) = self.group.compute_cartesian_path(
                                    [pose_stamped.pose],   # waypoints to follow
                                    0.01,        # eef_step
                                    0.0)
                success = self.group.execute(plan, wait=True)
        elif joint_states:
            self.logger(f'Setting joint state target: {joint_states}')
            self.group.set_joint_value_target(joint_states)
            self.plan_and_execute()
        self.group.stop()
        self.group.clear_pose_targets()




    def plan_and_execute(self):
        self.logger('Planning...')
        plan = self.group.plan()  # No unpacking, as plan() returns a single object
        # Check if the plan contains valid trajectory points
        if plan[0]:
            self.logger('Plan found. Executing...')
            success = self.group.execute(plan[1], wait=True)
            if success:
                self.logger('Execution finished successfully')
            else:
                self.logger('Execution failed')
        else:
            self.logger('Planning failed')

        rospy.sleep(1)
    

def custom_pose():
    return MyPos(
        pose=[-0.12387, 0.20431, 0.61006],
        orien=[0.6708, -0.22, -0.66698, -0.23497]
    )
    return pose

def ready_joint_states():
    return {
        'bob_joint1': 0.0,
        'bob_joint2': -0.785,
        'bob_joint3': 0.0,
        'bob_joint4': -2.356,
        'bob_joint5': 0.0,
        'bob_joint6': 1.571,
        'bob_joint7': 0.785,
    }
    return joints_states

def main():
    rospy.init_node('franka_robot_moveit')
    robot = FrankaRobotMoveit()
    robot.go(pose=custom_pose())
    robot.go(joint_states=ready_joint_states())

    rospy.spin()

if __name__ == '__main__':
    main()