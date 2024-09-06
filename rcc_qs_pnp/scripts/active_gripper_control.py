#!/usr/bin/env python

import rospy
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal
import actionlib

class ActiveGripperControl:
    def __init__(self):
        """
        Initialize the gripper control for Franka Emika Panda.
        """
        rospy.init_node('active_gripper_control', anonymous=True)

        # Action clients for moving and grasping with the gripper
        self.grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        self.move_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)

        # Wait for the action servers to start
        rospy.loginfo("Waiting for the gripper action servers to start...")
        self.grasp_client.wait_for_server()
        self.move_client.wait_for_server()
        rospy.loginfo("Gripper action servers are up.")

    def open(self, width=0.08, speed=0.1):
        """
        Open the gripper to the specified width.
        :param width: The desired opening width (in meters).
        :param speed: The speed at which to open the gripper (in m/s).
        """
        move_goal = MoveGoal(width=width, speed=speed)
        rospy.loginfo(f"Sending gripper open command: width={width}, speed={speed}")
        self.move_client.send_goal(move_goal)
        self.move_client.wait_for_result()
        rospy.loginfo("Gripper opened.")

    def grasp(self, width=0.02, force=40.0, epsilon_inner=0.005, epsilon_outer=0.005, speed=0.1):
        """
        Close the gripper and grasp an object.
        :param width: The desired closing width (in meters).
        :param force: The force to apply when closing the gripper (in N).
        :param epsilon_inner: Inner tolerance for grasp success (in meters).
        :param epsilon_outer: Outer tolerance for grasp success (in meters).
        :param speed: The speed at which to close the gripper (in m/s).
        """
        grasp_goal = GraspGoal()
        grasp_goal.width = width
        grasp_goal.speed = speed
        grasp_goal.force = force
        grasp_goal.epsilon.inner = epsilon_inner
        grasp_goal.epsilon.outer = epsilon_outer

        rospy.loginfo(f"Sending gripper grasp command: width={width}, force={force}, speed={speed}")
        self.grasp_client.send_goal(grasp_goal)
        self.grasp_client.wait_for_result()
        rospy.loginfo("Gripper grasped.")

def main():
    gripper_control = ActiveGripperControl()

    # Example usage
    rospy.sleep(1)
    gripper_control.open()
    rospy.sleep(1)
    gripper_control.grasp()

if __name__ == "__main__":
    main()
