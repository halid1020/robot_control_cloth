#!/usr/bin/env python

import rospy
import time
import actionlib
import franka_gripper.msg

class PandaGripperControl:
    def __init__(self):
        rospy.init_node('client_test', anonymous=True)
        self.gripper_client = actionlib.SimpleActionClient('/franka_gripper/move', franka_gripper.msg.MoveAction)
        self.gripper_client.wait_for_server()

        self.grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', franka_gripper.msg.GraspAction)
        self.grasp_client.wait_for_server()
        self.open_width = 0.02
        self.close_width = 0.0001


    def open(self):
        # Creates a goal to send to the action server.
        goal = franka_gripper.msg.MoveGoal()
        goal.width = self.open_width
        goal.speed = 0.1
        self.gripper_client.send_goal(goal)
        self.gripper_client.wait_for_result()
        result = self.gripper_client.get_result()
        print("Gripper result received: success=%s", result.success)
        return result

       

    def grasp(self):
        print('Grasp !!')
         
        # Creates a goal to send to the action server.
        goal = franka_gripper.msg.GraspGoal()
        goal.width = self.close_width
        goal.epsilon = franka_gripper.msg.GraspEpsilon(inner=0.001, outer=0.001)  # Correctly initialize epsilon
        goal.speed = 0.1
        goal.force = 20


        # rospy.loginfo("Sending goal: width=%s, epsilon.inner=%s, epsilon.outer=%s, speed=%s, force=%s",
        #             goal.width, goal.epsilon.inner, goal.epsilon.outer, goal.speed, goal.force)
        
        
        self.grasp_client.wait_for_server()

        # Sends the goal to the action server.
        self.grasp_client.send_goal(goal)

        #rospy.loginfo("Goal sent, waiting for result...")
        # Waits for the server to finish performing the action.
        self.grasp_client.wait_for_result()

        # Get the result
        result = self.grasp_client.get_result()
        
        #   rospy.loginfo("Result received: success=%s, error message=%s", result.success, result.error)

        # Prints out the result of executing the action
        return result 
        

def main():
    io_client = PandaGripperControl()
    io_client.open()
    io_client.grasp()
    io_client.open()





if __name__ == '__main__':
    main()