#!/usr/bin/env python

import rospy
import moveit_commander
from moveit_commander import PlanningSceneInterface, RobotCommander
from moveit_commander.robot_trajectory import RobotTrajectory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject
from shape_msgs.msg import SolidPrimitive
import time
from utils import prepare_posestamped, MyPos

class FrankaRobotMoveit:
    def __init__(self):
        rospy.init_node('franka_robot_moveit', anonymous=True)
        
        # Initialize MoveIt commander
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("panda_arm")
        self.planning_frame = self.group.get_planning_frame()

        # Logger
        self.logger = rospy.loginfo

        # Add environment collisions
        self.add_ground()
        self.add_ceiling()
        self.add_wall()
        self.add_gripper_collision()

        self.logger('Finished initializing Franka Robot MoveIt interface')

    def add_ceiling(self):
        collision_object = CollisionObject()
        collision_object.header.frame_id = self.planning_frame
        collision_object.id = "ceiling"

        box_pose = PoseStamped()
        box_pose.pose.position.x = 0.0
        box_pose.pose.position.y = 0.0
        box_pose.pose.position.z = 1.5  # Adjust to ceiling height

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [2.0, 2.0, 0.01]

        self.scene.add_box(collision_object.id, box_pose, size=(2.0, 2.0, 0.01))
        self.logger('Ceiling added to the scene')

    def add_wall(self):
        collision_object = CollisionObject()
        collision_object.header.frame_id = self.planning_frame
        collision_object.id = "wall"

        box_pose = PoseStamped()
        box_pose.pose.position.x = 0.5
        box_pose.pose.position.y = 0.0
        box_pose.pose.position.z = 1.0  # Adjust as necessary for wall height

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.01, 2.0, 2.0]  # Wall dimensions

        self.scene.add_box(collision_object.id, box_pose, size=(0.01, 2.0, 2.0))
        self.logger('Wall added to the scene')

    def add_ground(self):
        collision_object = CollisionObject()
        collision_object.header.frame_id = self.planning_frame
        collision_object.id = "ground"

        box_pose = PoseStamped()
        box_pose.pose.position.x = 0.0
        box_pose.pose.position.y = 0.0
        box_pose.pose.position.z = 0.0

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [2.0, 2.0, 0.01]

        self.scene.add_box(collision_object.id, box_pose, size=(2.0, 2.0, 0.01))
        self.logger('Ground added to the scene')

    def add_gripper_collision(self):
        collision_object = CollisionObject()
        collision_object.header.frame_id = "panda_hand"
        collision_object.id = "gripper"

        box_pose = PoseStamped()
        box_pose.pose.position.x = 0.0
        box_pose.pose.position.y = 0.0
        box_pose.pose.position.z = 0.1  # Adjust as necessary

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.1, 0.1, 0.2]  # Gripper size

        self.scene.attach_box(collision_object.header.frame_id, collision_object.id, box_pose, size=(0.1, 0.1, 0.2))
        self.logger('Gripper collision object added')

    def go(self, pose=None, name=None, joint_states=None):
        if name:
            self.logger(f'Setting named target: {name}')
            self.group.set_named_target(name)
        elif pose:
            self.logger(f'Setting pose target: {pose}')
            pose_stamped = prepare_posestamped(pose, frame_id=self.planning_frame)
            self.group.set_pose_target(pose_stamped)
        elif joint_states:
            self.logger(f'Setting joint state target: {joint_states}')
            self.group.set_joint_value_target(joint_states)

        self.plan_and_execute()

    def plan_and_execute(self):
        self.logger('Planning...')
        plan = self.group.plan()

        if plan:
            self.logger('Executing...')
            self.group.execute(plan, wait=True)
            self.logger('Execution finished')
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
        'panda_joint1': 0.0,
        'panda_joint2': -0.785,
        'panda_joint3': 0.0,
        'panda_joint4': -2.356,
        'panda_joint5': 0.0,
        'panda_joint6': 1.571,
        'panda_joint7': 0.785,
    }
    return joints_states

def main():
    rospy.init_node('franka_robot_moveit')
    robot = FrankaRobotMoveit()

    robot.go(name='ready')
    robot.go(name='home')
    robot.go(pose=custom_pose())
    robot.go(joint_states=ready_joint_states())

    rospy.spin()

if __name__ == '__main__':
    main()
