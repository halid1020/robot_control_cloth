#!/usr/bin/env python

import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped, Pose
from moveit_commander import MoveGroupCommander, PlanningSceneInterface, RobotCommander
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject
from shape_msgs.msg import SolidPrimitive
import time
from utils import prepare_posestamped, MyPos

class PandaRobotMoveit:
    def __init__(self):
        """
        Initialize the Franka Panda robot using MoveIt.
        """
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('panda_robot_moveit', anonymous=True)

        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group = MoveGroupCommander("panda_arm")  # Panda arm's planning group

        # Set reference frame and tolerances
        self.group.set_pose_reference_frame("panda_link0")
        self.group.set_goal_position_tolerance(0.01)
        self.group.set_goal_orientation_tolerance(0.01)

        time.sleep(2.0)  # Allow some time for scene initialization

        # Add environment collision objects
        self.add_ground()
        self.add_ceiling()
        self.add_wall()
        self.add_gripper_collision()

        rospy.loginfo("Panda Robot MoveIt interface initialized.")

    def add_ceiling(self):
        """
        Adds a ceiling to the collision environment.
        """
        ceiling = CollisionObject()
        ceiling.header.frame_id = "panda_link0"
        ceiling.id = "ceiling"

        box_pose = Pose()
        box_pose.position.x = 0.0
        box_pose.position.y = 0.0
        box_pose.position.z = 1.5  # Adjust as needed

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [2.0, 2.0, 0.01]

        ceiling.primitives.append(box)
        ceiling.primitive_poses.append(box_pose)
        ceiling.operation = CollisionObject.ADD

        self.scene.apply_collision_object(ceiling)

    def add_wall(self):
        """
        Adds a wall to the collision environment.
        """
        wall = CollisionObject()
        wall.header.frame_id = "panda_link0"
        wall.id = "wall"

        box_pose = Pose()
        box_pose.position.x = 0.2  # Adjust as needed
        box_pose.position.y = 0.0
        box_pose.position.z = 0.0

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.01, 2.0, 2.0]  # Adjust dimensions as needed

        wall.primitives.append(box)
        wall.primitive_poses.append(box_pose)
        wall.operation = CollisionObject.ADD

        self.scene.apply_collision_object(wall)

    def add_ground(self):
        """
        Adds ground to the collision environment.
        """
        ground = CollisionObject()
        ground.header.frame_id = "panda_link0"
        ground.id = "ground"

        box_pose = Pose()
        box_pose.position.x = 0.0
        box_pose.position.y = 0.0
        box_pose.position.z = -0.02  # Slightly below base

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [2.0, 2.0, 0.01]

        ground.primitives.append(box)
        ground.primitive_poses.append(box_pose)
        ground.operation = CollisionObject.ADD

        self.scene.apply_collision_object(ground)

    def add_gripper_collision(self):
        """
        Adds a box around the gripper to account for collisions.
        """
        gripper_collision = CollisionObject()
        gripper_collision.header.frame_id = "panda_hand"
        gripper_collision.id = "gripper"

        box_pose = Pose()
        box_pose.position.x = 0.0  # Adjust based on the actual position of the gripper
        box_pose.position.y = 0.0
        box_pose.position.z = 0.08  # Adjust height as needed

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.14, 0.14, 0.14]  # Adjust as needed

        gripper_collision.primitives.append(box)
        gripper_collision.primitive_poses.append(box_pose)
        gripper_collision.operation = CollisionObject.ADD

        attached_collision = AttachedCollisionObject()
        attached_collision.link_name = "panda_hand"
        attached_collision.object = gripper_collision

        self.scene.apply_attached_collision_object(attached_collision)

    def go(self, pose=None, name=None, joint_states=None):
        """
        Move the robot arm to a target pose, named configuration, or joint states.
        :param pose: End-effector pose (MyPos object).
        :param name: Predefined robot configuration.
        :param joint_states: Dictionary of joint states for direct control.
        """
        if name:
            rospy.loginfo(f"Setting goal to named configuration: {name}")
            self.group.set_named_target(name)

        elif pose:
            rospy.loginfo(f"Setting goal to pose: {pose}")
            pose_stamped = prepare_posestamped(pose, frame_id='panda_link0')
            self.group.set_pose_target(pose_stamped)

        elif joint_states:
            rospy.loginfo(f"Setting joints to: {joint_states}")
            self.group.set_joint_value_target(joint_states)

        self.plan_and_execute()

    def plan_and_execute(self):
        """
        Plan and execute the motion for the current target.
        """
        rospy.loginfo("Starting planning.")
        plan = self.group.plan()

        if plan:
            rospy.loginfo("Executing the plan.")
            self.group.execute(plan, wait=True)
        else:
            rospy.logerr("Planning failed.")

        time.sleep(1)  # Small pause after execution

    def stop(self):
        """
        Stops the robot movement.
        """
        self.group.stop()

    def get_current_pose(self):
        """
        Retrieve the current pose of the end-effector.
        """
        return self.group.get_current_pose().pose

def custom_pose():
    """
    Return a custom pose for testing.
    """
    return MyPos(
        pose=[0.3, 0.0, 0.5],  # Example position
        orien=[0.0, 1.0, 0.0, 0.0]  # Example orientation as quaternion
    )

def ready_joint_states():
    """
    Return a set of joint states for the Panda robot's "ready" position.
    """
    return {
        'panda_joint1': 0.0,
        'panda_joint2': -0.785398,  # Adjust joint values as needed
        'panda_joint3': 0.0,
        'panda_joint4': -2.356194,
        'panda_joint5': 0.0,
        'panda_joint6': 1.570796,
        'panda_joint7': 0.785398
    }

def main():
    """
    Main function to run the robot control program.
    """
    rospy.init_node('panda_robot_moveit', anonymous=True)
    robot = PandaRobotMoveit()

    robot.go(name='ready')
    robot.go(pose=custom_pose())
    robot.go(joint_states=ready_joint_states())

    rospy.spin()

if __name__ == '__main__':
    main()
