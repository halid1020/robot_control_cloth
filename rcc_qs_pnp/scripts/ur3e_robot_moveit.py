#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit.planning import MoveItPy, PlanningComponent
from moveit.core.robot_state import RobotState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import CollisionObject
from moveit_msgs.msg import AttachedCollisionObject
from shape_msgs.msg import SolidPrimitive
from rclpy.logging import get_logger
import quaternion
import time
from geometry_msgs.msg import Pose

from utils import prepare_posestamped, MyPos

class UR3eRobotMoveit(Node):
    def __init__(self):
        super().__init__('ur3e_robot_moveit')
        self.logger = self.get_logger()
        self.logger.info('Start initializing')

        self.moveit = MoveItPy(node_name="moveit_py")
        
        # Get the planning component for the arm
        self.arm = self.moveit.get_planning_component("ur_manipulator")
        self.planning_scene_monitor = self.moveit.get_planning_scene_monitor()

        time.sleep(3.0)

        self.add_ground()
        self.add_ceiling()
        self.add_wall()
        self.add_gripper_collision()

        self.logger.info('Finished initializing')
    
    def add_ceiling(self):
        with self.planning_scene_monitor.read_write() as scene:
            collision_object = CollisionObject()
            collision_object.header.frame_id = "base_link"
            collision_object.id = "ceiling"

            box_pose = Pose()
            box_pose.position.x = 0.0
            box_pose.position.y = 0.0
            box_pose.position.z = 0.8

            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [2.0, 2.0, 0.01]

            collision_object.primitives.append(box)
            collision_object.primitive_poses.append(box_pose)
            collision_object.operation = CollisionObject.ADD

            scene.apply_collision_object(collision_object)
            scene.current_state.update()
        
    def add_wall(self):
        with self.planning_scene_monitor.read_write() as scene:
            collision_object = CollisionObject()
            collision_object.header.frame_id = "base_link"
            collision_object.id = "wall"

            box_pose = Pose()
            box_pose.position.x = 0.2
            box_pose.position.y = 0.0
            box_pose.position.z = 0.0

            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [0.01, 2.0, 2.0]

            collision_object.primitives.append(box)
            collision_object.primitive_poses.append(box_pose)
            collision_object.operation = CollisionObject.ADD

            scene.apply_collision_object(collision_object)
            scene.current_state.update()

    def add_ground(self):
        with self.planning_scene_monitor.read_write() as scene:
            collision_object = CollisionObject()
            collision_object.header.frame_id = "base_link"
            collision_object.id = "ground"

            box_pose = Pose()
            box_pose.position.x = 0.0
            box_pose.position.y = 0.0
            box_pose.position.z = 0.0

            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [2.0, 2.0, 0.01]

            collision_object.primitives.append(box)
            collision_object.primitive_poses.append(box_pose)
            collision_object.operation = CollisionObject.ADD

            scene.apply_collision_object(collision_object)
            scene.current_state.update()
    
    def add_gripper_collision(self):
        with self.planning_scene_monitor.read_write() as scene:
           
            #collision_object.header.frame_id = "tool0"  # Attach to tool0
            #collision_object.id = "collision_sphere"

            collision_object = CollisionObject()
            collision_object.header.frame_id = "tool0"
            collision_object.id = "gripper"

            box_pose = Pose()
            box_pose.position.x = 0.0  # Adjust as needed
            box_pose.position.y = 0.0  # Adjust as needed
            box_pose.position.z = 0.071  # Adjust as needed

            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [0.14, 0.14, 0.14]  # Radius of the sphere

            collision_object.primitives.append(box)
            collision_object.primitive_poses.append(box_pose)

            collision_object.operation = CollisionObject.ADD

            attached_collision_object = AttachedCollisionObject()
            attached_collision_object.link_name = "tool0"
            attached_collision_object.object = collision_object

            scene.process_attached_collision_object(attached_collision_object)
            scene.current_state.update()
        
    def go(self, pose=None, name=None, joint_states=None):
        with self.planning_scene_monitor.read_write() as scene:
            self.arm.set_start_state_to_current_state()
        
            if name is not None:
                self.logger.info(f'Set goal to {name}')
                self.arm.set_goal_state(configuration_name=name)

            elif pose is not None:
                self.logger.info(f'Set tool0 to {pose}')
                pose_stamped = prepare_posestamped(pose, frame_id='base_link')
                self.arm.set_goal_state(pose_stamped_msg=pose_stamped, pose_link='tool0')
            
            elif joint_states is not None:
                self.logger.info(f'Set joints to {joint_states}')
                robot_model = self.moveit.get_robot_model()
                robot_state = RobotState(robot_model)
                #robot_state = scene.current_state
                robot_state.joint_positions = joint_states
                #robot_state.update()
                self.arm.set_goal_state(robot_state=robot_state)

        self.plan_and_execute()
    
    def plan_and_execute(self):
        self.logger.info('Start Planning')
        plan_result = self.arm.plan()
        self.logger.info('Finished Planning')

        if plan_result:
            robot_trajectory = plan_result.trajectory
            self.logger.info('Start Execute')
            self.moveit.execute(robot_trajectory, controllers=[])
            self.logger.info('Finish Execute')
        else:
            self.logger.error("Planning failed")
        
        time.sleep(1)

def custom_pose():
    pose = MyPos(
        pose=[-0.12387, 0.20431, 0.61006],
        orien=[0.6708, -0.22, -0.66698,-0.23497]
    )
    return pose

def ready_joint_states():
    joints_states = {
        'shoulder_pan_joint': 3.143893241882324,
        'shoulder_lift_joint': -1.5633965845084568,
        'elbow_joint': 0.3434990088092249,
        'wrist_1_joint': -0.32642240942034917,
        'wrist_2_joint': -1.5632756392108362,
        'wrist_3_joint': 0.0,

    }
    return joints_states

def main(args=None):
    rclpy.init(args=args)
    robot = UR3eRobotMoveit()

    robot.go(name='up')
    robot.go(name='home')
    robot.go(pose=custom_pose())
    #robot.go(name='home')
    robot.go(joint_states=ready_joint_states())

    rclpy.shutdown()

if __name__ == '__main__':
    main()
