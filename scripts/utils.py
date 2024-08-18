import numpy as np
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
from collections import namedtuple
from dotmap import DotMap   
import yaml
import os
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
import cv2

MyPos = namedtuple('Pos', ['pose', 'orien'])

def quaternion_to_euler(q):
            r = R.from_quat(q)
            return r.as_euler('xyz', degrees=True)

def euler_to_quaternion(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    return r.as_quat()

def save_color(img, filename='color', directory="."):
    cv2.imwrite('{}/{}.png'.format(directory, filename), img)

def save_depth(depth, filename='depth', directory="."):
    depth = (depth - np.min(depth))/(np.max(depth) - np.min(depth))
    depth = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
    cv2.imwrite('{}/{}.png'.format(directory, filename), depth)

def normalise_quaterion(q):
    q = np.array(q)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Cannot normalize a zero quaternion.")
    return q / norm

def load_config(config_name):
    config_dir = os.path.join(get_package_share_directory('robot_control_cloth'), 'config')
    config_file_path = os.path.join(config_dir, f"{config_name}.yaml")
    with open(config_file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return DotMap(config_dict)

def prepare_posestamped(pose: MyPos, frame_id):
    posestamp = PoseStamped()
    posestamp.header.frame_id = frame_id
    posestamp.pose.position.x = pose.pose[0]
    posestamp.pose.position.y = pose.pose[1]
    posestamp.pose.position.z = pose.pose[2]
    posestamp.pose.orientation.x = pose.orien[0]
    posestamp.pose.orientation.y = pose.orien[1]
    posestamp.pose.orientation.z = pose.orien[2]
    posestamp.pose.orientation.w = pose.orien[3]
    return posestamp

def add_quaternions(quat1, quat2):
    """
    Multiplies two quaternions to combine their rotations.
    
    Parameters:
    quat1 (list or np.array): The first quaternion [w, x, y, z].
    quat2 (list or np.array): The second quaternion [w, x, y, z].
    
    Returns:
    np.array: The resulting quaternion after combining the rotations.
    """
    # Convert the input quaternions to Rotation objects
    r1 = R.from_quat(quat1)
    r2 = R.from_quat(quat2)
    
    # Multiply the quaternions
    combined_rotation = r1 * r2
    
    # Return the resulting quaternion
    return combined_rotation.as_quat()

# def camera2base(camera_pose, camera_orientation_quat, particles_camera):
#     r = R.from_quat(camera_orientation_quat)
#     rotation_matrix = r.as_matrix()
#     particles_camera = np.array(particles_camera)
#     particle_base = np.matmul(rotation_matrix, particles_camera) + np.array(camera_pose)
#     return particle_base

def camera2base(camera_pos: MyPos, particles_camera):
    camera_pose = camera_pos.pose
    camera_orientation_quat = camera_pos.orien
    r = R.from_quat(camera_orientation_quat)
    rotation_matrix = r.as_matrix()
    particles_camera = np.array(particles_camera)
    particle_base = np.matmul(rotation_matrix, particles_camera) + np.array(camera_pose)
    return particle_base

def pixel2camera(pixel_point, depth, intrinsic):
    pixel_point = [int(pixel_point[0]), int(pixel_point[1])]
    return rs.rs2_deproject_pixel_to_point(intrinsic, pixel_point, depth)

def camera2pixel(point_3d, intrinsic):
    pixel = rs.rs2_project_point_to_pixel(intrinsic, point_3d)
    return pixel

def pixel2base(pixel_point, camera_intrinsic, camera_pos:MyPos, depth):
    
    camera_p = pixel2camera(pixel_point, depth, camera_intrinsic)
    
    
    base_p = camera2base(camera_pos, camera_p)

    return base_p

def interpolate_positions(start_pos, target_pos, num_points=100):
    return np.linspace(start_pos, target_pos, num_points)

def bilinear_interpolation(x, y, x1, y1, x2, y2, q11, q21, q12, q22):
    """
    Perform bilinear interpolation.
    
    Parameters:
        x, y: Coordinates of the target point.
        x1, y1, x2, y2: Coordinates of the four corners.
        q11, q21, q12, q22: Values at the four corners.
        
    Returns:
        Interpolated value at the target point.
    """
    denom = (x2 - x1) * (y2 - y1)
    w11 = (x2 - x) * (y2 - y) / denom
    w21 = (x - x1) * (y2 - y) / denom
    w12 = (x2 - x) * (y - y1) / denom
    w22 = (x - x1) * (y - y1) / denom
    
    interpolated_value = q11 * w11 + q21 * w21 + q12 * w12 + q22 * w22
    return interpolated_value


def interpolate_image(height, width, corner_values):
    interpolated_image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            x = i/height
            y = j/width
            x1 = int(x)
            y1 = int(y)
            x2 = x1 + 1
            y2 = y1 + 1
            q11 = corner_values[(x1, y1)]
            q21 = corner_values[(x2, y1)]
            q12 = corner_values[(x1, y2)]
            q22 = corner_values[(x2, y2)]
            interpolated_image[i, j] = \
                bilinear_interpolation(x, y, x1, y1, x2, y2, q11, q21, q12, q22)
    return interpolated_image