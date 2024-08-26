import time
import select
import sys

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
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import distance_transform_edt, sobel
import math
import torch
import signal
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from segment_anything import sam_model_registry
from agent_arena.utilities.visualisation_utils import draw_pick_and_place, filter_small_masks

import subprocess
import shlex

MyPos = namedtuple('Pos', ['pose', 'orien'])


def start_ffmpeg_recording(output_file, device='/dev/video6', resolution='1920x1080', framerate=30):
    """
    Start FFmpeg recording in the background.
    
    :param output_file: Name of the output file (e.g., 'output.mp4')
    :param device: Input device (default: '/dev/video6')
    :param resolution: Video resolution (default: '1920x1080')
    :param framerate: Frame rate (default: 30)
    :return: Subprocess object
    """
    if os.path.exists(output_file):
        print('Remove old record!')
        os.remove(output_file)
    command = f"ffmpeg -f v4l2 -video_size {resolution} -framerate {framerate} -i {device} " \
              f"-c:v libx264 -crf 23 -preset medium -bufsize 5M {output_file}"
    
    # Split the command string into a list of arguments
    args = shlex.split(command)
    
    # Start the FFmpeg process
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    print(f"FFmpeg recording started. Output file: {output_file}")
    return process

def wait_for_user_input(timeout=1):
    """
    Wait for user input for a specified timeout period.
    Returns True if input is received, False otherwise.
    """
    print("\n\n[User Attention!] Please Press [Enter] to finish, or wait 1 second to continue...\n\n")
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        os.read(sys.stdin.fileno(), 1024)
        #sys.stdin.readline()
        return True
    print("\n\n[User Attention!] Continue to next step...\n\n")
    return False

def get_mask_generator():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device {}'.format(DEVICE))

    ### Masking Model Macros ###
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint='sam_vit_h_4b8939.pth')
    sam.to(device=DEVICE)
    return SamAutomaticMaskGenerator(sam)


def get_orientation(point, mask):
    mask = (mask > 0).astype(np.uint8)
    
    # Compute gradients
    grad_y = sobel(mask, axis=0)
    grad_x = sobel(mask, axis=1)
    
    x, y = point
    
    # Calculate orientation
    gx = grad_x[y, x]
    gy = grad_y[y, x]
    orientation_rad = np.arctan2(gy, gx)
    
    # Convert orientation to degrees
    orientation_deg = np.degrees(orientation_rad)
    
    # Normalize to range [0, 360)
    orientation_deg = (orientation_deg - 90 + 360) % 360
    
    return orientation_deg


def stop_ffmpeg_recording(process):
    """
    Stop the FFmpeg recording process gracefully.
    
    :param process: Subprocess object returned by start_ffmpeg_recording
    :return: None
    """
    #time.sleep(1)
    process.terminate()
    #os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        # # Wait for the process to finish (with a timeout)
        # try:
        #     process.wait(timeout=10)
        #     print("FFmpeg recording stopped successfully.")
        # except subprocess.TimeoutExpired:
        #     print("FFmpeg didn't stop gracefully. Forcing termination...")
        #     process.terminate()
        #     try:
        #         process.wait(timeout=5)
        #     except subprocess.TimeoutExpired:
        #         print("FFmpeg still didn't terminate. Killing the process...")
        #         process.kill()
        
        # # Ensure the process is terminated
        # if process.poll() is None:
        #     print("Failed to stop FFmpeg recording.")
        # else:
        #     print("FFmpeg recording has been stopped.")
    # else:
    #     print("FFmpeg recording was not running.")

def visualize_points_and_orientations(mask, points_with_orientations, line_length=10):
    # Ensure mask is 8-bit single-channel
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    
    # Convert to BGR
    vis_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    for x, y, orientation in points_with_orientations:
        rad = orientation*math.pi/180
        cv2.circle(vis_mask, (int(x), int(y)), 3, (0, 255, 0), -1)
        end_x = int(x + line_length * np.cos(rad))
        end_y = int(y + line_length * np.sin(rad))
        cv2.line(vis_mask, (int(x), int(y)), (end_x, end_y), (0, 0, 255), 2)
    
    cv2.imshow('Points and Orientations', vis_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def adjust_points(points, mask, min_distance=3):
    """
    Adjust points to be at least min_distance pixels away from the mask border.
    
    :param points: List of (x, y) coordinates
    :param mask: 2D numpy array where 0 is background and 1 is foreground
    :param min_distance: Minimum distance from the border (default: 2)
    :return: List of adjusted (x, y) coordinates
    """
    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)
    
    # Compute distance transform
    dist_transform = distance_transform_edt(mask)
    
    # Create a new mask where pixels < min_distance from border are 0
    eroded_mask = (dist_transform >= min_distance).astype(np.uint8)
    
    adjusted_points = []
    for x, y in points:
        if eroded_mask[y, x] == 0:  # If point is too close to border
            # Find the nearest valid point
            y_indices, x_indices = np.where(eroded_mask == 1)
            distances = np.sqrt((x - x_indices)**2 + (y - y_indices)**2)
            nearest_index = np.argmin(distances)
            new_x, new_y = x_indices[nearest_index], y_indices[nearest_index]
            adjusted_points.append((new_x, new_y))
        else:
            adjusted_points.append((x, y))
    
    return adjusted_points, eroded_mask

def imgmsg_to_cv2_custom(img_msg, encoding="bgr8"):
    # Get the image dimensions
    height = img_msg.height
    width = img_msg.width
    #channels = 3  # Assuming BGR format

    # Extract the raw data
    if encoding == "64FC1":
        data = np.frombuffer(img_msg.data, dtype=np.float64).reshape((height, width, -1))
        return data
    
    data = np.frombuffer(img_msg.data, dtype=np.uint8)

    # Reshape the data to match the image dimensions
    image = data.reshape((height, width, -1))

    # If the encoding is not BGR, convert it using OpenCV
    if encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif encoding == "mono8":
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image

def save_color(img, filename='color', directory="."):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('{}/{}.png'.format(directory, filename), img_bgr)

def save_depth(depth, filename='depth', directory=".", colour=False):
    depth = (depth - np.min(depth))/(np.max(depth) - np.min(depth))
    if colour:
        depth = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
    else:
        depth = np.uint8(255 * depth)
    cv2.imwrite('{}/{}.png'.format(directory, filename), depth)

def save_mask(mask, filename='mask', directory="."):
    mask = mask.astype(np.int8)*255
    cv2.imwrite('{}/{}.png'.format(directory, filename), mask)

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