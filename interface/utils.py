import time
import select
import sys

import matplotlib.pyplot as plt
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
from agent_arena.utilities.visual_utils \
    import draw_pick_and_place, filter_small_masks

import subprocess
import shlex
from scipy.ndimage import rotate, shift
from skimage.measure import label, regionprops

MyPos = namedtuple('Pos', ['pose', 'orien'])

MASK_THRESHOLD_V2=350000

def extract_square_crop_mask(mask):
    coords = np.argwhere(mask)
    y_center, x_center = coords.mean(axis=0).astype(int)
    side_length = min(mask.shape)
    x_start = max(x_center - side_length // 2, 0)
    y_start = max(y_center - side_length // 2, 0)
    x_start = min(x_start, mask.shape[1] - side_length)
    y_start = min(y_start, mask.shape[0] - side_length)
    return mask[y_start: y_start + side_length, x_start: x_start + side_length]
    #return (y_start, , x_start, x_start + side_length)

def calculate_iou(mask1, mask2):
    if mask1.shape[0] > 128:
        mask1 = cv2.resize(mask1.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
        mask1 = (mask1 > 0.5).astype(bool)
        

    if mask2.shape[0] > 128:
        mask2 = cv2.resize(mask2.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
        mask2 = (mask2 > 0.5).astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def get_max_IoU(mask1, mask2):
    """
    Calculate the maximum IoU between two binary mask images,
    allowing for rotation and translation of mask1.
    
    :param mask1: First binary mask (numpy array)
    :param mask2: Second binary mask (numpy array)
    :return: Tuple of (Maximum IoU value, Matched mask)
    """

    ## if mask is rectangular, make it square by padding
    if mask1.shape[0] > mask1.shape[1]:
        pad = (mask1.shape[0] - mask1.shape[1]) // 2
        mask1 = np.pad(mask1, ((0, 0), (pad, pad)), mode='constant')
    elif mask1.shape[1] > mask1.shape[0]:
        pad = (mask1.shape[1] - mask1.shape[0]) // 2
        mask1 = np.pad(mask1, ((pad, pad), (0, 0)), mode='constant')
    
    if mask2.shape[0] > mask2.shape[1]:
        pad = (mask2.shape[0] - mask2.shape[1]) // 2
        mask2 = np.pad(mask2, ((0, 0), (pad, pad)), mode='constant')
    elif mask2.shape[1] > mask2.shape[0]:
        pad = (mask2.shape[1] - mask2.shape[0]) // 2
        mask2 = np.pad(mask2, ((pad, pad), (0, 0)), mode='constant')
    
    # print('mask1 shape:', mask1.shape)
    # print('mask2 shape:', mask2.shape)

    # if resolution above 128, we need to resize the mask to 128
    if mask1.shape[0] > 128:
        mask1 = cv2.resize(mask1.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
        mask1 = (mask1 > 0.5).astype(np.uint8)
        

    if mask2.shape[0] > 128:
        mask2 = cv2.resize(mask2.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
        mask2 = (mask2 > 0.5).astype(np.uint8)
    
    

    # Get the properties of mask2
    props = regionprops(label(mask2))[0]
    center_y, center_x = props.centroid

    max_iou = 0
    best_mask = None
    
    # Define rotation angles to try
    angles = range(0, 360, 1)  # Rotate from 0 to 350 degrees in 10-degree steps
    
    for angle in angles:
        # Rotate mask1
        rotated_mask = rotate(mask1, angle, reshape=False)

        # if the mask is blank, skip
        if np.sum(rotated_mask) == 0:
            continue
        
        # Get properties of rotated mask
        rotated_props = regionprops(label(rotated_mask))[0]
        rotated_center_y, rotated_center_x = rotated_props.centroid
        
        # Calculate translation
        dy = center_y - rotated_center_y
        dx = center_x - rotated_center_x
        
        # Translate rotated mask
        translated_mask = shift(rotated_mask, (dy, dx))
        
        #translated_mask = (translated_mask > 0.1).astype(int)
        # Calculate IoU
        iou = calculate_iou(translated_mask, mask2)
        
        # Update max_iou and best_mask if necessary
        if iou > max_iou:
            max_iou = iou
            best_mask = translated_mask

    # Ensure the best_mask is binary
    if best_mask is None:
        return 0, None
    
    best_mask = (best_mask > 0.5).astype(int)

    return max_iou, best_mask

def get_IoU(mask1, mask2):
    """
    Calculate the maximum IoU between two binary mask images,
    allowing for rotation and translation of mask1.
    
    :param mask1: First binary mask (numpy array)
    :param mask2: Second binary mask (numpy array)
    :return: Tuple of (Maximum IoU value, Matched mask)
    """
    
    def calculate_iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0

    # Get the properties of mask2
    props = regionprops(label(mask2))[0]
    center_y, center_x = props.centroid

    max_iou = 0
    best_mask = None
    
    # Define rotation angles to try
    angles = range(0, 360, 10)  # Rotate from 0 to 350 degrees in 10-degree steps
    
    for angle in angles:
        # Rotate mask1
        rotated_mask = rotate(mask1, angle, reshape=False)
        
        # Get properties of rotated mask
        rotated_props = regionprops(label(rotated_mask))[0]
        rotated_center_y, rotated_center_x = rotated_props.centroid
        
        # Calculate translation
        dy = center_y - rotated_center_y
        dx = center_x - rotated_center_x
        
        # Translate rotated mask
        translated_mask = shift(rotated_mask, (dy, dx))
        
        # Calculate IoU
        iou = calculate_iou(translated_mask, mask2)
        
        # Update max_iou and best_mask if necessary
        if iou > max_iou:
            max_iou = iou
            best_mask = translated_mask

    # Ensure the best_mask is binary
    best_mask = (best_mask > 0.5).astype(int)

    return max_iou, best_mask

def get_mask_v2(mask_generator, rgb, mask_treshold=160000):
        """
        Generate a mask for the given RGB image that is most different from the background.
        
        Parameters:
        - rgb: A NumPy array representing the RGB image.
        
        Returns:
        - A binary mask as a NumPy array with the same height and width as the input image.
        """
        # Generate potential masks from the mask generator
        results = mask_generator.generate(rgb)
        
        final_mask = None
        max_color_difference = 0
        print('Processing mask results...')
        save_color(rgb, 'rgb', './tmp')
        mask_data = []

        # Iterate over each generated mask result
        for i, result in enumerate(results):
            segmentation_mask = result['segmentation']
            mask_shape = rgb.shape[:2]

            ## count no mask corner of the mask
            margin = 5 #5
            mask_corner_value = 1.0*segmentation_mask[margin, margin] + 1.0*segmentation_mask[margin, -margin] + \
                                1.0*segmentation_mask[-margin, margin] + 1.0*segmentation_mask[-margin, -margin]
            
            

            #print('mask corner value', mask_corner_value)
            # Ensure the mask is in the correct format
            orginal_mask = segmentation_mask.copy()
            segmentation_mask = segmentation_mask.astype(np.uint8) * 255
            
            # Calculate the masked region and the background region
            masked_region = cv2.bitwise_and(rgb, rgb, mask=segmentation_mask)
            background_region = cv2.bitwise_and(rgb, rgb, mask=cv2.bitwise_not(segmentation_mask))
            
            # Calculate the average color of the masked region
            masked_pixels = masked_region[segmentation_mask == 255]
            if masked_pixels.size == 0:
                continue
            avg_masked_color = np.mean(masked_pixels, axis=0)
            
            # Calculate the average color of the background region
            background_pixels = background_region[segmentation_mask == 0]
            if background_pixels.size == 0:
                continue
            avg_background_color = np.mean(background_pixels, axis=0)
            
            # Calculate the Euclidean distance between the average colors
            color_difference = np.linalg.norm(avg_masked_color - avg_background_color)
            #print(f'color difference {i} color_difference {color_difference}')
            #save_mask(orginal_mask, f'mask_candidate_{i}')
            
            # Select the mask with the maximum color difference from the background
            #mask_region_size = np.sum(segmentation_mask == 255)
            

            if mask_corner_value >= 2:
                # if the mask has more than 2 corners, the flip the value
                orginal_mask = 1 - orginal_mask
            
            mask_region_size = np.sum(orginal_mask == 1)
            if mask_region_size > mask_treshold:
                continue

            mask_data.append({
                'mask': orginal_mask,
                'color_difference': color_difference,
                'mask_region_size': mask_region_size,
            })
        
        top_num = 3
        top_5_masks = sorted(mask_data, key=lambda x: x['color_difference'], reverse=True)[:top_num]
        final_mask_data = sorted(top_5_masks, key=lambda x: x['mask_region_size'], reverse=True)[0]
        final_mask = final_mask_data['mask']

        ## make the margine of the final mask to be 0
        margin = 5
        final_mask[:margin, :] = 0
        final_mask[-margin:, :] = 0
        final_mask[:, :margin] = 0
        final_mask[:, -margin:] = 0

        ## print the average color of the mask background
        masked_region = np.expand_dims(final_mask, -1) * rgb
        background_region = (1 - np.expand_dims(final_mask, -1)) * rgb
        masked_pixels = masked_region[final_mask == 1]
        avg_masked_color = np.mean(masked_pixels, axis=0)
        background_pixels = background_region[final_mask == 0]
        avg_background_color = np.mean(background_pixels, axis=0)
        #print(f'avg_masked_color {avg_masked_color} avg_background_color {avg_background_color}')
        
        #save_mask(final_mask, 'final_mask')
        #print('Final mask generated.')

        return final_mask

def get_mask_v1(mask_generator, rgb):
    ## similar as get_mask_v2, but only get the top_1 mask
    results = mask_generator.generate(rgb)
        
    final_mask = None
    max_color_difference = 0
    #print('Processing mask results...')
    save_color(rgb, 'rgb', './tmp')
    mask_data = []

    # Iterate over each generated mask result
    for i, result in enumerate(results):
        segmentation_mask = result['segmentation']
        mask_shape = rgb.shape[:2]

        ## count no mask corner of the mask
        margin = 5
        mask_corner_value = 1.0*segmentation_mask[margin, margin] + 1.0*segmentation_mask[margin, -margin] + \
                            1.0*segmentation_mask[-margin, margin] + 1.0*segmentation_mask[-margin, -margin]
        
        

        #print('mask corner value', mask_corner_value)
        # Ensure the mask is in the correct format
        orginal_mask = segmentation_mask.copy()
        segmentation_mask = segmentation_mask.astype(np.uint8) * 255
        
        # Calculate the masked region and the background region
        masked_region = cv2.bitwise_and(rgb, rgb, mask=segmentation_mask)
        background_region = cv2.bitwise_and(rgb, rgb, mask=cv2.bitwise_not(segmentation_mask))
        
        # Calculate the average color of the masked region
        masked_pixels = masked_region[segmentation_mask == 255]
        if masked_pixels.size == 0:
            continue
        avg_masked_color = np.mean(masked_pixels, axis=0)
        
        # Calculate the average color of the background region
        background_pixels = background_region[segmentation_mask == 0]
        if background_pixels.size == 0:
            continue
        avg_background_color = np.mean(background_pixels, axis=0)
        
        # Calculate the Euclidean distance between the average colors
        color_difference = np.linalg.norm(avg_masked_color - avg_background_color)
        #print(f'color difference {i} color_difference {color_difference}')
        #save_mask(orginal_mask, f'mask_candidate_{i}')
        
        # Select the mask with the maximum color difference from the background
        #mask_region_size = np.sum(segmentation_mask == 255)
        

        if mask_corner_value >= 2:
            # if the mask has more than 2 corners, the flip the value
            orginal_mask = 1 - orginal_mask
        
        mask_region_size = np.sum(orginal_mask == 1)
        if mask_region_size > 160000:
            continue

        mask_data.append({
            'mask': orginal_mask,
            'color_difference': color_difference,
            'mask_region_size': mask_region_size,
        })
    
    final_mask = sorted(mask_data, key=lambda x: x['color_difference'], reverse=True)[0]['mask']
    
    #save_mask(final_mask, 'final_mask')
    print('Final mask generated.')

    return final_mask

def get_mask_v0(rgb):
    ### segment the region that is under the threshold
    ## rgb in shape H, W, 3

    threshold = [105, 105, 105]
    mask = np.ones((rgb.shape[0], rgb.shape[1]))
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            if np.all(rgb[i, j] < threshold):
                mask[i, j] = 0
    return mask


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
    sam = sam_model_registry[MODEL_TYPE](
        checkpoint=os.path.join(
            os.environ['ROBOT_CONTROL_CLOTH_DIR'], 'interface', 'sam_vit_h_4b8939.pth'))

    sam.to(device=DEVICE)
    return SamAutomaticMaskGenerator(sam)


# def get_orientation(point, mask):
#     mask = (mask > 0).astype(np.uint8)
    
#     # Compute gradients
#     grad_y = sobel(mask, axis=0)
#     grad_x = sobel(mask, axis=1)
    
#     x, y = point
    
#     # Calculate orientation
#     gx = grad_x[y, x]
#     gy = grad_y[y, x]
#     orientation_rad = np.arctan2(gy, gx)
    
#     # Convert orientation to degrees
#     orientation_deg = np.degrees(orientation_rad)
    
#     # Normalize to range [0, 360)
#     orientation_deg = (orientation_deg - 90 + 360) % 360
    
#     return orientation_deg

def get_orientation(point, mask, window_size=21):
    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)
    
    # Extract a window around the point
    x, y = point
    half_window = window_size // 2
    window = mask[max(0, y-half_window):min(mask.shape[0], y+half_window+1),
                  max(0, x-half_window):min(mask.shape[1], x+half_window+1)]
    
    # Compute gradients
    grad_y = sobel(window, axis=0)
    grad_x = sobel(window, axis=1)

    if np.all(grad_x == 0) and np.all(grad_y == 0):
        return 0.0  # Return 0 degrees if no gradient
    
    # Compute orientation using Principal Component Analysis (PCA)
    cov = np.cov(np.array([grad_x.ravel(), grad_y.ravel()]))
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # The eigenvector corresponding to the smaller eigenvalue 
    # gives the direction perpendicular to the dominant edge
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate orientation
    orientation_rad = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
    
    # Convert orientation to degrees
    orientation_deg = np.degrees(orientation_rad)
    
    # Normalize to range [0, 360)
    orientation_deg = (orientation_deg + 360) % 360 - 180
    
    return orientation_deg

def refine_orientation(point, mask, initial_orientation, refinement_angle=10, num_steps=21):
    x, y = point
    best_orientation = initial_orientation
    max_sum = 0
    
    for angle in np.linspace(initial_orientation - refinement_angle, 
                             initial_orientation + refinement_angle, num_steps):
        line_points = get_line_points(point, angle, length=20)
        line_sum = sum(mask[py, px] for px, py in line_points if 0 <= py < mask.shape[0] and 0 <= px < mask.shape[1])
        
        if line_sum > max_sum:
            max_sum = line_sum
            best_orientation = angle
    
    return best_orientation

def get_line_points(start, angle, length):
    x, y = start
    rad = np.radians(angle)
    return [(int(x + i * np.cos(rad)), int(y + i * np.sin(rad))) for i in range(length)]

def improved_get_orientation(point, mask):
    initial_orientation = get_orientation(point, mask)
    refined_orientation = refine_orientation(point, mask, initial_orientation)
    return refined_orientation

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
    elif encoding == "8UC1":
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image

def save_color(img, filename='color', directory="."):
    #print('save color img' , img.shape)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('{}/{}.png'.format(directory, filename), img_bgr)

def save_depth(depth, filename='depth', directory=".", colour=False, remap=True):
    if remap:
        depth = (depth - np.min(depth))/(np.max(depth) - np.min(depth))
    
    if colour:
        depth = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
    else:
        depth = np.uint8(255 * depth)
    
    cv2.imwrite('{}/{}.png'.format(directory, filename), depth)

def save_depth_distribution(depth_image, filename='depth_distribution', directory="."):
    """
    Saves a histogram of depth values from a depth image to a PNG file.

    Parameters:
        depth_image (numpy.ndarray): A 2D array representing the depth image.
        filename (str): The name of the output file (without extension). Default is 'depth_distribution'.
        directory (str): The directory where the file will be saved. Default is the current directory (".").
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Compute histogram with 100 bins
    depth_image = depth_image.flatten()
    hist, bins = np.histogram(depth_image, bins=100)

    
    # Compute bin centers for better alignment in the bar plot
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot and save histogram
    plt.figure()
    plt.bar(bin_centers, hist, width=np.diff(bins), align='center')
    plt.xlabel('Depth Value')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.title('Depth Distribution')
    plt.savefig(f'{directory}/{filename}.png')
    plt.close()


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